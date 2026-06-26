import os
import anthropic
import voyageai
import psycopg
from dotenv import load_dotenv
from opentelemetry import trace as otel_trace
from phoenix.otel import register

load_dotenv()  # pull DATABASE_URL + API keys from .env before anything reads them

# --- 1. Register the tracer ONCE, at import time -------------------------------
tracer_provider = register(
    project_name="job-matcher",
    auto_instrument=True,          # picks up the installed openinference-* instrumentors
    endpoint=f"{os.environ['PHOENIX_COLLECTOR_ENDPOINT']}/v1/traces",
    batch=True,
)
tracer = tracer_provider.get_tracer(__name__)

anthropic_client = anthropic.Anthropic()
voyage = voyageai.Client()

EMBED_MODEL = "voyage-4"              # current Voyage generalist (voyage-4-lite to cut cost)
MATCH_MODEL = "claude-sonnet-4-6"     # rerank + judge
PREFILTER_MODEL = "claude-haiku-4-5-20251001"  # optional cheap pass


# --- 2. Each pipeline stage is a traced function -------------------------------
@tracer.chain
def embed_query(resume_text: str) -> list[float]:
    # input_type="query" matters for retrieval quality (vs "document" at index time)
    return voyage.embed([resume_text], model=EMBED_MODEL, input_type="query").embeddings[0]


@tracer.chain
def retrieve(resume_text: str, k: int = 10) -> list[dict]:
    """Vector-search the k nearest job candidates for a resume, closest first.

    Each returned dict carries job fields plus `score` (1 − cosine distance,
    so higher means a closer match).
    """
    qvec = embed_query(resume_text)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute(
            # `<=>` = cosine distance in pgvector; lower is closer
            "SELECT id, title, company, description, (embedding <=> %s::vector) AS distance "
            "FROM jobs ORDER BY embedding <=> %s::vector LIMIT %s",
            (qvec, qvec, k),
        ).fetchall()

    jobs = [
        {"job_id": r[0], "title": r[1], "company": r[2], "description": r[3],
         "score": 1.0 - r[4]}
        for r in rows
    ]

    # The one manual touch: record retrieval-specific attributes on the current span.
    span = otel_trace.get_current_span()
    span.set_attribute("retrieval.k", k)
    span.set_attribute("retrieval.job_ids", [j["job_id"] for j in jobs])
    span.set_attribute("retrieval.scores", [j["score"] for j in jobs])
    return jobs
    # For the dedicated "Retrieval" UI panel with ranked docs, set OpenInference retriever
    # semantic conventions instead (openinference.semconv.trace). Custom attrs are fine for v0.


@tracer.chain
def rerank(resume_text: str, jobs: list[dict], top_n: int = 5) -> list[dict]:
    """Reorder retrieved candidates by LLM judgment, returning the top_n best."""
    listing = "\n".join(f"[{j['job_id']}] {j['title']} @ {j['company']}" for j in jobs)
    msg = anthropic_client.messages.create(
        model=MATCH_MODEL, max_tokens=512,
        messages=[{"role": "user", "content":
            f"Resume:\n{resume_text}\n\nJobs:\n{listing}\n\n"
            f"Return the {top_n} best-matching job_ids, most relevant first, comma-separated."}],
    )  # <-- auto-instrumented: tokens/cost/latency captured as a child LLM span
    order = [s.strip() for s in msg.content[0].text.split(",")]
    by_id = {str(j["job_id"]): j for j in jobs}
    # Keep only IDs that were actually in the candidate set — guards against the
    # model returning a hallucinated or malformed job_id.
    return [by_id[i] for i in order if i in by_id][:top_n]


@tracer.chain
def judge_match(resume_text: str, job_text: str) -> dict:
    """Score a SINGLE (resume, job) pair → {"score": int, "rationale": str}.

    temperature=0 → deterministic, reproducible scoring. The rubric + band anchors
    live in the prompt so the judge's numbers land on the same good/marginal/bad
    scale the gold set was labeled with.
    """
    import json
    import re

    rubric = (
        "Scoring rubric — apply it strictly and score on THIS scale:\n"
        "- 70-100 = good: right role AND seniority, must-have skills present, domain adjacent.\n"
        "- 40-69 = marginal: partial fit — real overlap AND a real gap.\n"
        "- 0-39 = bad: wrong role or wrong core stack.\n"
        "Weigh role and seniority fit and must-have skills first; domain and location are secondary. "
        "Tech-stack overlap does NOT rescue a fundamentally different role (e.g. a sales or non-eng role)."
    )
    msg = anthropic_client.messages.create(
        model=MATCH_MODEL, max_tokens=512, temperature=0,
        system="You are a precise hiring evaluator. Reply with ONLY a JSON object — no prose, no markdown.",
        messages=[
            {"role": "user", "content":
                f"{rubric}\n\n"
                "Score how well this RESUME matches this JOB from 0-100 using the rubric above, then justify "
                "in 2 sentences citing ONLY skills present in the resume.\n\n"
                f"RESUME:\n{resume_text}\n\nJOB:\n{job_text}\n\n"
                'Respond as JSON: {"score": <int>, "rationale": "<text>"}'},
        ],
    )
    text = msg.content[0].text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge returned no JSON object: {text!r}")
    return json.loads(match.group(0))


@tracer.chain
def match_pipeline(resume_text: str, k: int = 10, top_n: int = 5) -> list[dict]:
    """Two-stage match: cheap vector retrieval of k candidates, then LLM rerank to top_n.

    Retrieval runs over the whole corpus cheaply and sets the candidate ceiling; the
    expensive LLM call then runs only on those k, never the full table. k trades recall
    against cost; top_n is how many matches to surface.
    """
    candidates = retrieve(resume_text, k=k)
    return rerank(resume_text, candidates, top_n=top_n)