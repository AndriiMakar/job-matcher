import ast
import os

import psycopg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from phoenix.client.experiments import run_experiment
from pipeline import retrieve
from gold_set import retrieval_ds
from metrics import recall_at_k_value

K = 10


# ============================================================================ #
# Experiment 1 — REAL retriever (embedding / vector search)
# ============================================================================ #
def retrieval_task(input: dict) -> dict:
    jobs = retrieve(input["resume_text"], k=K)
    return {"retrieved_job_ids": [str(j["job_id"]) for j in jobs]}


# Evaluator: Phoenix binds `output` (what the task returned) and `expected`
# (the example's output_keys — here, the labeled relevant_job_ids).
def _as_id_list(v):
    """Phoenix returns a list-valued dataset column as a stringified list
    (e.g. "['78', '32']"). Parse it back to a real list so we compare IDs, not
    individual characters. A real list passes straight through.
    """
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = [v]
    return [str(x) for x in v]


def recall_at_k(output: dict, expected: dict) -> float | None:
    return recall_at_k_value(output["retrieved_job_ids"], _as_id_list(expected["relevant_job_ids"]))


vector_experiment = run_experiment(
    dataset=retrieval_ds,
    task=retrieval_task,
    evaluators=[recall_at_k],
    experiment_name=f"vector-retrieval-k{K}",
)


# ============================================================================ #
# Experiment 2 — keyword / TF-IDF baseline (the "dumb" retriever to beat)
# Same dataset, same evaluator — only the retrieval method differs.
# ============================================================================ #
def _load_corpus() -> tuple[list[str], list[str]]:
    """Pull every job's text from the SAME corpus the vector search uses."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute(
            "SELECT id, title || ' ' || company || ' ' || description FROM jobs"
        ).fetchall()
    return [str(r[0]) for r in rows], [r[1] or "" for r in rows]


_JOB_IDS, _JOB_TEXTS = _load_corpus()
_VECTORIZER = TfidfVectorizer(stop_words="english")
_JOB_MATRIX = _VECTORIZER.fit_transform(_JOB_TEXTS)        # rows are L2-normalized


def keyword_task(input: dict) -> dict:
    qvec = _VECTORIZER.transform([input["resume_text"]])
    sims = linear_kernel(qvec, _JOB_MATRIX)[0]            # = cosine, given the L2 norm
    top = sims.argsort()[::-1][:K]
    return {"retrieved_job_ids": [_JOB_IDS[i] for i in top]}


baseline_experiment = run_experiment(
    dataset=retrieval_ds,
    task=keyword_task,
    evaluators=[recall_at_k],
    experiment_name=f"keyword-retrieval-k{K}",
)