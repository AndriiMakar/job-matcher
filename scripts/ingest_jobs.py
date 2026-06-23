"""Ingest remote-job postings into pgvector for the matcher's retrieval layer.

Fetches from RemoteOK, Remotive (JSON APIs) and WeWorkRemotely (RSS), cleans the
HTML descriptions, embeds each posting with Voyage (the *document* side), and
upserts into the `jobs` table that pipeline.retrieve() searches. Idempotent —
re-running won't create duplicates.

    python scripts/ingest_jobs.py --limit 400 --search engineer

Requires DATABASE_URL and VOYAGE_API_KEY in the environment (or a .env file).
Attribution: RemoteOK, Remotive, and WeWorkRemotely all require linking back to
the source posting, which is why each job's canonical `url` is stored.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import psycopg
import requests
import voyageai
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

EMBED_MODEL = "voyage-4"      # MUST match the model pipeline.embed_query uses

# Embedding throughput. Defaults assume a Voyage account WITH a payment method.
# Free tier (no card) is limited to 3 RPM / 10K TPM — if you hit RateLimitError,
# use the free-tier values noted on each line.
EMBED_BATCH = 100        # free tier: 8
DESC_CHARS = 4000        # free tier: 1500
EMBED_SLEEP = 0          # free tier: 21  (seconds between batches, to stay under 3 RPM)

HTTP_TIMEOUT = 30
USER_AGENT = "job-matcher/0.1 (personal portfolio project)"

REMOTEOK_URL = "https://remoteok.com/api"
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
WEWORKREMOTELY_URL = "https://weworkremotely.com/remote-jobs.rss"

voyage = voyageai.Client()  # reads VOYAGE_API_KEY from the environment


# --------------------------------------------------------------------------- #
# Fetch + normalize
# --------------------------------------------------------------------------- #
def _clean(text: str | None) -> str:
    """Strip HTML tags and unescape entities — descriptions arrive as raw HTML."""
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(no_tags)).strip()


def fetch_remoteok(search: str | None) -> list[dict]:
    """Return normalized jobs from RemoteOK's public JSON API.

    Element 0 of the response is a legal notice, not a job, so it's skipped.
    A User-Agent header is required or the endpoint may reject the request.
    """
    resp = requests.get(
        REMOTEOK_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json()[1:]  # drop the legal-notice element
    jobs = []
    for r in rows:
        title = r.get("position") or r.get("title") or ""
        tags = r.get("tags") or []
        if search and search.lower() not in f"{title} {' '.join(tags)}".lower():
            continue
        jobs.append({
            "source": "remoteok",
            "external_id": str(r.get("id")),
            "title": title,
            "company": r.get("company") or "",
            "description": _clean(r.get("description")),
            "url": r.get("url") or "",
            "tags": tags,
        })
    return jobs


def fetch_remotive(category: str, search: str | None) -> list[dict]:
    """Return normalized jobs from Remotive's public API (jobs live under the 'jobs' key)."""
    params: dict[str, str] = {}
    if category:
        params["category"] = category
    if search:
        params["search"] = search
    resp = requests.get(
        REMOTIVE_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    rows = resp.json().get("jobs", [])
    return [{
        "source": "remotive",
        "external_id": str(r.get("id")),
        "title": r.get("title") or "",
        "company": r.get("company_name") or "",
        "description": _clean(r.get("description")),
        "url": r.get("url") or "",
        "tags": r.get("tags") or [],
    } for r in rows]


def fetch_weworkremotely(search: str | None) -> list[dict]:
    """Return normalized jobs from WeWorkRemotely's RSS feed.

    WWR has no JSON API — it publishes RSS (XML), so this adapter parses XML
    instead of JSON, and item titles arrive as "Company: Position" (split on the
    first colon). Different input shape, identical normalized output: that is the
    entire point of the adapter pattern.
    """
    resp = requests.get(
        WEWORKREMOTELY_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    jobs = []
    for item in root.iterfind(".//item"):
        title_raw = (item.findtext("title") or "").strip()
        company, position = title_raw.split(":", 1) if ":" in title_raw else ("", title_raw)
        tags = [c.text for c in item.findall("category") if c.text]
        if search and search.lower() not in f"{title_raw} {' '.join(tags)}".lower():
            continue
        link = (item.findtext("link") or "").strip()
        jobs.append({
            "source": "weworkremotely",
            "external_id": (item.findtext("guid") or link).strip(),
            "title": position.strip(),
            "company": company.strip(),
            "description": _clean(item.findtext("description")),
            "url": link,
            "tags": tags,
        })
    return jobs


def gather_jobs(search: str | None, category: str, limit: int) -> list[dict]:
    """Fetch from all sources, dedupe by (source, external_id), cap at `limit`."""
    collected: list[dict] = []
    sources = [
        ("RemoteOK", lambda: fetch_remoteok(search)),
        ("Remotive", lambda: fetch_remotive(category, search)),
        ("WeWorkRemotely", lambda: fetch_weworkremotely(search)),
    ]
    for name, fn in sources:
        try:
            got = fn()
            print(f"  {name}: {len(got)} jobs")
            collected.extend(got)
        except Exception as e:  # one source failing shouldn't abort the whole run
            print(f"  {name}: FAILED ({e})", file=sys.stderr)
        time.sleep(1)  # be polite between APIs

    seen, unique = set(), []
    for j in collected:
        key = (j["source"], j["external_id"])
        if j["external_id"] and j["title"] and key not in seen:
            seen.add(key)
            unique.append(j)
    return unique[:limit]


# --------------------------------------------------------------------------- #
# Embed + store
# --------------------------------------------------------------------------- #
def embed_documents(jobs: list[dict]) -> list[list[float]]:
    """Embed each job with Voyage, batched.

    input_type='document' is load-bearing: it must pair with input_type='query'
    at search time (pipeline.embed_query). Mismatch the type, model, or dimension
    and retrieval silently degrades with no error.
    """
    vectors: list[list[float]] = []
    for i in range(0, len(jobs), EMBED_BATCH):
        batch = jobs[i:i + EMBED_BATCH]
        texts = [
            f"{j['title']}. {j['company']}. {' '.join(j['tags'])}. {j['description'][:DESC_CHARS]}"
            for j in batch
        ]
        result = voyage.embed(texts, model=EMBED_MODEL, input_type="document", truncation=True)
        vectors.extend(result.embeddings)
        print(f"  embedded {min(i + EMBED_BATCH, len(jobs))}/{len(jobs)}")
        if EMBED_SLEEP:
            time.sleep(EMBED_SLEEP)
    return vectors


def ensure_schema(conn: psycopg.Connection, dim: int) -> None:
    """Create the vector extension, jobs table, and a cosine HNSW index if absent.

    `dim` is detected from the model's actual output, so the column width always
    matches whatever EMBED_MODEL returns.
    """
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS jobs (
            id          SERIAL PRIMARY KEY,
            source      TEXT NOT NULL,
            external_id TEXT NOT NULL,
            title       TEXT NOT NULL,
            company     TEXT,
            description TEXT,
            url         TEXT,
            tags        TEXT[],
            embedding   vector({dim}) NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT now(),
            UNIQUE (source, external_id)
        )
    """)
    # pipeline.retrieve uses `<=>` (cosine distance), so the index must use cosine ops.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS jobs_embedding_idx "
        "ON jobs USING hnsw (embedding vector_cosine_ops)"
    )
    conn.commit()


def upsert(conn: psycopg.Connection, jobs: list[dict], vectors: list[list[float]]) -> int:
    """Insert jobs, skipping any already present. Returns the count actually inserted."""
    register_vector(conn)  # lets us pass Python lists straight into a vector column
    inserted = 0
    with conn.cursor() as cur:
        for j, vec in zip(jobs, vectors):
            cur.execute(
                """
                INSERT INTO jobs
                    (source, external_id, title, company, description, url, tags, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, external_id) DO NOTHING
                """,
                (j["source"], j["external_id"], j["title"], j["company"],
                 j["description"], j["url"], j["tags"], vec),
            )
            inserted += cur.rowcount  # 1 if inserted, 0 if it was a duplicate
    conn.commit()
    return inserted


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest remote jobs into pgvector.")
    ap.add_argument("--limit", type=int, default=400, help="max jobs to ingest")
    ap.add_argument("--search", default=None, help="optional keyword filter, e.g. 'engineer'")
    ap.add_argument("--remotive-category", default="software-dev", help="Remotive category slug")
    args = ap.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set (put it in .env or export it).")

    print("Fetching jobs…")
    jobs = gather_jobs(args.search, args.remotive_category, args.limit)
    if not jobs:
        sys.exit("No jobs fetched — check connectivity or loosen --search.")
    print(f"  → {len(jobs)} unique jobs to embed")

    print("Embedding…")
    vectors = embed_documents(jobs)
    dim = len(vectors[0])
    print(f"  embedding dimension = {dim}")

    print("Writing to Postgres…")
    with psycopg.connect(db_url) as conn:
        ensure_schema(conn, dim)
        inserted = upsert(conn, jobs, vectors)
        total = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]

    print(f"Done. Inserted {inserted} new, skipped {len(jobs) - inserted} duplicate(s). "
          f"Table now holds {total} jobs.")


if __name__ == "__main__":
    main()
