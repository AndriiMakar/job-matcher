# Fixed mini corpus for the CI gate (no live fetch).
# DESTRUCTIVE: drops + recreates `jobs` so ids are deterministic (1..8). For a FRESH db
# (CI Postgres or a throwaway local one) — the guard stops from wiping real data by accident.
import os
import sys

EMBED_MODEL = "voyage-4"   # MUST match pipeline.EMBED_MODEL (same model + dim as real ingest)

# (title, company, description) — ids assigned 1..8 in this order; the fixture references them.
JOBS = [
    ("Senior Backend Engineer",     "Acme",         "Python, FastAPI, PostgreSQL, AWS, Docker, REST, microservices."),
    ("Frontend Engineer",           "Bravo",        "React, TypeScript, Next.js, Redux, CSS."),
    ("Machine Learning Engineer",   "Cobalt",       "Python, PyTorch, RAG, vector search, embeddings, LLM evaluation."),
    ("DevOps Engineer",             "Delta",        "Kubernetes, Terraform, AWS, Docker, CI/CD, Linux."),
    ("Data Engineer",               "Echo",         "Python, Spark, Airflow, SQL, ETL, warehousing."),
    ("iOS Mobile Engineer",         "Foxtrot",      "Swift, UIKit, SwiftUI, iOS SDK."),
    ("Registered Nurse",            "Mercy Health", "RN license, clinical experience, patient assessment."),
    ("Senior Java Backend Engineer","Golf",         "Java, Spring Boot, PostgreSQL, microservices, REST."),
]

def job_text(title, company, description):   # the exact string used to embed AND mirrored in the fixture
    return f"{title} at {company}. {description}"

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()   # picks up VOYAGE_API_KEY from .env; an explicit DATABASE_URL still wins
    import voyageai
    import psycopg
    from pgvector.psycopg import register_vector

    vectors = voyageai.Client().embed(
        [job_text(*j) for j in JOBS], model=EMBED_MODEL, input_type="document"
    ).embeddings
    dim = len(vectors[0])

    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        # Guard: never drop an EXISTING jobs table without explicit consent. A fresh db
        # (CI, or a new test db) has no `jobs` table, so it proceeds; otherwise it stops.
        if (conn.execute("SELECT to_regclass('public.jobs')").fetchone()[0] is not None
                and os.environ.get("ALLOW_SEED_OVERWRITE") != "1"):
            n = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
            sys.exit(f"REFUSING: a 'jobs' table already exists ({n} rows). Never run against "
                     "your dev DB; to rebuild a test DB, re-run with ALLOW_SEED_OVERWRITE=1.")
        conn.execute("DROP TABLE IF EXISTS jobs")
        conn.execute(f"""CREATE TABLE jobs (
            id SERIAL PRIMARY KEY, title TEXT NOT NULL, company TEXT NOT NULL,
            description TEXT NOT NULL, embedding vector({dim}) NOT NULL)""")
        for (title, company, description), vec in zip(JOBS, vectors):
            conn.execute("INSERT INTO jobs (title, company, description, embedding) "
                         "VALUES (%s, %s, %s, %s)", (title, company, description, vec))
        conn.commit()
    print(f"Seeded {len(JOBS)} jobs (embedding dim = {dim}).")

if __name__ == "__main__":
    main()