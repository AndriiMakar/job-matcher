#!/usr/bin/env python
"""Run the resume → job matcher against the jobs already in pgvector.

Usage:
    python match.py "Senior backend engineer — Python, FastAPI, Postgres, AWS, 8 yrs"
    python match.py --resume-file my_resume.txt
    python match.py "<resume text>" --top-n 10

Requires the corpus to be ingested first (see `scripts/ingest_jobs.py`) and
DATABASE_URL + ANTHROPIC_API_KEY + VOYAGE_API_KEY set (a .env file is loaded).
"""
import argparse
import sys

from pipeline import match_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Match a resume against the ingested job corpus.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("resume", nargs="?", help="Resume text, quoted.")
    src.add_argument("--resume-file", help="Path to a file containing the resume text.")
    parser.add_argument("--top-n", type=int, default=5, help="How many matches to return (default 5).")
    args = parser.parse_args()

    if args.resume_file:
        with open(args.resume_file, encoding="utf-8") as f:
            resume_text = f.read()
    else:
        resume_text = args.resume
    if not resume_text or not resume_text.strip():
        sys.exit("Resume text is empty.")

    matches = match_pipeline(resume_text, top_n=args.top_n)
    if not matches:
        sys.exit("No matches — is the job corpus ingested? See scripts/ingest_jobs.py.")

    for rank, m in enumerate(matches, 1):
        print(f"{rank}. [{m['job_id']}] {m['title']} @ {m['company']}  (score {m['score']:.2f})")


if __name__ == "__main__":
    main()
