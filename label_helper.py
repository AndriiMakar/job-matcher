# What it does: runs real retriever over each resume, then writes the top
# candidate jobs into gold_labels.csv with the `label` and `human_score` columns
# left BLANK. You open that file and fill those two columns in by hand — a
# spreadsheet app (Numbers / Excel / Google Sheets) is by far the easiest way.
#
# Run it from the project root:   python label_helper.py
import csv
import os
import re

from pipeline import retrieve

OUTPUT_CSV = "gold_labels.csv"
CANDIDATES_PER_RESUME = 10     # how many jobs to surface per resume
JOB_TEXT_CHARS = 5000          # trim each job description to keep the file readable

# --- Resumes to label against. -----------------------------------------------
# The defaults below are GENERIC examples and are safe to commit. Put YOUR real
# resumes (real names/employers and all) in a git-ignored resumes_local.py — see
# the README — and they override these, so your real resume data never reaches
# the public repo.
RESUMES = {
    "r01": "Senior backend engineer, 8 years. Python, FastAPI, Postgres, AWS, Docker, REST APIs, microservices.",
    "r02": "Frontend engineer, 5 years. React, TypeScript, Next.js, Redux, CSS, design systems, web performance.",
    "r03": "Machine learning engineer, 6 years. Python, PyTorch, RAG pipelines, vector search, embeddings, LLM evaluation.",
}

try:
    from resumes_local import RESUMES  # private resumes (git-ignored); overrides the defaults
except ImportError:
    pass  # no local file present — fall back to the generic examples above


def _one_line(text: str) -> str:
    """Collapse whitespace so each job sits in one tidy CSV cell."""
    return re.sub(r"\s+", " ", text or "").strip()


def main() -> None:
    # Refuse to overwrite — protects labels you've already filled in. Delete or
    # rename the file first if you really want to regenerate from scratch.
    if os.path.exists(OUTPUT_CSV):
        raise SystemExit(
            f"{OUTPUT_CSV} already exists — not overwriting it (your labels are safe).\n"
            f"Delete or rename it first if you want to regenerate."
        )

    # csv.writer handles all the quoting/escaping — job text has commas and quotes.
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["resume_id", "resume_text", "job_id", "job_text", "label", "human_score"]
        )

        total = 0
        for resume_id, resume_text in RESUMES.items():
            jobs = retrieve(resume_text, k=CANDIDATES_PER_RESUME)
            print(f"  {resume_id}: surfaced {len(jobs)} candidate jobs")
            for j in jobs:
                job_text = _one_line(f"{j['title']} at {j['company']}. {j['description']}")[:JOB_TEXT_CHARS]
                # label and human_score left blank on purpose — that's your job.
                writer.writerow([resume_id, resume_text, j["job_id"], job_text, "", ""])
                total += 1

    print(f"\nWrote {total} rows to {OUTPUT_CSV}.")
    print("Next: open it in a spreadsheet and fill the last two columns for every row —")
    print("  label       = good / marginal / bad")
    print("  human_score = 0-100")
    print("Delete any row you don't want to keep. Don't leave label/score blank.")


if __name__ == "__main__":
    main()
