# job-matcher

![eval-gate](https://github.com/AndriiMakar/job-matcher/actions/workflows/eval-gate.yml/badge.svg)

A resume → job matcher built as a RAG pipeline — but the real artifact is the **evaluation and
observability harness** around it. The pipeline retrieves and scores job matches; the harness measures
whether those matches are actually good, and a CI gate blocks a merge if quality regresses.

> Built to demonstrate LLM **evaluation engineering** — a hand-labelled gold set, an LLM-as-judge
> validated against human labels, retrieval and faithfulness metrics, and evals wired into CI — not just
> a working demo.

---

## Results

| Signal                  | Result                                           | What it checks                                                   |
| ----------------------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| Retrieval recall@10     | **1.00** (vector) vs **0.80** (keyword baseline) | Do the relevant jobs land in the top-K?                          |
| Judge ↔ human agreement | **ρ = 0.70**, 68% bucket agreement (40 pairs)    | Does the LLM judge's 0–100 score track a human's?                |
| Rationale faithfulness  | grounded (guardrail)                             | Does the judge cite only skills actually present in the texts?   |
| CI eval gate            | **green on every push**                          | Fails the build if recall or judge agreement drops below a floor |

---

## Architecture

Two subsystems, not one app:

**1. The matching pipeline** (`pipeline.py`)

```
resume
  └─▶ embed (Voyage voyage-4)
        └─▶ pgvector retrieve  → top-K candidate jobs
              └─▶ Claude rerank (Sonnet)      → top-N, reordered
                    └─▶ Claude judge (Sonnet) → score 0–100 + rationale
```

**2. The eval + observability harness**

- **Arize Phoenix** traces every span — tokens, cost, latency, retrieved docs — so cost-per-query and
  latency dashboards populate automatically.
- **Retrieval experiment** — recall@K, vector vs a keyword/TF-IDF baseline.
- **Judge validation** — the LLM judge is checked against human labels (Spearman ρ + bucket agreement)
  _before_ it's trusted.
- **Generation experiment** — is the judge's score within N points of the human's, and is its rationale
  faithful to the texts?
- **CI eval gate** — the metric math is pure (no Phoenix dependency), so CI asserts the numbers on a
  fixed synthetic fixture without standing up a server.

---

## Quickstart

Requires Python 3.12, Docker, and API keys for [Anthropic](https://console.anthropic.com) and
[Voyage AI](https://voyageai.com).

```bash
# 1. Clone
git clone https://github.com/AndriiMakar/job-matcher.git
cd job-matcher

# 2. Python env + dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Secrets — create .env (see the block below)

# 4. Start Phoenix + Postgres (pgvector)
docker compose up -d

# 5. Ingest a corpus of real jobs into pgvector
python scripts/ingest_jobs.py --limit 400 --search engineer

# 6. Run the matcher
python match.py "Senior backend engineer — Python, FastAPI, Postgres, AWS, 8 yrs"
```

`.env` (the host port must match what `docker-compose.yml` maps Postgres to — `5433` here):

```bash
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_BASE_URL=http://localhost:6006
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/postgres
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
```

Then open **http://localhost:6006** — every matcher run shows a full trace tree
(`match_pipeline → retrieve → embed_query`, `rerank`, `judge`) with cost, latency, and tokens on the
LLM spans.

---

## Running the evals

The experiments score the pipeline against a hand-labelled gold set of (resume, job) pairs that you
assemble. Real resumes and labels are **private and git-ignored**, so you provide your own:

1. **Provide the resumes.** The matcher reads them from a `RESUMES` dict (`id → resume text`). Three
   generic placeholder resumes ship inside `label_helper.py`, so it runs out of the box. To use your
   own, create a **git-ignored** `resumes_local.py` — it overrides the defaults and never reaches the
   repo:
   ```python
   # resumes_local.py  (git-ignored — your real resumes live here)
   RESUMES = {
       "r04": """Senior full-stack engineer, 9 years. TypeScript, React, Next.js, Node, Postgres, AWS …""",
       "r05": """… another full resume; triple quotes allow multiple lines …""",
   }
   ```
2. **Generate the labeling template.** This runs the retriever over each resume and writes the top
   candidate jobs it surfaces into `gold_labels.csv`, with the label/score columns left blank:
   ```bash
   python label_helper.py
   ```
3. **Label it by hand — required manual step.** Open `gold_labels.csv` and, for **every row**, fill in
   two columns yourself:
   - `label` → `good`, `marginal`, or `bad`
   - `human_score` → an integer `0–100`: _your_ judgment of the match. This is what the LLM judge is
     later calibrated against, so it can't be auto-filled.

   Criteria are in [`LABELING_GUIDE.md`](LABELING_GUIDE.md) — `good` 70–100, `marginal` 40–69, `bad` 0–39; pick the
   score first, the label is its band.

4. **Upload to Phoenix:**
   ```bash
   python gold_set.py            # uploads two dataset "views" (retrieval + generation)
   ```

Then run each experiment — results open in the Phoenix UI:

```bash
python eval_retrieval.py      # recall@K: vector vs keyword baseline
python validate_judge.py      # judge ↔ human agreement (Spearman ρ + bucket %)
python eval_generation.py     # within-N score check + rationale faithfulness
```

---

## CI eval gate

`.github/workflows/eval-gate.yml` runs on every pull request and push to `main`. It spins up a fresh
Postgres, seeds a fixed 8-job corpus (`scripts/seed_test_jobs.py`) plus a tiny **synthetic** fixture
(`tests/fixtures/gold_labels.csv` — no private data), and **fails the build** if recall@K or judge
agreement falls below the floors in `tests/test_gates.py`.

Run the same gate locally before pushing:

```bash
make seed-test-db   # once, into a throwaway test DB
make eval           # recall + judge-agreement gates
```

**CI setup:** add `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` under the repo's
_Settings → Secrets and variables → Actions_. (An optional `pre-push` hook in `scripts/hooks/` runs the
gate locally before each push — enable with `git config core.hooksPath scripts/hooks`.)

---

## Project layout

```
pipeline.py            # the instrumented retrieve → rerank → judge pipeline
match.py               # CLI entry point — match a resume against the ingested corpus
metrics.py             # pure metric functions (recall@K) — reused by the experiment AND CI
gold_set.py            # uploads the gold set to Phoenix as two dataset views
label_helper.py        # generates the labeling template (committed, generic resumes)
LABELING_GUIDE.md              # good / marginal / bad labeling criteria
eval_retrieval.py      # recall@K experiment (vector vs keyword baseline)
validate_judge.py      # judge ↔ human agreement (the "earn trust" step)
eval_generation.py     # within-N score check + rationale faithfulness
conftest.py            # puts the repo root on sys.path so tests can import the modules
Makefile               # `make eval`, `make seed-test-db`
scripts/
  ingest_jobs.py       # fetch + embed + load real jobs into pgvector
  seed_test_jobs.py    # deterministic 8-job corpus for CI / local test runs
  hooks/pre-push       # optional: run the eval gate before pushing
tests/
  test_gates.py        # the CI recall + judge-agreement gates
  fixtures/
    gold_labels.csv    # tiny SYNTHETIC label set for CI (safe to commit)
.github/workflows/
  eval-gate.yml        # the CI workflow
```

`resumes_local.py` and the root `gold_labels.csv` hold real data and are git-ignored — they never reach
the repo. The `gold_labels.csv` under `tests/fixtures/` is synthetic and committed.

---

## Tech stack

<ol>
<li>Python</li>
<li><a href="https://voyageai.com">Voyage AI</a> <code>voyage-4</code> embeddings</li>
<li>PostgreSQL + <code>pgvector</code></li>
<li><a href="https://www.anthropic.com">Claude</a> (Sonnet for rerank + judge, Haiku for faithfulness)</li>
<li><a href="https://phoenix.arize.com">Arize Phoenix</a> for tracing &amp; experiments; pytest + GitHub Actions for the eval gate</li>
</ol>
