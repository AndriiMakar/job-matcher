# Gold-set labeling guide

You're scoring how well a **resume** matches a **job** — one `human_score` (0–100) and one
`label` per pair. This guide keeps your scoring consistent across all pairs, and consistency is
exactly what makes the gold set a trustworthy answer key.

Keep this open while you fill in `gold_labels.csv`.

## What to weigh (rough priority order)

1. **Role & seniority fit** — right kind of role, at the right level? (Senior backend resume vs a
   senior backend role = strong; vs a junior role or an eng-manager role = weaker.)
2. **Must-have skills present** — does the resume show the core skills the job centers on? (Job
   wants Python / Postgres / AWS; resume has them = strong.)
3. **Domain adjacency** — same or neighboring domain? (Both web/SaaS = good; fintech resume vs a
   game studio = weaker even when individual skills overlap.)
4. **Location / visa workability** — if the job says "US only" and the resume is clearly elsewhere
   with no work authorization, that caps the score.

## The three bands

- **good (70–100)** — right role and level, must-have skills present, domain same or adjacent,
  location workable. A job you'd genuinely send this person.
- **marginal (40–69)** — partial fit. Real overlap _and_ a real gap: wrong seniority, one missing
  core skill, adjacent-but-not-quite role, or a location question. Worth a look, not a clear yes.
- **bad (0–39)** — wrong role or wrong stack. Little meaningful overlap. You would not send this.

## Keep `label` and `human_score` consistent

Pick the **score** first; the **label** is just its band:

| Score  | Label    |
| ------ | -------- |
| 70–100 | good     |
| 40–69  | marginal |
| 0–39   | bad      |

They must agree — a row labeled `good` with a score of 30 will confuse the later steps (retrieval
uses the label, judge calibration uses the score).

## Worked examples

Resume: _senior backend engineer — Python, FastAPI, Postgres, AWS, Docker, 8 yrs._

| Job                                          | Label    | Score | Why                                          |
| -------------------------------------------- | -------- | ----- | -------------------------------------------- |
| Senior Python/FastAPI backend, US-remote     | good     | 88    | role, level, and core skills all line up     |
| Full-stack (Node + React + Postgres)         | marginal | 55    | shares Postgres/backend, but front-end heavy |
| DevOps / SRE (Terraform, K8s, AWS)           | marginal | 50    | shares AWS/Docker, but different focus       |
| Frontend (React, TypeScript, design systems) | bad      | 15    | wrong stack and role                         |
| Data engineer (Spark, Airflow, dbt)          | bad      | 25    | different stack entirely                     |

## Tips for staying consistent

- **Score the pair, not the resume or the job alone.** A great resume against an irrelevant job is
  still a bad _match_.
- **Use the full range.** Don't cluster everything around 50 — spread good/marginal/bad apart.
- **Unsure between two bands? That's a `marginal`.** Those borderline cases are the most useful for
  testing the judge later, so don't avoid them.
- **~30–60 seconds per pair.** Don't overthink it — your honest gut call is the signal you want.
