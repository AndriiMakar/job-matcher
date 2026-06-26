import pandas as pd
from scipy.stats import spearmanr
from pipeline import judge_match

labels = pd.read_csv("gold_labels.csv")

rows = []
for _, r in labels.iterrows():
    verdict = judge_match(r["resume_text"], r["job_text"])
    rows.append({"human": r["human_score"], "judge": verdict["score"]})

scores = pd.DataFrame(rows)
# Correlation is an AGGREGATE over all rows — one number for the whole set — so it
# lives here, not as a per-example Phoenix evaluator (those return one score per row).
rho, p = spearmanr(scores["human"], scores["judge"])

# Bucket accuracy: do human & judge agree on good/marginal/bad bands?
def bucket(s):
    if s >= 70:
        return "good"
    if s >= 40:
        return "marginal"
    return "bad"
agree = (scores["human"].map(bucket) == scores["judge"].map(bucket)).mean()

print(f"Spearman ρ = {rho:.2f} (p={p:.3f}) | bucket agreement = {agree:.0%}")