import os
import pandas as pd
from pipeline import retrieve, judge_match
from metrics import mean_recall_at_k

RECALL_FLOOR = 0.80
BUCKET_AGREEMENT_FLOOR = 0.75
GOLD = os.environ.get("GOLD_LABELS", "gold_labels.csv")  # CI points this at a fixture

def _retrieval_rows():
    labels = pd.read_csv(GOLD)
    rows = []
    for rid, g in labels.groupby("resume_id"):
        relevant = g[g.label == "good"]["job_id"].astype(str).tolist()
        if not relevant:
            continue
        retrieved = [str(j["job_id"]) for j in retrieve(g.iloc[0]["resume_text"], k=10)]
        rows.append({"retrieved": retrieved, "relevant": relevant})
    return rows

def test_recall_gate():
    assert mean_recall_at_k(_retrieval_rows(), k=10) >= RECALL_FLOOR

def test_judge_agreement_gate():
    labels = pd.read_csv(GOLD)
    def bucket(s): return "good" if s >= 70 else ("marginal" if s >= 40 else "bad")
    agree = []
    for _, r in labels.iterrows():
        j = judge_match(r["resume_text"], r["job_text"])["score"]
        agree.append(bucket(j) == bucket(r["human_score"]))
    assert sum(agree) / len(agree) >= BUCKET_AGREEMENT_FLOOR