# Imported by BOTH the Phoenix experiment (eval_retrieval.py) and the CI gate
# (tests/test_gates.py). Keeping the math here — separate from Phoenix — means
# the same numbers power the rich dev UI and the CI check, and CI can assert on
# them without standing up a Phoenix server.


def recall_at_k_value(retrieved_ids, relevant_ids) -> float | None:
    """Fraction of the relevant jobs that appear in the retrieved list.

        recall = (relevant jobs that were retrieved) / (all relevant jobs)

    1.0 means every job that should have been found was found; 0.5 means half.
    Returns None when there are no relevant jobs for this row (nothing to
    measure), so the caller skips it instead of dividing by zero.
    """
    relevant = set(map(str, relevant_ids))
    if not relevant:
        return None
    hits = len(set(map(str, retrieved_ids)) & relevant)
    return hits / len(relevant)


def mean_recall_at_k(eval_rows, k) -> float:
    """Average recall@k across rows (rows with no relevant jobs are skipped)."""
    vals = [recall_at_k_value(r["retrieved"][:k], r["relevant"]) for r in eval_rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals)