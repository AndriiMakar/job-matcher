# Build the two Phoenix dataset views from gold_labels.csv.
#
# Idempotent: each dataset is fetched by name if it already exists, else created.
# That matters because the eval scripts do `from gold_set import retrieval_ds`,
# which re-runs this file — so re-running/importing must NOT try to re-create an
# existing dataset. To make a *fresh* dataset (e.g. after re-labeling), bump the
# "-v1" suffix in the names below to "-v2".
import pandas as pd
from phoenix.client import Client
from dotenv import load_dotenv

load_dotenv()
px = Client()

labels = pd.read_csv("gold_labels.csv")

# --- View A: retrieval (one row per resume; expected = the set of relevant job_ids) ---
retrieval_rows = (
    labels[labels.label == "good"]
    .groupby(["resume_id", "resume_text"])["job_id"]
    .apply(lambda s: [str(x) for x in s])
    .reset_index(name="relevant_job_ids")
)


def _create_or_get(name, dataframe, input_keys, output_keys):
    """Fetch the dataset by name if it exists, else create it from the dataframe."""
    try:
        return px.datasets.get_dataset(dataset=name)        # exists → reuse it
    except Exception:
        return px.datasets.create_dataset(                  # first time → create it
            name=name,
            dataframe=dataframe,
            input_keys=list(input_keys),
            output_keys=list(output_keys),
        )


retrieval_ds = _create_or_get(
    "resume-retrieval-v2",
    retrieval_rows,
    input_keys=["resume_id", "resume_text"],
    output_keys=["relevant_job_ids"],
)

# --- View B: generation (one row per pair; expected = human_score/label) -------
generation_ds = _create_or_get(
    "resume-job-pairs-v2",
    labels,
    input_keys=["resume_text", "job_text"],
    output_keys=["human_score", "label"],
)


if __name__ == "__main__":
    print("Datasets ready in Phoenix: resume-retrieval-v2, resume-job-pairs-v2")