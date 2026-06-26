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
retrieval_ds = px.datasets.create_dataset(
    name="resume-retrieval-v1",
    dataframe=retrieval_rows,
    input_keys=["resume_id", "resume_text"],
    output_keys=["relevant_job_ids"],
)

# --- View B: generation (one row per pair; expected = human_score/label) -------
generation_ds = px.datasets.create_dataset(
    name="resume-job-pairs-v1",
    dataframe=labels,
    input_keys=["resume_text", "job_text"],
    output_keys=["human_score", "label"],
)