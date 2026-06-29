# Generation experiment (within-N score check + faithfulness).
from phoenix.client.experiments import run_experiment
from phoenix.evals.llm import LLM
from phoenix.evals.metrics import FaithfulnessEvaluator
from pipeline import judge_match
from gold_set import generation_ds


def judge_task(input: dict) -> dict:
    return judge_match(input["resume_text"], input["job_text"])   # {"score", "rationale"}


# Evaluator 1 (per-example): is the judge's score within 15 points of the human's?
# float() guards against the dataset handing the value back as a float/str.
def within_15(output: dict, expected: dict) -> bool:
    return abs(float(output["score"]) - float(expected["human_score"])) <= 15


# Evaluator 2 (per-example): is the judge's RATIONALE grounded in the resume — i.e. does
# it cite skills that are actually there, not hallucinated? Run with a DIFFERENT model
# than the generator (Haiku vs the Sonnet judge) as a bias mitigation: a model shouldn't
# grade its own output. A different *family* (OpenAI) would be stricter still — swap to
# LLM(provider="openai", model="gpt-4o") if you have an OpenAI key + the openai package.
_faith = FaithfulnessEvaluator(llm=LLM(provider="anthropic", model="claude-haiku-4-5-20251001"))


def faithful_rationale(input: dict, output: dict) -> float:
    # The rationale legitimately references BOTH the resume (skills the candidate has)
    # and the job (requirements it's missing). Grounding it against the resume ALONE
    # falsely flags every true statement about the job as a hallucination — so the
    # context must be the full (resume, job) pair the judge was actually given.
    result = _faith.evaluate({
        "input": "Is this match rationale grounded in the resume and job below?",
        "context": f"RESUME:\n{input['resume_text']}\n\nJOB:\n{input['job_text']}",
        "output": output["rationale"],
    })
    return result[0].score                    # 1.0 grounded / 0.0 fabricated


experiment = run_experiment(
    dataset=generation_ds,
    task=judge_task,
    evaluators=[within_15, faithful_rationale],
    experiment_name="generation-judge-v2",
)