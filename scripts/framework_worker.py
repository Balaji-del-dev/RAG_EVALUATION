"""Scores answers with Ragas or DeepEval. Runs under venv-frameworks, with no Django and no torch.

Usage: framework_worker.py {ragas|deepeval} INPUT.json OUTPUT.json

Results go to a file, not stdout, because both libraries print to stdout. A failure on one metric is
recorded in "errors" and the rest carry on.
"""

import asyncio
import importlib.metadata
import json
import math
import os
import sys
import traceback

# Local-first: neither library may phone home with this data.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["ERROR_REPORTING"] = "NO"


def _record(output: dict, row: dict, metric: str, score) -> None:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        output["errors"].append({"result_id": row["result_id"], "metric": metric, "error": "The framework returned no score"})
    else:
        output["scores"].append({"result_id": row["result_id"], "metric": metric, "score": float(score)})


def _record_failure(output: dict, row: dict, metric: str) -> None:
    last_line = traceback.format_exc().strip().splitlines()[-1][:500]
    output["errors"].append({"result_id": row["result_id"], "metric": metric, "error": last_line})


def run_ragas(job: dict, output: dict) -> None:
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    client = AsyncOpenAI(base_url=f"{job['ollama_url']}/v1", api_key="ollama", timeout=job["llm_timeout"])
    llm = llm_factory(job["model"], provider="openai", client=client, max_tokens=job["max_tokens"])
    embeddings = embedding_factory("openai", model=job["embedding_model"], client=client)
    metrics = {
        "faithfulness": (
            Faithfulness(llm=llm),
            lambda m, r: m.ascore(user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"]),
        ),
        "answer_relevancy": (
            AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
            lambda m, r: m.ascore(user_input=r["question"], response=r["answer"]),
        ),
        "context_recall": (
            ContextRecall(llm=llm),
            lambda m, r: m.ascore(user_input=r["question"], retrieved_contexts=r["contexts"], reference=r["reference"]),
        ),
        "context_precision": (
            ContextPrecision(llm=llm),
            lambda m, r: m.ascore(user_input=r["question"], reference=r["reference"], retrieved_contexts=r["contexts"]),
        ),
    }

    async def score_all():
        for row in job["rows"]:
            for name in job["metrics"]:
                metric, call = metrics[name]
                try:
                    result = await call(metric, row)
                    _record(output, row, name, result.value)
                except Exception:  # one bad judgement must not lose the other scores
                    _record_failure(output, row, name)

    asyncio.run(score_all())


def run_deepeval(job: dict, output: dict) -> None:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import OllamaModel
    from deepeval.test_case import LLMTestCase

    model = OllamaModel(model=job["model"], base_url=job["ollama_url"], temperature=0)
    options = {"model": model, "async_mode": False, "include_reason": False}
    metrics = {
        "faithfulness": FaithfulnessMetric(**options),
        "answer_relevancy": AnswerRelevancyMetric(**options),
        "context_recall": ContextualRecallMetric(**options),
        "context_precision": ContextualPrecisionMetric(**options),
    }
    for row in job["rows"]:
        case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row["reference"],
            retrieval_context=row["contexts"],
        )
        for name in job["metrics"]:
            try:
                metrics[name].measure(case)
                _record(output, row, name, metrics[name].score)
            except Exception:
                _record_failure(output, row, name)


RUNNERS = {"ragas": run_ragas, "deepeval": run_deepeval}


def main() -> int:
    framework, input_path, output_path = sys.argv[1:4]
    with open(input_path) as handle:
        job = json.load(handle)
    output = {"framework": framework, "version": importlib.metadata.version(framework), "scores": [], "errors": []}
    try:
        RUNNERS[framework](job, output)
    except Exception:
        output["fatal"] = traceback.format_exc()[-2000:]
    with open(output_path, "w") as handle:
        json.dump(output, handle)
    return 1 if "fatal" in output else 0


if __name__ == "__main__":
    sys.exit(main())
