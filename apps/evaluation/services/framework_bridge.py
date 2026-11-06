"""Runs Ragas or DeepEval in their separate environment and stores the scores next to the built-in ones.

They run in a subprocess because they cannot share an environment with current transformers (see
requirements/evaluation.txt). This also means their memory is returned to the system when they finish.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path

from django.conf import settings

from apps.chatbot.services.llm_gateway import _ollama_name
from apps.evaluation.models import EvaluationError, EvaluationRun, MetricDefinition, MetricScore
from apps.evaluation.services.claims import strip_citations
from config.env import env

logger = logging.getLogger(__name__)
FRAMEWORK_PYTHON = Path(settings.BASE_DIR) / "venv-frameworks" / "bin" / "python"
WORKER = Path(settings.BASE_DIR) / "scripts" / "framework_worker.py"
TIMEOUT_SECONDS = 3600
SCORED_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
EMBEDDING_MODEL = "all-minilm"


def _rows(run: EvaluationRun) -> list[dict]:
    """Only real answers to answerable questions: these frameworks have no notion of a correct refusal."""
    return [
        {
            "result_id": result.id,
            "question": result.case.question,
            "answer": strip_citations(result.answer),
            "contexts": result.contexts,
            "reference": result.case.reference_answer,
        }
        for result in run.results.select_related("case")
        if result.case.answerable and not result.abstained and not result.blocked and result.contexts
    ]


def score_run(framework: str, run: EvaluationRun, definitions: dict[str, MetricDefinition]) -> None:
    if not FRAMEWORK_PYTHON.exists():
        raise RuntimeError(f"{FRAMEWORK_PYTHON} not found. See requirements/evaluation.txt to create it.")
    model = _ollama_name(run.rag_config.llm_model)
    if not model:
        raise RuntimeError("Framework scoring currently supports Ollama models only")
    rows = _rows(run)
    if not rows:
        return

    job = {
        "rows": rows,
        "metrics": SCORED_METRICS,
        "model": model,
        "embedding_model": EMBEDDING_MODEL,
        "ollama_url": env.OLLAMA_BASE_URL,
        "llm_timeout": env.LLM_TIMEOUT_SECONDS,
        "max_tokens": 1024,
    }
    with tempfile.TemporaryDirectory() as folder:
        input_path, output_path = Path(folder) / "in.json", Path(folder) / "out.json"
        input_path.write_text(json.dumps(job))
        completed = subprocess.run(
            [str(FRAMEWORK_PYTHON), str(WORKER), framework, str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        if not output_path.exists():
            raise RuntimeError(f"{framework} worker crashed: {completed.stderr[-1000:]}")
        output = json.loads(output_path.read_text())

    if output.get("fatal"):
        raise RuntimeError(f"{framework} failed: {output['fatal']}")

    results = {result.id: result for result in run.results.all()}
    for item in output["scores"]:
        MetricScore.objects.update_or_create(
            run=run,
            result=results[item["result_id"]],
            metric=definitions[item["metric"]],
            framework=framework,
            defaults={"score": item["score"]},
        )
    for item in output["errors"]:
        EvaluationError.objects.create(
            run=run,
            case=results[item["result_id"]].case,
            stage=f"{framework}:{item['metric']}",
            error_message=item["error"],
        )
    run.snapshot.setdefault("frameworks", {})[framework] = output["version"]
    run.save(update_fields=["snapshot", "updated_at"])
    logger.info("%s: %d scores, %d errors", framework, len(output["scores"]), len(output["errors"]))
