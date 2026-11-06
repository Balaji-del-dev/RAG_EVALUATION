"""Execute one evaluation run: answer every question, then score everything, then summarize.

Answering and scoring are separate phases so the LLM can be unloaded before the NLI model loads. On a
7.5 GB machine, holding both at once is what triggers the out-of-memory killer.
"""

import contextlib
import importlib.metadata
import logging

from django.conf import settings
from django.db import connection
from django.utils import timezone

from apps.chatbot.services import llm_gateway
from apps.chatbot.services.generation import PROMPTS
from apps.chatbot.services.rag_workflow import run_rag
from apps.common.constants import JobStatus
from apps.common.exceptions import ServiceUnavailable
from apps.documents.models import Document
from apps.documents.services.ingestion import ensure_index
from apps.evaluation.models import EvaluationError, EvaluationResult, EvaluationRun, MetricDefinition, MetricScore
from apps.evaluation.services import dataset_manager
from apps.evaluation.services.evaluator_registry import EvalContext, RunContext, load_all_metrics, registry
from apps.evaluation.services.roadmap import ROADMAP
from apps.evaluation.services.security_evaluator import execute_security_suite
from apps.monitoring.services.cost_tracker import compute_cost
from apps.retrieval.services.bm25_search import backend as bm25_backend
from apps.security.services.access_control import all_ready_document_ids

logger = logging.getLogger(__name__)
TRACKED_PACKAGES = ["django", "torch", "sentence-transformers", "litellm", "langgraph", "presidio-analyzer"]
# These six make up the overall quality score shown on the dashboard. Each counts equally.
QUALITY_SCORE_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "abstention_accuracy",
]


def sync_metric_definitions() -> None:
    """Make the metric_definitions table match the code, which is the source of truth."""
    load_all_metrics()
    for order, (name, display_name, priority, question) in enumerate(ROADMAP, start=1):
        evaluator = registry.evaluators[name]
        MetricDefinition.objects.update_or_create(
            name=name,
            defaults={
                "display_name": display_name,
                "order": order,
                "priority": priority,
                "question": question,
                "category": evaluator.category,
                "formula": evaluator.formula,
                "description": evaluator.description,
                "scope": evaluator.scope,
                "higher_is_better": evaluator.higher_is_better,
                "unit": evaluator.unit,
            },
        )


def _package_versions() -> dict:
    versions = {}
    for package in TRACKED_PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[package] = importlib.metadata.version(package)
    return versions


def build_snapshot(run: EvaluationRun) -> dict:
    """Record everything needed to reproduce the run, even if the config is later edited."""
    config = run.rag_config
    return {
        "rag_config": {
            field.name: str(getattr(config, field.name))
            for field in config._meta.concrete_fields
            if field.name not in ("id", "created_at", "updated_at", "description", "is_default")
        },
        "index_key": config.index_key,
        "bm25_backend": bm25_backend().name,
        "prompt": {"version": config.prompt_version, "text": PROMPTS[config.prompt_version]},
        "llm_digest": llm_gateway.model_digest(config.llm_model),
        "dataset": {"id": run.dataset_id, "name": run.dataset.name, "version": run.dataset.version} if run.dataset else None,
        "corpus": sorted(corpus_for(run)) if run.dataset else None,
        "thresholds": settings.EVALUATION,
        "packages": _package_versions(),
        "judge": {"nli": settings.NLI_MODEL, "embedding": settings.EVALUATION["JUDGE_EMBEDDING_MODEL"]},
    }


def _warm_up(config, allowed: list[int]) -> None:
    """Load the models before timing starts. In testing a cold first question took 61 s against 5 s for
    the rest, which would otherwise dominate p95 and hide the latency that users actually see."""
    try:
        run_rag("What is this collection of documents about?", config, allowed)
    except ServiceUnavailable as exc:
        # If a service is down, each case will fail and record that error itself.
        logger.warning("Warm-up failed: %s", exc)


def corpus_for(run: EvaluationRun) -> list[int]:
    """The documents a run may search. A dataset is written against particular documents, so uploading
    something new must not change its scores, or runs from different days could not be compared."""
    pinned = run.dataset.documents.filter(status=Document.Status.READY).values_list("id", flat=True)
    return list(pinned) or all_ready_document_ids()


def answer_cases(run: EvaluationRun) -> None:
    """Phase 1: run every question through the real pipeline.

    When resuming an interrupted run, saved answers are kept and earlier failures get another try.
    """
    config = run.rag_config
    allowed = corpus_for(run)
    run.errors.filter(stage="answering").delete()
    answered = set(run.results.values_list("case_id", flat=True))
    all_cases = list(run.dataset.cases.all())
    remaining = [case for case in all_cases if case.id not in answered]
    completed, failed = len(all_cases) - len(remaining), 0
    run.total_cases = len(all_cases)
    run.completed_cases, run.failed_cases = completed, failed
    run.progress_stage = "answering"
    run.save(update_fields=["total_cases", "completed_cases", "failed_cases", "progress_stage", "updated_at"])

    if remaining:
        _warm_up(config, allowed)
    for case in remaining:
        try:
            outcome = run_rag(case.question, config, allowed)
        except ServiceUnavailable as exc:
            # One failed question must not lose the rest of the run. It feeds the error-rate metric.
            logger.warning("Case %s failed: %s", case.id, exc)
            EvaluationError.objects.create(run=run, case=case, stage="answering", error_message=str(exc))
            failed += 1
        else:
            input_cost, output_cost = compute_cost(config, outcome.prompt_tokens, outcome.completion_tokens)
            EvaluationResult.objects.update_or_create(
                run=run,
                case=case,
                defaults={
                    "answer": outcome.answer,
                    "contexts": [chunk.text for chunk in outcome.chunks],
                    "retrieved_chunk_ids": [chunk.chunk_id for chunk in outcome.chunks],
                    "relevant_chunk_ids": sorted(
                        dataset_manager.relevant_chunk_ids(case.evidence, config.index_key, allowed)
                    ),
                    "citations": outcome.citations,
                    "abstained": outcome.abstained,
                    "blocked": outcome.blocked,
                    "retrieval_ms": outcome.retrieval_ms,
                    "generation_ms": outcome.generation_ms,
                    "total_ms": outcome.total_ms,
                    "prompt_tokens": outcome.prompt_tokens,
                    "completion_tokens": outcome.completion_tokens,
                    "cost": input_cost + output_cost,
                },
            )
            completed += 1
        run.completed_cases, run.failed_cases = completed, failed
        run.save(update_fields=["completed_cases", "failed_cases", "updated_at"])


def context_for(result: EvaluationResult, k: int) -> EvalContext:
    return EvalContext(
        question=result.case.question,
        answer=result.answer,
        reference=result.case.reference_answer,
        contexts=result.contexts,
        retrieved_ids=result.retrieved_chunk_ids,
        relevant_ids=set(result.relevant_chunk_ids),
        evidence=result.case.evidence,
        answerable=result.case.answerable,
        abstained=result.abstained,
        blocked=result.blocked,
        k=k,
    )


def score_cases(run: EvaluationRun, definitions: dict[str, MetricDefinition]) -> None:
    """Phase 2: score each saved answer. A metric that crashes is logged, and the others still run."""
    run.progress_stage = "scoring"
    run.save(update_fields=["progress_stage", "updated_at"])
    evaluators = registry.for_scope("case", run.selected_metrics)
    for result in run.results.select_related("case"):
        ctx = context_for(result, run.k)
        for evaluator in evaluators:
            try:
                value = registry.evaluate(evaluator.name, ctx)
            except Exception as exc:  # any model or math failure must stay contained to this one score
                logger.exception("Metric %s failed on result %s", evaluator.name, result.id)
                EvaluationError.objects.create(
                    run=run, case=result.case, stage=f"metric:{evaluator.name}", error_message=str(exc)
                )
                continue
            if value is not None:
                MetricScore.objects.update_or_create(
                    run=run,
                    result=result,
                    metric=definitions[evaluator.name],
                    framework="builtin",
                    defaults={"score": value.score, "details": value.details},
                )


def score_run_level(run: EvaluationRun, definitions: dict[str, MetricDefinition]) -> None:
    ctx = RunContext(
        run=run,
        results=list(run.results.all()),
        security_results=list(run.security_results.select_related("test")),
        failed_cases=run.failed_cases,
    )
    for evaluator in registry.for_scope("run", run.selected_metrics):
        try:
            value = registry.evaluate(evaluator.name, ctx)
        except Exception as exc:
            logger.exception("Run metric %s failed", evaluator.name)
            EvaluationError.objects.create(run=run, stage=f"metric:{evaluator.name}", error_message=str(exc))
            continue
        if value is not None:
            MetricScore.objects.update_or_create(
                run=run,
                result=None,
                metric=definitions[evaluator.name],
                framework="builtin",
                defaults={"score": value.score, "details": value.details},
            )


SUMMARY_SQL = """
SELECT md.name, ms.framework, md.category, md.display_name, md.higher_is_better, md.unit, md."order", md.priority,
       AVG(ms.score) AS mean, MIN(ms.score) AS min, MAX(ms.score) AS max, COUNT(*) AS count
FROM metric_scores ms
JOIN metric_definitions md ON md.id = ms.metric_id
WHERE ms.run_id = %s
GROUP BY md.id, ms.framework
"""


def summarize(run: EvaluationRun) -> dict:
    with connection.cursor() as cursor:
        cursor.execute(SUMMARY_SQL, [run.id])
        rows = cursor.fetchall()
    metrics = {}
    for name, framework, category, display_name, higher_is_better, unit, order, priority, mean, low, high, count in rows:
        key = name if framework == "builtin" else f"{name}:{framework}"
        metrics[key] = {
            "metric": name,
            "framework": framework,
            "display_name": display_name,
            "category": category,
            "higher_is_better": higher_is_better,
            "unit": unit,
            # jsonb does not keep key order, so each metric carries its own position.
            "order": order,
            "priority": priority,
            "mean": mean,
            "min": low,
            "max": high,
            "count": count,
        }
    quality = [metrics[name]["mean"] for name in QUALITY_SCORE_METRICS if name in metrics]
    return {
        "metrics": metrics,
        "overall_quality": sum(quality) / len(quality) if quality else None,
        "metrics_scored": sum(1 for key in metrics if ":" not in key),
    }


def execute_run(run_id: int, worker: str = "") -> EvaluationRun | None:
    run = EvaluationRun.objects.select_related("rag_config", "dataset").filter(pk=run_id).first()
    if run is None:
        logger.info("Run %s was deleted while it was queued", run_id)
        return None
    # Several restarts can each queue a resume for the same run. Only the first one has work to do.
    if run.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        return run
    resuming = run.status == JobStatus.RUNNING
    run.status = JobStatus.RUNNING
    run.worker = worker
    run.error = ""
    sync_metric_definitions()
    if not resuming:
        # A resumed run keeps its original start time and snapshot: they describe how it began.
        run.started_at = timezone.now()
        run.snapshot = build_snapshot(run)
    run.save()
    definitions = {d.name: d for d in MetricDefinition.objects.all()}

    try:
        run.progress_stage = "indexing"
        run.save(update_fields=["progress_stage", "updated_at"])
        ensure_index(run.rag_config)

        if run.run_type in (EvaluationRun.RunType.QUALITY, EvaluationRun.RunType.FULL):
            answer_cases(run)
        if run.run_type in (EvaluationRun.RunType.SECURITY, EvaluationRun.RunType.FULL):
            run.progress_stage = "security"
            run.save(update_fields=["progress_stage", "updated_at"])
            execute_security_suite(run)

        # Frameworks judge with the LLM, so they run while it is loaded and before the NLI model is.
        _run_frameworks(run, definitions)
        llm_gateway.unload(run.rag_config.llm_model)
        score_cases(run, definitions)
        score_run_level(run, definitions)

        run.summary = summarize(run)
        run.status = JobStatus.COMPLETED
    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        run.status = JobStatus.FAILED
        run.error = str(exc)[:2000]
        run.save(update_fields=["status", "error", "updated_at"])
        raise
    finally:
        run.finished_at = timezone.now()
        run.progress_stage = ""
        run.save(update_fields=["finished_at", "progress_stage", "updated_at"])
    run.save()
    return run


def _run_frameworks(run: EvaluationRun, definitions: dict[str, MetricDefinition]) -> None:
    """Optional Ragas and DeepEval scoring. A framework failure is recorded but does not fail the run,
    because the built-in scores are already saved."""
    adapters = {}
    if "ragas" in run.frameworks:
        from apps.evaluation.services.ragas_evaluator import score_run as score_with_ragas

        adapters["ragas"] = score_with_ragas
    if "deepeval" in run.frameworks:
        from apps.evaluation.services.deepeval_evaluator import score_run as score_with_deepeval

        adapters["deepeval"] = score_with_deepeval
    for name, adapter in adapters.items():
        run.progress_stage = name
        run.save(update_fields=["progress_stage", "updated_at"])
        try:
            adapter(run, definitions)
        except Exception as exc:
            logger.exception("%s scoring failed", name)
            EvaluationError.objects.create(run=run, stage=name, error_message=str(exc)[:2000])
