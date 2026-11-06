"""Dashboard summary and run exports."""

import csv
import io

from django.db.models import Count

from apps.common.constants import JobStatus
from apps.documents.models import Document, DocumentChunk
from apps.evaluation.models import EvaluationDataset, EvaluationRun
from apps.monitoring.services.latency import live_metrics
from apps.reports.models import Report
from apps.security.models import SecurityResult

# CSV formula injection: a cell starting with one of these runs as a formula when opened in Excel. Model
# answers can contain anything, so such cells get a leading apostrophe.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value) -> str:
    text = "" if value is None else str(value)
    return f"'{text}" if text.startswith(FORMULA_PREFIXES) else text


def dashboard_summary() -> dict:
    completed = EvaluationRun.objects.filter(status=JobStatus.COMPLETED).select_related("rag_config", "dataset")
    latest = completed.order_by("-finished_at").first()
    latest_security = completed.exclude(run_type="quality").order_by("-finished_at").first()
    return {
        "documents": {
            "total": Document.objects.count(),
            "by_status": {row["status"]: row["count"] for row in Document.objects.values("status").annotate(count=Count("id"))},
        },
        "chunks": DocumentChunk.objects.count(),
        "datasets": EvaluationDataset.objects.count(),
        "runs": {
            "total": EvaluationRun.objects.count(),
            "by_status": {
                row["status"]: row["count"] for row in EvaluationRun.objects.values("status").annotate(count=Count("id"))
            },
        },
        "latest_run": _run_card(latest) if latest else None,
        "latest_security_run": _run_card(latest_security) if latest_security else None,
        "recent_runs": [_run_card(run) for run in completed.order_by("-finished_at")[:10]],
        "live": live_metrics(7),
    }


def _run_card(run: EvaluationRun) -> dict:
    return {
        "id": run.id,
        "name": run.name or f"Run {run.id}",
        "run_type": run.run_type,
        "rag_config": run.rag_config.name,
        "dataset": str(run.dataset) if run.dataset else None,
        "finished_at": run.finished_at,
        "overall_quality": (run.summary or {}).get("overall_quality"),
        "metrics_scored": (run.summary or {}).get("metrics_scored", 0),
        "metrics": (run.summary or {}).get("metrics", {}),
    }


def run_report(run: EvaluationRun) -> dict:
    return {
        "run": {
            "id": run.id,
            "name": run.name,
            "type": run.run_type,
            "status": run.status,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
        },
        "snapshot": run.snapshot,
        "summary": run.summary,
        "cases": [
            {
                "question": result.case.question,
                "reference_answer": result.case.reference_answer,
                "answer": result.answer,
                "abstained": result.abstained,
                "blocked": result.blocked,
                "total_ms": result.total_ms,
                "scores": {score.metric.name: score.score for score in result.scores.all() if score.framework == "builtin"},
            }
            for result in run.results.select_related("case").prefetch_related("scores__metric")
        ],
        "security": [
            {
                "test": result.test.test_name,
                "attack_type": result.test.attack_type,
                "severity": result.severity,
                "passed": result.passed,
                "guard_blocked": result.guard_blocked,
                "attack_succeeded": result.attack_succeeded,
            }
            for result in SecurityResult.objects.filter(run=run).select_related("test")
        ],
    }


def run_report_csv(run: EvaluationRun) -> str:
    """One row per question, one column per metric."""
    results = list(run.results.select_related("case").prefetch_related("scores__metric"))
    metric_names = sorted({s.metric.name for r in results for s in r.scores.all() if s.framework == "builtin"})
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["question", "answer", "abstained", "blocked", "total_ms", *metric_names])
    for result in results:
        scores = {s.metric.name: s.score for s in result.scores.all() if s.framework == "builtin"}
        writer.writerow(
            [
                _csv_safe(result.case.question),
                _csv_safe(result.answer),
                result.abstained,
                result.blocked,
                round(result.total_ms, 1),
                *[("" if name not in scores else round(scores[name], 4)) for name in metric_names],
            ]
        )
    return output.getvalue()


def record_export(run: EvaluationRun, report_format: str, user) -> None:
    Report.objects.create(run=run, format=report_format, created_by=user)
