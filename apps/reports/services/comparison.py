"""Experiment comparison: the same metrics side by side, and which settings differed between the runs."""

from apps.evaluation.models import EvaluationRun

IGNORED_SETTINGS = {"name"}


def _better(values: dict[int, float], higher_is_better: bool) -> int | None:
    """The run with the best value, or None if the leaders are tied at the two decimals shown. On a small
    dataset, calling 0.968 better than 0.965 would present noise as a finding."""
    if len(values) < 2:
        return None
    ranked = sorted(values, key=values.get, reverse=higher_is_better)
    if round(values[ranked[0]], 2) == round(values[ranked[1]], 2):
        return None
    return ranked[0]


def compare_runs(run_ids: list[int]) -> dict:
    runs = {run.id: run for run in EvaluationRun.objects.select_related("rag_config", "dataset").filter(id__in=run_ids)}
    ordered = [runs[run_id] for run_id in run_ids if run_id in runs]

    keys: dict[str, dict] = {}
    for run in ordered:
        for key, metric in (run.summary.get("metrics") or {}).items():
            keys.setdefault(key, metric)

    rows = []
    for key, metric in keys.items():
        values = {}
        for run in ordered:
            entry = (run.summary.get("metrics") or {}).get(key)
            if entry is not None:
                values[run.id] = entry["mean"]
        rows.append(
            {
                "key": key,
                "metric": metric["metric"],
                "framework": metric["framework"],
                "display_name": metric["display_name"],
                "category": metric["category"],
                "order": metric["order"],
                "priority": metric["priority"],
                "higher_is_better": metric["higher_is_better"],
                "unit": metric["unit"],
                "values": values,
                "best_run": _better(values, metric["higher_is_better"]),
            }
        )

    rows.sort(key=lambda row: (row["order"], row["framework"]))

    settings_by_run = {run.id: (run.snapshot.get("rag_config") or {}) for run in ordered}
    names = sorted({name for snapshot in settings_by_run.values() for name in snapshot} - IGNORED_SETTINGS)
    differing = [
        {"setting": name, "values": {run_id: snapshot.get(name) for run_id, snapshot in settings_by_run.items()}}
        for name in names
        if len({snapshot.get(name) for snapshot in settings_by_run.values()}) > 1
    ]

    return {
        "runs": [
            {
                "id": run.id,
                "name": run.name or f"Run {run.id}",
                "rag_config": run.rag_config.name,
                "dataset": str(run.dataset) if run.dataset else None,
                "status": run.status,
                "overall_quality": (run.summary or {}).get("overall_quality"),
            }
            for run in ordered
        ],
        "metrics": rows,
        "differing_settings": differing,
    }
