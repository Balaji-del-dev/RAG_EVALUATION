"""Live-traffic metrics, computed in SQL so that dashboards never load raw rows into Python."""

from datetime import timedelta

from django.db import connection
from django.utils import timezone

SUMMARY_SQL = """
SELECT
    COUNT(*)                                                                         AS requests,
    COUNT(*) FILTER (WHERE NOT p.succeeded)                                          AS errors,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY p.total_ms) FILTER (WHERE p.succeeded)     AS p50_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY p.total_ms) FILTER (WHERE p.succeeded)     AS p95_ms,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY p.total_ms) FILTER (WHERE p.succeeded)     AS p99_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY p.retrieval_ms) FILTER (WHERE p.succeeded) AS retrieval_p95_ms,
    COALESCE(SUM(t.input_tokens), 0)                                                 AS input_tokens,
    COALESCE(SUM(t.output_tokens), 0)                                                AS output_tokens,
    COALESCE(SUM(c.input_cost + c.output_cost), 0)                                   AS total_cost
FROM performance_logs p
LEFT JOIN token_usage t ON t.message_id = p.message_id
LEFT JOIN cost_records c ON c.message_id = p.message_id
WHERE p.created_at >= %s
"""

DAILY_SQL = """
SELECT
    date_trunc('day', p.created_at)                                                  AS day,
    COUNT(*)                                                                         AS requests,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY p.total_ms) FILTER (WHERE p.succeeded) AS p95_ms
FROM performance_logs p
WHERE p.created_at >= %s
GROUP BY 1
ORDER BY 1
"""


def _fetch_dicts(sql: str, params: list) -> list[dict]:
    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [column.name for column in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def live_metrics(days: int = 7) -> dict:
    since = timezone.now() - timedelta(days=days)
    summary = _fetch_dicts(SUMMARY_SQL, [since])[0]
    requests = summary["requests"]
    return {
        "days": days,
        "requests": requests,
        "errors": summary["errors"],
        "error_rate": summary["errors"] / requests if requests else 0.0,
        "p50_ms": summary["p50_ms"] or 0.0,
        "p95_ms": summary["p95_ms"] or 0.0,
        "p99_ms": summary["p99_ms"] or 0.0,
        "retrieval_p95_ms": summary["retrieval_p95_ms"] or 0.0,
        "input_tokens": summary["input_tokens"],
        "output_tokens": summary["output_tokens"],
        "total_cost": float(summary["total_cost"]),
        "cost_per_query": float(summary["total_cost"]) / requests if requests else 0.0,
        "daily": [
            {"day": row["day"].date().isoformat(), "requests": row["requests"], "p95_ms": row["p95_ms"] or 0.0}
            for row in _fetch_dicts(DAILY_SQL, [since])
        ],
    }
