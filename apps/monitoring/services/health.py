"""Reachability of each backing service. Each check is independent, so one outage cannot hide another."""

import httpx
from django.core.cache import cache
from django.db import connection

from apps.documents.services import storage
from apps.monitoring.services.tracing import langfuse_enabled
from apps.retrieval.services import bm25_search, vector_search
from config.env import env


def _check(fn) -> bool:
    try:
        return bool(fn())
    except Exception:  # a health check reports failures, it never raises them
        return False


def _database() -> bool:
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1")
    return True


def _redis() -> bool:
    cache.set("health", "1", 5)
    return cache.get("health") == "1"


def _ollama() -> bool:
    return httpx.get(f"{env.OLLAMA_BASE_URL}/api/tags", timeout=3).status_code == 200


def _worker() -> bool:
    from apps.evaluation.tasks import WORKER_HEARTBEAT_KEY

    return cache.get(WORKER_HEARTBEAT_KEY) is not None


def service_health() -> dict:
    bm25 = bm25_search.backend()
    return {
        "worker": _check(_worker),
        "postgresql": _check(_database),
        "redis": _check(_redis),
        "minio": _check(lambda: storage._client().bucket_exists(env.MINIO_BUCKET) is not None),
        "qdrant": _check(vector_search.ping),
        "ollama": _check(_ollama),
        "bm25": {"backend": bm25.name, "ok": _check(bm25.ping)},
        "langfuse": {"configured": langfuse_enabled()},
    }
