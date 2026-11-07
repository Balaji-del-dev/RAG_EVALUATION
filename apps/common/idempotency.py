"""Idempotency keys: a client that retries a POST after a timeout sends the same Idempotency-Key header, and gets
the first response again instead of a second answer or a duplicate upload."""

import hashlib
import json
import re
from functools import wraps

from django.core.cache import cache
from rest_framework.response import Response

KEY_RE = re.compile(r"[A-Za-z0-9_-]{8,128}")
TTL_SECONDS = 24 * 3600
RUNNING_SECONDS = 600


def _fingerprint(request) -> str:
    try:
        body = json.dumps(request.data, sort_keys=True, default=str)
    except (TypeError, ValueError):
        body = str(request.data)
    return hashlib.sha256(body.encode()).hexdigest()


def idempotent(method):
    """For a DRF view method. Without the header, the request runs as usual."""

    @wraps(method)
    def wrapper(view, request, *args, **kwargs):
        key = request.headers.get("Idempotency-Key", "").strip()
        if not key:
            return method(view, request, *args, **kwargs)
        if not KEY_RE.fullmatch(key):
            return Response({"error": "Idempotency-Key must be 8 to 128 letters, digits, - or _."}, status=400)
        stored = f"idempotency:{request.user.pk}:{request.path}:{key}"
        fingerprint = _fingerprint(request)
        if (saved := cache.get(stored)) is not None:
            if saved["fingerprint"] != fingerprint:
                return Response({"error": "This Idempotency-Key was used with a different request."}, status=422)
            response = Response(saved["data"], status=saved["status"])
            response["Idempotent-Replay"] = "true"
            return response
        if not cache.add(stored + ":running", 1, timeout=RUNNING_SECONDS):
            return Response({"error": "A request with this Idempotency-Key is still running."}, status=409)
        try:
            response = method(view, request, *args, **kwargs)
            # A server error may succeed on retry, so only answers the client should keep are saved.
            if response.status_code < 500:
                cache.set(
                    stored,
                    {"fingerprint": fingerprint, "status": response.status_code, "data": response.data},
                    TTL_SECONDS,
                )
            return response
        finally:
            cache.delete(stored + ":running")

    return wrapper
