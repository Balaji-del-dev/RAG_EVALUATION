"""Request IDs: every call gets one, so that a log line, a trace span, a saved answer and a user's report of a
problem can all be matched up."""

import contextvars
import logging
import re
import uuid

current_request_id = contextvars.ContextVar("request_id", default="")
# A caller's own ID is kept only if it is plainly an ID, so that it cannot inject text into the logs.
VALID_ID_RE = re.compile(r"[A-Za-z0-9._-]{8,64}")


def _valid(value: str | None) -> str:
    return value if value and VALID_ID_RE.fullmatch(value) else ""


class RequestIdMiddleware:
    """Uses the caller's X-Request-ID when it is valid, or makes one, and returns it. X-Correlation-ID groups the
    requests of one larger action, such as everything a page load sends, and defaults to the request ID."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.request_id = _valid(request.META.get("HTTP_X_REQUEST_ID")) or uuid.uuid4().hex
        request.correlation_id = _valid(request.META.get("HTTP_X_CORRELATION_ID")) or request.request_id
        token = current_request_id.set(request.request_id)
        try:
            response = self.get_response(request)
        finally:
            current_request_id.reset(token)
        response["X-Request-ID"] = request.request_id
        response["X-Correlation-ID"] = request.correlation_id
        return response


class RequestIdLogFilter(logging.Filter):
    """Adds the current request ID to every log line, or "-" outside a request, as in the Celery worker."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id.get() or "-"
        return True
