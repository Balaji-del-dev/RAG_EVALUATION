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
        from apps.monitoring.services import tracing

        request.request_id = _valid(request.META.get("HTTP_X_REQUEST_ID")) or uuid.uuid4().hex
        request.correlation_id = _valid(request.META.get("HTTP_X_CORRELATION_ID")) or request.request_id
        token = current_request_id.set(request.request_id)
        try:
            # One trace per request, continuing the caller's when it sends a W3C traceparent header. Every pipeline
            # span is a child of this one, so a slow answer can be followed from the request down to the model call.
            with tracing.request_span(request) as trace_id:
                request.trace_id = trace_id
                response = self.get_response(request)
        finally:
            current_request_id.reset(token)
        response["X-Request-ID"] = request.request_id
        response["X-Correlation-ID"] = request.correlation_id
        response["X-Trace-ID"] = request.trace_id
        return response


class RequestSizeMiddleware:
    """413 for a body larger than the request may be, said before the body is read. Uploads may be as large as
    MAX_UPLOAD_BYTES; everything else is JSON and far smaller."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        from django.conf import settings
        from django.http import JsonResponse

        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        upload = request.content_type == "multipart/form-data"
        limit = settings.MAX_UPLOAD_BYTES + 1024 * 1024 if upload else settings.MAX_JSON_BODY_BYTES
        if length > limit:
            return JsonResponse(
                {"error": f"The request is {length / 1e6:.1f} MB, more than the {limit / 1e6:.0f} MB this accepts."},
                status=413,
            )
        return self.get_response(request)


class RequestIdLogFilter(logging.Filter):
    """Adds the current request ID to every log line, or "-" outside a request, as in the Celery worker."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id.get() or "-"
        return True
