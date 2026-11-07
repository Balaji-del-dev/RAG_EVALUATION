"""Tracing: OpenTelemetry spans for every pipeline stage, and a Langfuse trace per question.

Both are optional. With no OTLP endpoint and no Langfuse keys the spans go nowhere. A tracing failure must never fail
a user's question.

With Langfuse keys, the spans also go to Langfuse's OpenTelemetry endpoint, so each question there shows its stages
(guards, search, rerank, the model call) with their real durations. Each model call is marked as a generation, with
its model and token counts but not its prompt, which holds the source text. The question and answer are attached
once per question, with personal details masked (see rag_workflow._for_traces).

Only question traffic is sent: chat, streaming chat, search, and evaluation runs. The dashboard polls the API every few
seconds, and a trace for each of those requests would bury the questions.
"""

import base64
import json
import logging
from contextlib import contextmanager
from functools import lru_cache

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config.env import env

logger = logging.getLogger(__name__)

# Requests whose trace is worth keeping: the ones that run the pipeline.
TRACED_PATHS = {"/api/rag/query/": "chat", "/api/rag/query/stream/": "chat", "/api/rag/retrieve/": "search"}


class QuestionsOnly(SpanProcessor):
    """Passes on every span except the request span of a request that ran no pipeline."""

    def __init__(self, inner: SpanProcessor):
        self.inner = inner

    def on_start(self, span, parent_context=None):
        self.inner.on_start(span, parent_context=parent_context)

    def on_end(self, span):
        if span.name == "http.request" and span.attributes.get("http.target") not in TRACED_PATHS:
            return
        self.inner.on_end(span)

    def shutdown(self):
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000):
        return self.inner.force_flush(timeout_millis)


def _langfuse_exporter():
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    token = base64.b64encode(f"{env.LANGFUSE_PUBLIC_KEY}:{env.LANGFUSE_SECRET_KEY}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=f"{env.LANGFUSE_HOST.rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {token}"},
    )


@lru_cache(maxsize=1)
def _tracer():
    provider = TracerProvider(resource=Resource.create({"service.name": "rag-evaluation-platform"}))
    if env.OTEL_EXPORTER_OTLP_ENDPOINT:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{env.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces"))
        )
    if env.LANGFUSE_PUBLIC_KEY and env.LANGFUSE_SECRET_KEY:
        provider.add_span_processor(QuestionsOnly(BatchSpanProcessor(_langfuse_exporter())))
    return provider.get_tracer("rag-evaluation-platform")


def mark_generation(current, *, model: str, prompt_tokens: int, completion_tokens: int, completion_start=None):
    """Marks a model-call span as a Langfuse generation: its model, token counts, and when the first token came
    (`completion_start`, a datetime, when streamed). The span's own start and end give the real duration."""
    current.set_attribute("langfuse.observation.type", "generation")
    current.set_attribute("langfuse.observation.model.name", model)
    current.set_attribute(
        "langfuse.observation.usage_details", json.dumps({"input": prompt_tokens, "output": completion_tokens})
    )
    if completion_start:
        current.set_attribute("langfuse.observation.completion_start_time", completion_start.isoformat())


@contextmanager
def request_span(request):
    """A root span for one HTTP request, yielding its trace ID as 32 hex digits."""
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    carrier = {"traceparent": request.META.get("HTTP_TRACEPARENT", "")}
    parent = TraceContextTextMapPropagator().extract(carrier) if carrier["traceparent"] else None
    attributes = {"http.method": request.method, "http.target": request.path, "request.id": request.request_id}
    if request.path in TRACED_PATHS:
        attributes["langfuse.trace.name"] = TRACED_PATHS[request.path]
    with _tracer().start_as_current_span("http.request", context=parent, attributes=attributes) as current:
        yield format(current.get_span_context().trace_id, "032x")


@contextmanager
def span(name: str, **attributes):
    from apps.common.middleware import current_request_id

    if current_request_id.get():
        attributes.setdefault("request.id", current_request_id.get())
    with _tracer().start_as_current_span(name, attributes=attributes) as current:
        yield current


@lru_cache(maxsize=1)
def _langfuse():
    if not (env.LANGFUSE_PUBLIC_KEY and env.LANGFUSE_SECRET_KEY):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(public_key=env.LANGFUSE_PUBLIC_KEY, secret_key=env.LANGFUSE_SECRET_KEY, host=env.LANGFUSE_HOST)
    except Exception:
        logger.exception("Could not start Langfuse; continuing without it")
        return None


def langfuse_enabled() -> bool:
    return _langfuse() is not None


def record_generation(*, name, model, question, answer, prompt_tokens, completion_tokens, metadata) -> None:
    """Send one finished LLM call to Langfuse."""
    client = _langfuse()
    if client is None:
        return
    try:
        # A span, not a generation: the model call itself is the generation (see mark_generation), and counting the
        # tokens twice would double the usage Langfuse reports.
        with client.start_as_current_observation(
            as_type="span", name=name, input=question, metadata={**metadata, "model": model}
        ) as observation:
            observation.update(
                output=answer, metadata={**metadata, "model": model, "tokens": [prompt_tokens, completion_tokens]}
            )
        client.flush()
    except Exception:
        logger.exception("Langfuse export failed; the question itself is unaffected")
