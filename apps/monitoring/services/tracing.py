"""Tracing: OpenTelemetry spans for every pipeline stage, and a Langfuse trace per question.

Both are optional. With no OTLP endpoint the spans go nowhere, and with no Langfuse keys the Langfuse
calls are skipped. A tracing failure must never fail a user's question.
"""

import logging
from contextlib import contextmanager
from functools import lru_cache

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config.env import env

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _tracer():
    provider = TracerProvider(resource=Resource.create({"service.name": "rag-evaluation-platform"}))
    if env.OTEL_EXPORTER_OTLP_ENDPOINT:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{env.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces"))
        )
    return provider.get_tracer("rag-evaluation-platform")


@contextmanager
def span(name: str, **attributes):
    with _tracer().start_as_current_span(name, attributes=attributes) as current:
        yield current


@lru_cache(maxsize=1)
def _langfuse():
    if not (env.LANGFUSE_PUBLIC_KEY and env.LANGFUSE_SECRET_KEY):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=env.LANGFUSE_PUBLIC_KEY, secret_key=env.LANGFUSE_SECRET_KEY, host=env.LANGFUSE_HOST
        )
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
        with client.start_as_current_observation(
            as_type="generation", name=name, model=model, input=question, metadata=metadata
        ) as generation:
            generation.update(
                output=answer, usage_details={"input": prompt_tokens, "output": completion_tokens}
            )
        client.flush()
    except Exception:
        logger.exception("Langfuse export failed; the question itself is unaffected")
