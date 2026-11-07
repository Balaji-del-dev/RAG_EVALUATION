"""The only place that talks to an LLM. LiteLLM lets a RagConfig switch provider by changing a model
string, for example ollama_chat/qwen2.5:1.5b or openai/gpt-4o-mini."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from django.core.cache import cache

from apps.common.exceptions import ServiceUnavailable
from apps.monitoring.services import metrics
from apps.monitoring.services.tracing import span
from config.env import env, split_csv

logger = logging.getLogger(__name__)


def _ollama_name(model: str) -> str | None:
    return model.split("/", 1)[1] if model.startswith("ollama") and "/" in model else None


def unload(model: str) -> None:
    """Ask Ollama to free the model's RAM now. Failure only means that the RAM is freed later."""
    name = _ollama_name(model)
    if name:
        try:
            httpx.post(f"{env.OLLAMA_BASE_URL}/api/generate", json={"model": name, "keep_alive": 0}, timeout=10)
        except httpx.HTTPError as exc:
            logger.info("Could not unload %s: %s", name, exc)


def model_digest(model: str) -> str:
    """The exact weights behind a tag, for reproducibility: tags such as "latest" can change."""
    name = _ollama_name(model)
    if not name:
        return ""
    try:
        tags = httpx.get(f"{env.OLLAMA_BASE_URL}/api/tags", timeout=5).json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return ""
    return next((tag.get("digest", "") for tag in tags if tag.get("name") == name), "")


# A failing model is retried twice, after 1 s and 3 s. Three failures in a row pause calls to it for 30 s, so a down
# Ollama gives a quick, clear error instead of every question waiting out its own timeouts.
RETRY_DELAYS = (1.0, 3.0)
FAILURES_TO_OPEN = 3
OPEN_SECONDS = 30


def _breaker(model: str) -> str:
    return f"llm-breaker:{model}"


def _record_failure(model: str) -> None:
    failures = (cache.get(_breaker(model)) or 0) + 1
    cache.set(_breaker(model), failures, OPEN_SECONDS * 4)
    if failures >= FAILURES_TO_OPEN:
        cache.set(_breaker(model) + ":open", True, OPEN_SECONDS)
        cache.delete(_breaker(model))


def _is_timeout(exc: Exception) -> bool:
    return "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()


# Ollama stops a model that repeats the same token too long. It is the model's failure, not the service's, so it is
# told apart: the answer can be asked for again, where a service failure cannot.
REPEAT_ABORT = "token repeat limit reached"


class RepetitionAborted(ServiceUnavailable):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    first_token_ms: float = 0.0  # when streamed: how long until the first text arrived


def complete(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    on_token: Callable[[str], None] | None = None,
) -> LLMResponse:
    """One chat completion. With on_token, the text is streamed to it as it is written, and the full response is
    still returned at the end. Models in LLM_FALLBACK_MODELS are tried in turn if this one fails."""
    import litellm  # slow import (several seconds), so it waits until the first LLM call

    litellm.suppress_debug_info = True
    kwargs = {}
    if model.startswith("ollama"):
        kwargs = {
            "api_base": env.OLLAMA_BASE_URL,
            "keep_alive": env.OLLAMA_KEEP_ALIVE,
            "num_ctx": env.OLLAMA_NUM_CTX,
            "num_thread": env.OLLAMA_NUM_THREAD,
        }

    if fallbacks := split_csv(env.LLM_FALLBACK_MODELS):
        kwargs["fallbacks"] = fallbacks
    request = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=env.LLM_TIMEOUT_SECONDS,
        num_retries=0,  # retried here, with backoff, so that failures can open the circuit breaker
        **kwargs,
    )
    if cache.get(_breaker(model) + ":open"):
        metrics.record_llm_problem("circuit_open")
        raise ServiceUnavailable(
            f"{model} failed {FAILURES_TO_OPEN} times in a row, so calls to it pause for {OPEN_SECONDS} seconds. "
            "Try again shortly."
        )
    started = time.perf_counter()
    first_token: list[float] = []

    def call():
        with span("llm.complete", model=model, streamed=on_token is not None):
            if on_token is None:
                return litellm.completion(**request)
            parts = []
            for part in litellm.completion(**request, stream=True):
                parts.append(part)
                text = part.choices[0].delta.content if part.choices else None
                if text:
                    if not first_token:
                        first_token.append((time.perf_counter() - started) * 1000)
                    on_token(text)
            # Puts the pieces back together, with token usage, as if it had not been streamed.
            return litellm.stream_chunk_builder(parts, messages=messages)

    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            response = call()
            break
        except Exception as exc:  # LiteLLM wraps provider errors in many exception types
            if REPEAT_ABORT in str(exc):
                metrics.record_llm_problem("repetition")
                raise RepetitionAborted(f"{model} kept repeating itself and Ollama stopped it") from exc
            # A timeout already waited long; text already shown cannot be taken back. Neither is retried.
            if attempt < len(RETRY_DELAYS) and not first_token and not _is_timeout(exc):
                logger.info("LLM call to %s failed, retrying in %ss: %s", model, RETRY_DELAYS[attempt], exc)
                metrics.record_llm_problem("retry")
                time.sleep(RETRY_DELAYS[attempt])
                continue
            _record_failure(model)
            metrics.record_llm_problem("failed")
            raise ServiceUnavailable(f"LLM call to {model} failed: {exc}") from exc
    cache.delete(_breaker(model))

    usage = response.usage
    return LLMResponse(
        text=(response.choices[0].message.content or "").strip(),
        model=model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        latency_ms=(time.perf_counter() - started) * 1000,
        first_token_ms=first_token[0] if first_token else 0.0,
    )
