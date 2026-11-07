"""The only place that talks to an LLM. LiteLLM lets a RagConfig switch provider by changing a model
string, for example ollama_chat/qwen2.5:1.5b or openai/gpt-4o-mini."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from apps.common.exceptions import ServiceUnavailable
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


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


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
        num_retries=1,
        **kwargs,
    )
    started = time.perf_counter()
    try:
        with span("llm.complete", model=model, streamed=on_token is not None):
            if on_token is None:
                response = litellm.completion(**request)
            else:
                parts = []
                for part in litellm.completion(**request, stream=True):
                    parts.append(part)
                    text = part.choices[0].delta.content if part.choices else None
                    if text:
                        on_token(text)
                # Puts the pieces back together, with token usage, as if it had not been streamed.
                response = litellm.stream_chunk_builder(parts, messages=messages)
    except Exception as exc:  # LiteLLM wraps provider errors in many exception types
        raise ServiceUnavailable(f"LLM call to {model} failed: {exc}") from exc

    usage = response.usage
    return LLMResponse(
        text=(response.choices[0].message.content or "").strip(),
        model=model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
