"""The models Ollama has installed, which of them can answer questions, and which fit in the memory free right now.

The chat lets people pick one. A model that does not fit is refused rather than tried: on this machine, running out of
memory has made the kernel kill the user's browser and editor, and swap is already nearly full. Free memory is read
from MemAvailable, without swap, plus whatever Ollama holds now, because Ollama unloads an idle model to make room.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from apps.common.exceptions import ServiceUnavailable
from config.env import env

PREFIX = "ollama_chat/"
# Models that turn text into vectors and cannot answer questions.
EMBEDDING_FAMILIES = {"bert", "nomic-bert", "clip"}
# Loaded size is about the file plus a tenth, plus the context cache and runtime for the configured window.
GROWTH = 1.1
OVERHEAD_BYTES = 400 * 1024**2
GB = 1000**3  # decimal, as `ollama ls` shows sizes


@dataclass(frozen=True)
class LocalModel:
    name: str  # as Ollama lists it, such as "qwen2.5:3b"
    model: str  # as a configuration or request names it, such as "ollama_chat/qwen2.5:3b"
    size_bytes: int
    parameters: str
    family: str
    quantization: str
    can_chat: bool
    needed_bytes: int
    loaded: bool
    fits: bool

    def as_dict(self) -> dict:
        return asdict(self)


def available_bytes() -> int:
    """MemAvailable from /proc/meminfo, in bytes. Swap is not counted: a model in swap is too slow to use."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def needed_bytes(size_bytes: int) -> int:
    return int(size_bytes * GROWTH) + OVERHEAD_BYTES


def list_models() -> tuple[list[LocalModel], int]:
    """Installed models, chat models first and smallest first, and the memory a newly loaded model could use."""
    try:
        tags = httpx.get(f"{env.OLLAMA_BASE_URL}/api/tags", timeout=5).json().get("models", [])
        running = httpx.get(f"{env.OLLAMA_BASE_URL}/api/ps", timeout=5).json().get("models", [])
    except (httpx.HTTPError, ValueError) as exc:
        raise ServiceUnavailable(f"Could not reach Ollama at {env.OLLAMA_BASE_URL}: {exc}") from exc
    loaded = {model["name"]: int(model.get("size", 0)) for model in running}
    free = available_bytes() + sum(loaded.values())
    models = []
    for tag in tags:
        details = tag.get("details") or {}
        family = str(details.get("family") or "")
        need = needed_bytes(int(tag.get("size", 0)))
        models.append(
            LocalModel(
                name=tag["name"],
                model=PREFIX + tag["name"],
                size_bytes=int(tag.get("size", 0)),
                parameters=str(details.get("parameter_size") or ""),
                family=family,
                quantization=str(details.get("quantization_level") or ""),
                can_chat=family not in EMBEDDING_FAMILIES and "embed" not in tag["name"],
                needed_bytes=need,
                loaded=tag["name"] in loaded,
                fits=tag["name"] in loaded or need <= free,
            )
        )
    models.sort(key=lambda model: (not model.can_chat, model.size_bytes))
    return models, free


def why_not(model: str) -> str:
    """Why this model cannot answer now, or "" if it can. Only installed chat models are accepted, so a request cannot
    send questions to a hosted provider."""
    models, free = list_models()
    found = next((candidate for candidate in models if candidate.model == model), None)
    if not found:
        return f"{model.removeprefix(PREFIX)} is not installed in Ollama."
    if not found.can_chat:
        return f"{found.name} turns text into vectors and cannot answer questions."
    if not found.fits:
        return (
            f"{found.name} needs about {math.ceil(found.needed_bytes / GB * 10) / 10:.1f} GB of memory and "
            f"{math.floor(free / GB * 10) / 10:.1f} GB is free. "
            "Close other apps, or pick a smaller model."
        )
    return ""
