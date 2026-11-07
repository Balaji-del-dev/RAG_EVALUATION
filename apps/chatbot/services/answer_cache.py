"""Answers kept for a while, so the same question, asked again with the same permissions, settings and documents, is
answered at once instead of by a 20-second model call.

The key holds everything that could change the answer: the question (spacing and end punctuation aside), the
configuration and its model, the documents the asker may read, a version number that goes up whenever a document or
configuration changes, the pipeline code, and the date. So a cached answer is never served to someone with different
access, nor after an upload or a deploy.

Only exact repeats are served. A semantic cache, matching questions that merely look alike, was rejected: "may month
salary" and "march month salary" look almost the same to an embedding, and would get each other's figures.
"""

import hashlib
import json
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from django.core.cache import cache
from django.utils import timezone

from apps.retrieval.services.hybrid_search import RetrievedChunk
from config.env import env

VERSION_KEY = "answer-cache:version"
# The code that shapes an answer. Redis keeps entries across restarts, so a deploy that changes any of it must not
# serve answers the old code wrote.
PIPELINE_FILES = [
    "apps/chatbot/services/rag_workflow.py",
    "apps/chatbot/services/generation.py",
    "apps/chatbot/services/formats.py",
    "apps/chatbot/services/fields.py",
    "apps/chatbot/services/verification.py",
    "apps/retrieval/services/hybrid_search.py",
    "apps/security/services/guardrails.py",
]


@lru_cache(maxsize=1)
def code_version() -> str:
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for name in PIPELINE_FILES:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()[:16]


def version() -> int:
    return cache.get_or_set(VERSION_KEY, 1, None)


def invalidate(*args, **kwargs) -> None:
    """Called when a document or configuration changes. Old entries then never match, and expire on their own."""
    try:
        cache.incr(VERSION_KEY)
    except ValueError:
        cache.set(VERSION_KEY, 2, None)


def _normal(question: str) -> str:
    # Case is kept: the input guard's models can read "IGNORE" and "ignore" differently, and a cache hit skips them.
    return re.sub(r"\s+", " ", question).strip(" ?.!")


def _key(question: str, config, allowed_document_ids: list[int]) -> str:
    parts = [
        _normal(question),
        config.pk,
        config.llm_model,
        config.prompt_version,
        config.index_key,
        sorted(allowed_document_ids),
        version(),
        code_version(),
        # "this month" means another month tomorrow.
        timezone.localdate().isoformat(),
    ]
    return "answer-cache:" + hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()


def cacheable(outcome) -> bool:
    """Only clean answers are kept. A decline, an answer with figures to double-check, or one where the search had to
    fall back may come out better next time, so asking again should try again. Greetings have no search stages and
    cost nothing to repeat. Anything a guard masked or blocked is left out too."""
    return (
        bool(outcome.stages)
        and not (outcome.blocked or outcome.abstained or outcome.findings or outcome.checks)
        and bool(outcome.answer.strip())
    )


def get(question: str, config, allowed_document_ids: list[int]):
    if env.ANSWER_CACHE_SECONDS <= 0:
        return None
    from apps.chatbot.services.rag_workflow import RagOutcome

    data = cache.get(_key(question, config, allowed_document_ids))
    if not data:
        return None
    outcome = RagOutcome(**{**data, "chunks": [RetrievedChunk(**chunk) for chunk in data["chunks"]]})
    # Nothing was generated this time: no model, no tokens, no cost.
    outcome.cached, outcome.model, outcome.prompt_tokens, outcome.completion_tokens = True, "", 0, 0
    outcome.retrieval_ms = outcome.generation_ms = outcome.first_token_ms = outcome.tokens_per_second = 0.0
    return outcome


def put(question: str, config, allowed_document_ids: list[int], outcome) -> None:
    if env.ANSWER_CACHE_SECONDS > 0 and cacheable(outcome):
        cache.set(_key(question, config, allowed_document_ids), asdict(outcome), env.ANSWER_CACHE_SECONDS)
