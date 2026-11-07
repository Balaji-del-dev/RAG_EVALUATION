"""Prometheus counters for the RAG pipeline, beside the HTTP, model and cache metrics django-prometheus collects.

Each process counts on its own. With several gunicorn workers, set PROMETHEUS_MULTIPROC_DIR as the prometheus_client
documentation describes, or each scrape sees whichever worker answered it.
"""

from prometheus_client import Counter, Histogram

ANSWERS = Counter("rag_answers_total", "Chat answers, by how they ended", ["outcome"])
ANSWER_SECONDS = Histogram(
    "rag_answer_seconds",
    "Time from question to finished answer",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)
LLM_TOKENS = Counter("rag_llm_tokens_total", "LLM tokens used by chat answers", ["kind"])
FEEDBACK = Counter("rag_feedback_total", "Thumbs up and down given to answers", ["rating"])


def outcome_kind(outcome) -> str:
    if outcome.blocked:
        return "blocked"
    if outcome.abstained:
        return "declined"
    return "quoted" if outcome.quoted else "answered"


def record_answer(outcome) -> None:
    ANSWERS.labels(outcome_kind(outcome)).inc()
    ANSWER_SECONDS.observe(outcome.total_ms / 1000)
    if outcome.prompt_tokens:
        LLM_TOKENS.labels("prompt").inc(outcome.prompt_tokens)
    if outcome.completion_tokens:
        LLM_TOKENS.labels("completion").inc(outcome.completion_tokens)
