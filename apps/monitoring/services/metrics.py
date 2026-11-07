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
FIRST_TOKEN_SECONDS = Histogram(
    "rag_first_token_seconds",
    "Time from question to the first words on screen",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 10, 20, 30, 60),
)
TOKENS_PER_SECOND = Histogram(
    "rag_llm_tokens_per_second", "How fast the model wrote an answer", buckets=(1, 2, 5, 10, 15, 20, 30, 50, 100)
)
CACHE = Counter("rag_answer_cache_total", "Answer cache lookups", ["result"])
STAGE_SECONDS = Histogram(
    "rag_stage_seconds",
    "Time spent in each pipeline stage: embed, vector, bm25, rerank, mmr, expand, retrieval, generation",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
ROUTES = Counter(
    "rag_routes_total", "Chat answers by how they were made: search, library, conversation, ...", ["route"]
)
GUARD_BLOCKS = Counter("rag_guard_blocks_total", "Answers a guard blocked, by reason", ["reason"])
LLM_PROBLEMS = Counter(
    "rag_llm_problems_total",
    "LLM calls retried, failed, stopped for repeating, or refused by an open circuit",
    ["kind"],
)


def outcome_kind(outcome) -> str:
    if outcome.blocked:
        return "blocked"
    if outcome.abstained:
        return "declined"
    return "quoted" if outcome.quoted else "answered"


def record_cache(hit: bool) -> None:
    CACHE.labels("hit" if hit else "miss").inc()


def record_llm_problem(kind: str) -> None:
    LLM_PROBLEMS.labels(kind).inc()


def record_answer(outcome) -> None:
    ANSWERS.labels(outcome_kind(outcome)).inc()
    ANSWER_SECONDS.observe(outcome.total_ms / 1000)
    if outcome.first_token_ms:
        FIRST_TOKEN_SECONDS.observe(outcome.first_token_ms / 1000)
    if outcome.tokens_per_second:
        TOKENS_PER_SECOND.observe(outcome.tokens_per_second)
    if outcome.blocked:
        GUARD_BLOCKS.labels(outcome.block_reason or "unknown").inc()
    ROUTES.labels(outcome.route).inc()
    if outcome.cached:
        return  # nothing ran, so there are no stage times to record
    for stage, ms in outcome.stage_timings_ms.items():
        STAGE_SECONDS.labels(stage).observe(ms / 1000)
    for stage, ms in (("retrieval", outcome.retrieval_ms), ("generation", outcome.generation_ms)):
        if ms:
            STAGE_SECONDS.labels(stage).observe(ms / 1000)
    if outcome.prompt_tokens:
        LLM_TOKENS.labels("prompt").inc(outcome.prompt_tokens)
    if outcome.completion_tokens:
        LLM_TOKENS.labels("completion").inc(outcome.completion_tokens)
