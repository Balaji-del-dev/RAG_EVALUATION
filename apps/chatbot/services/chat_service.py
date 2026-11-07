"""One user question from start to finish: permissions, pipeline, and all the SQL records it leaves."""

import json
import logging
import re
import time
from contextlib import contextmanager

from django.db import transaction

from apps.chatbot.models import ChatMessage, ChatSession, LLMRun
from apps.chatbot.services import answer_cache, generation, routing
from apps.chatbot.services.rag_workflow import RagOutcome, for_traces, run_rag, stream_rag
from apps.common import deadline
from apps.common.exceptions import ServiceUnavailable
from apps.common.middleware import current_request_id
from apps.monitoring.models import CostRecord, PerformanceLog, TokenUsage
from apps.monitoring.services import metrics
from apps.monitoring.services.cost_tracker import compute_cost
from apps.monitoring.services.tracing import span
from apps.retrieval.models import RagConfig, RetrievalLog
from apps.security.models import AuditLog
from apps.security.services import guardrails
from apps.security.services.access_control import allowed_document_ids
from config.env import env

logger = logging.getLogger(__name__)


# A sentence ends at ., ! or ? followed by a space, or at a line break. "priya.raman@example.com" is not split.
SENTENCE_END_RE = re.compile(r"[.!?](?=\s)|\n")


def _begin(user, question: str, config: RagConfig, session: ChatSession | None) -> tuple[ChatMessage, dict]:
    if session is None:
        session = ChatSession.objects.create(user=user, title=question[:120], rag_config=config)
    earlier = previous_answer(session, question)
    message = ChatMessage.objects.create(session=session, question=question, request_id=current_request_id.get())
    # Saving moves this conversation to the top of the history list.
    session.save(update_fields=["updated_at"])
    context = {
        "earlier_question": (earlier.standalone_question or earlier.question) if earlier else None,
        "earlier_answer": earlier.answer if earlier else "",
        # Read only for a question about the conversation itself, such as "summarise this thread".
        "history": conversation_history(session) if routing.asks_about_conversation(question) else None,
    }
    return message, context


MAX_HISTORY = 20


def conversation_history(session: ChatSession) -> list[dict]:
    """The answered questions so far, oldest first. Blocked ones are left out: their text was never shown."""
    answered = session.messages.filter(blocked=False).exclude(answer="").order_by("-id")[:MAX_HISTORY]
    return [{"question": m.standalone_question or m.question, "answer": m.answer} for m in reversed(answered)]


def _complete_sentences(text: str) -> tuple[str, str]:
    """Split off the finished sentences, keeping the unfinished end for later."""
    ends = [match.end() for match in SENTENCE_END_RE.finditer(text)]
    return (text[: ends[-1]], text[ends[-1] :]) if ends else ("", text)


@contextmanager
def _chat_trace(user, session: ChatSession, question: str):
    """The root of a chat answer's trace in Langfuse: named "chat", grouped by conversation and user, with the
    question as its input. Every answer gets one, also those read from the documents by rule or from the cache, which
    make no model call. The search and model spans are its children."""
    attributes = {
        "langfuse.trace.name": "chat",
        "langfuse.internal.as_root": True,
        "session.id": str(session.id),
        "user.id": user.get_username(),
        "langfuse.observation.input": json.dumps(for_traces(question)),
    }
    with span("chat.answer", **attributes) as current:
        yield current


def _trace_outcome(current, outcome: RagOutcome, config: RagConfig) -> None:
    current.set_attribute("langfuse.observation.output", json.dumps(outcome.answer))
    details = {
        "route": outcome.route,
        "cached": outcome.cached,
        "quoted": outcome.quoted,
        "declined": outcome.abstained,
        "blocked": outcome.blocked,
        "model": outcome.model,
        "config": config.name,
        "format": outcome.answer_format,
        "total_ms": round(outcome.total_ms),
        "checks": sorted(outcome.checks),
        "chunk_ids": [chunk.chunk_id for chunk in outcome.chunks],
    }
    for key, value in details.items():
        current.set_attribute(f"langfuse.observation.metadata.{key}", json.dumps(value))


def _from_cache(question: str, config: RagConfig, allowed: list[int], context: dict, fresh: bool, started: float):
    """A cached answer to this exact question, or None. A follow-up depends on the conversation before it, so it is
    never served from the cache, nor kept in it."""
    if context["earlier_question"]:
        return None
    outcome = None if fresh else answer_cache.get(question, config, allowed)
    metrics.record_cache(hit=outcome is not None)
    if outcome:
        outcome.total_ms = outcome.first_token_ms = (time.perf_counter() - started) * 1000
    return outcome


def _keep(question: str, config: RagConfig, allowed: list[int], context: dict, outcome: RagOutcome) -> None:
    if not context["earlier_question"]:
        answer_cache.put(question, config, allowed, outcome)


def ask_streaming(user, question: str, config: RagConfig, session: ChatSession | None = None, fresh: bool = False):
    """ask(), yielding ("text", sentence) while the answer is written and then ("done", (message, outcome)).

    Each sentence passes the output guard before it is sent, because masking only the final answer would show PII
    on screen first. After a sentence the guard blocks, nothing more is sent, and the final answer replaces it all."""
    started = time.perf_counter()
    message, context = _begin(user, question, config, session)
    allowed = allowed_document_ids(user)
    with _chat_trace(user, message.session, question) as trace:
        if outcome := _from_cache(question, config, allowed, context, fresh, started):
            yield "text", outcome.answer
        else:
            pending, stopped, first_text_ms = "", False, 0.0
            try:
                with deadline.within(env.REQUEST_DEADLINE_SECONDS):
                    for kind, value in stream_rag(question, config, allowed, **context, interactive=True):
                        if kind == "outcome":
                            outcome = value
                            continue
                        if kind == "reset":
                            # The answer is being written again; what was sent so far will be replaced.
                            pending = ""
                            if not stopped:
                                yield "reset", ""
                            continue
                        ready, pending = _complete_sentences(pending + value)
                        if ready and not stopped:
                            verdict = guardrails.check_output(ready)
                            stopped = verdict.blocked
                            if not stopped:
                                first_text_ms = first_text_ms or (time.perf_counter() - started) * 1000
                                yield "text", verdict.text
            except ServiceUnavailable:
                PerformanceLog.objects.create(message=message, succeeded=False)
                raise
            # Nothing streamed (a quoted table, a decline): the first words came with the whole answer.
            outcome.first_token_ms = first_text_ms or outcome.total_ms
            _keep(question, config, allowed, context, outcome)
        _trace_outcome(trace, outcome, config)
    with transaction.atomic():
        _persist(message, outcome, config, user)
    yield "done", (message, outcome)


def ask(
    user, question: str, config: RagConfig, session: ChatSession | None = None, fresh: bool = False
) -> tuple[ChatMessage, RagOutcome]:
    started = time.perf_counter()
    message, context = _begin(user, question, config, session)
    allowed = allowed_document_ids(user)
    with _chat_trace(user, message.session, question) as trace:
        outcome = _from_cache(question, config, allowed, context, fresh, started)
        if outcome is None:
            try:
                with deadline.within(env.REQUEST_DEADLINE_SECONDS):
                    outcome = run_rag(question, config, allowed, **context, interactive=True)
            except ServiceUnavailable:
                # Failed requests still count: they are what the error-rate metric measures.
                PerformanceLog.objects.create(message=message, succeeded=False)
                raise
            _keep(question, config, allowed, context, outcome)
        _trace_outcome(trace, outcome, config)
    with transaction.atomic():
        _persist(message, outcome, config, user)
    return message, outcome


def previous_answer(session: ChatSession, question: str) -> ChatMessage | None:
    """The last answered message, if this question depends on it. Its question is used in complete form, so a chain
    of follow-ups keeps its meaning, and its answer only to find which month "this month" is. A standalone question
    gets none, so earlier topics cannot pull its search off course."""
    if not generation.is_follow_up(question):
        return None
    answered = session.messages.filter(blocked=False, abstained=False).exclude(answer="")
    return answered.order_by("-id").first()


def _persist(message: ChatMessage, outcome: RagOutcome, config: RagConfig, user) -> None:
    message.answer = outcome.answer
    message.standalone_question = outcome.standalone_question
    message.answer_format = outcome.answer_format
    message.citations = outcome.citations
    message.abstained = outcome.abstained
    message.decline_reason = outcome.decline_reason
    message.checks = outcome.checks
    message.cached = outcome.cached
    message.route = outcome.route
    message.quoted = outcome.quoted
    message.blocked = outcome.blocked
    message.block_reason = outcome.block_reason
    message.guard_findings = outcome.findings
    message.save()

    PerformanceLog.objects.create(
        message=message,
        retrieval_ms=outcome.retrieval_ms,
        generation_ms=outcome.generation_ms,
        total_ms=outcome.total_ms,
        first_token_ms=outcome.first_token_ms,
        tokens_per_second=outcome.tokens_per_second,
    )
    if outcome.stages:
        RetrievalLog.objects.create(
            message=message,
            rag_config=config,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in outcome.chunks],
            stages={**outcome.stages, "timings_ms": outcome.stage_timings_ms},
        )
    if outcome.model:
        LLMRun.objects.create(
            message=message,
            model_name=outcome.model,
            prompt_version=config.prompt_version,
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            latency_ms=outcome.generation_ms,
        )
        TokenUsage.objects.create(
            message=message,
            model_name=outcome.model,
            input_tokens=outcome.prompt_tokens,
            output_tokens=outcome.completion_tokens,
        )
        input_cost, output_cost = compute_cost(config, outcome.prompt_tokens, outcome.completion_tokens)
        CostRecord.objects.create(
            message=message, input_cost=input_cost, output_cost=output_cost, currency=config.cost_currency
        )
    if outcome.blocked or outcome.findings:
        AuditLog.objects.create(
            user=user,
            action="guard_blocked" if outcome.blocked else "guard_masked",
            target=f"chat_message:{message.id}",
            metadata={"reason": outcome.block_reason, "findings": outcome.findings},
        )
    metrics.record_answer(outcome)
