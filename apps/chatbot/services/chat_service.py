"""One user question from start to finish: permissions, pipeline, and all the SQL records it leaves."""

import logging
import re

from django.db import transaction

from apps.chatbot.models import ChatMessage, ChatSession, LLMRun
from apps.chatbot.services import generation
from apps.chatbot.services.rag_workflow import RagOutcome, run_rag, stream_rag
from apps.common.exceptions import ServiceUnavailable
from apps.common.middleware import current_request_id
from apps.monitoring.models import CostRecord, PerformanceLog, TokenUsage
from apps.monitoring.services import metrics
from apps.monitoring.services.cost_tracker import compute_cost
from apps.retrieval.models import RagConfig, RetrievalLog
from apps.security.models import AuditLog
from apps.security.services import guardrails
from apps.security.services.access_control import allowed_document_ids

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
    }
    return message, context


def _complete_sentences(text: str) -> tuple[str, str]:
    """Split off the finished sentences, keeping the unfinished end for later."""
    ends = [match.end() for match in SENTENCE_END_RE.finditer(text)]
    return (text[: ends[-1]], text[ends[-1] :]) if ends else ("", text)


def ask_streaming(user, question: str, config: RagConfig, session: ChatSession | None = None):
    """ask(), yielding ("text", sentence) while the answer is written and then ("done", (message, outcome)).

    Each sentence passes the output guard before it is sent, because masking only the final answer would show PII
    on screen first. After a sentence the guard blocks, nothing more is sent, and the final answer replaces it all."""
    message, context = _begin(user, question, config, session)
    pending, stopped, outcome = "", False, None
    try:
        for kind, value in stream_rag(question, config, allowed_document_ids(user), **context):
            if kind == "outcome":
                outcome = value
                continue
            ready, pending = _complete_sentences(pending + value)
            if ready and not stopped:
                verdict = guardrails.check_output(ready)
                stopped = verdict.blocked
                if not stopped:
                    yield "text", verdict.text
    except ServiceUnavailable:
        PerformanceLog.objects.create(message=message, succeeded=False)
        raise
    with transaction.atomic():
        _persist(message, outcome, config, user)
    yield "done", (message, outcome)


def ask(user, question: str, config: RagConfig, session: ChatSession | None = None) -> tuple[ChatMessage, RagOutcome]:
    message, context = _begin(user, question, config, session)
    try:
        outcome = run_rag(question, config, allowed_document_ids(user), **context)
    except ServiceUnavailable:
        # Failed requests still count: they are what the error-rate metric measures.
        PerformanceLog.objects.create(message=message, succeeded=False)
        raise
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
