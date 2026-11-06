"""One user question from start to finish: permissions, pipeline, and all the SQL records it leaves."""

import logging

from django.db import transaction

from apps.chatbot.models import ChatMessage, ChatSession, LLMRun
from apps.chatbot.services import generation
from apps.chatbot.services.rag_workflow import RagOutcome, run_rag
from apps.common.exceptions import ServiceUnavailable
from apps.monitoring.models import CostRecord, PerformanceLog, TokenUsage
from apps.monitoring.services.cost_tracker import compute_cost
from apps.retrieval.models import RagConfig, RetrievalLog
from apps.security.models import AuditLog
from apps.security.services.access_control import allowed_document_ids

logger = logging.getLogger(__name__)


def ask(user, question: str, config: RagConfig, session: ChatSession | None = None) -> tuple[ChatMessage, RagOutcome]:
    if session is None:
        session = ChatSession.objects.create(user=user, title=question[:120], rag_config=config)
    earlier = earlier_question(session, question)
    message = ChatMessage.objects.create(session=session, question=question)
    # Saving moves this conversation to the top of the history list.
    session.save(update_fields=["updated_at"])
    try:
        outcome = run_rag(question, config, allowed_document_ids(user), earlier)
    except ServiceUnavailable:
        # Failed requests still count: they are what the error-rate metric measures.
        PerformanceLog.objects.create(message=message, succeeded=False)
        raise
    with transaction.atomic():
        _persist(message, outcome, config, user)
    return message, outcome


def earlier_question(session: ChatSession, question: str) -> str | None:
    """The last answered question, if this one depends on it. Its complete form is used, so a chain of follow-ups
    keeps its meaning. A standalone question gets none, so earlier topics cannot pull its search off course."""
    if not generation.is_follow_up(question):
        return None
    answered = session.messages.filter(blocked=False, abstained=False).exclude(answer="")
    last = answered.order_by("-id").first()
    return (last.standalone_question or last.question) if last else None


def _persist(message: ChatMessage, outcome: RagOutcome, config: RagConfig, user) -> None:
    message.answer = outcome.answer
    message.standalone_question = outcome.standalone_question
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
