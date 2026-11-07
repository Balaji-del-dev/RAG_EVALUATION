import copy
import json
import logging
from dataclasses import asdict

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Count
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.exceptions import ValidationError
from rest_framework.mixins import DestroyModelMixin
from rest_framework.pagination import LimitOffsetPagination
from rest_framework.renderers import BaseRenderer, JSONRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ReadOnlyModelViewSet

from apps.chatbot.models import AnswerFeedback, ChatMessage, ChatSession
from apps.chatbot.serializers import (
    ChatSessionListSerializer,
    ChatSessionSerializer,
    FeedbackSerializer,
    QueryRequestSerializer,
    ReviewDatasetSerializer,
)
from apps.chatbot.services import downloads, formats, local_models, review
from apps.chatbot.services.chat_service import ask, ask_streaming
from apps.common.exceptions import ServiceUnavailable
from apps.common.idempotency import idempotent
from apps.common.middleware import current_request_id
from apps.common.permissions import IsAnalystOrAdmin
from apps.documents.services.ingestion import default_config
from apps.monitoring.services import metrics
from apps.security.models import AuditLog
from apps.security.services.guardrails import text_for_user

logger = logging.getLogger(__name__)


def _parse(request):
    serializer = QueryRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    data = serializer.validated_data
    # Filtering by user means another user's session id answers 404.
    session = get_object_or_404(ChatSession, pk=data["session"], user=request.user) if "session" in data else None
    config = data.get("rag_config") or (session and session.rag_config) or default_config()
    if (model := data.get("llm_model")) and model != config.llm_model:
        if reason := local_models.why_not(model):
            raise ValidationError({"error": reason})
        # This question only: the configuration, and the evaluation runs that use it, keep their own model.
        config = copy.copy(config)
        config.llm_model = model
    return data["question"], session, config, data.get("fresh", False)


class LocalModelsView(APIView):
    """GET models/: the models installed in Ollama, which can chat, which fit in free memory, and the default."""

    def get(self, request):
        models, free = local_models.list_models()
        return Response(
            {"default": default_config().llm_model, "available_bytes": free, "models": [m.as_dict() for m in models]}
        )


def answer_payload(request, message, outcome, config) -> dict:
    return {
        "message_id": message.id,
        "request_id": message.request_id,
        "feedback": None,
        "session": message.session_id,
        "rag_config": config.id,
        "question": message.question,
        "answer": message.answer,
        "abstained": message.abstained,
        "decline_reason": message.decline_reason,
        "checks": message.checks,
        "cached": message.cached,
        "quoted": message.quoted,
        "standalone_question": message.standalone_question,
        "answer_format": message.answer_format,
        "format": formats.describe(message.answer_format),
        "blocked": message.blocked,
        "block_reason": message.block_reason,
        "findings": message.guard_findings,
        "citations": message.citations,
        "chunks": [{**asdict(chunk), "text": text_for_user(request.user, chunk.text)} for chunk in outcome.chunks],
        "stages": outcome.stages,
        "timings_ms": {
            **outcome.stage_timings_ms,
            "retrieval": round(outcome.retrieval_ms, 1),
            "generation": round(outcome.generation_ms, 1),
            "total": round(outcome.total_ms, 1),
            "first_token": round(outcome.first_token_ms, 1),
        },
        "usage": {
            "model": outcome.model,
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "tokens_per_second": outcome.tokens_per_second,
        },
    }


class QueryView(APIView):
    throttle_scope = "query"

    @idempotent
    def post(self, request):
        question, session, config, fresh = _parse(request)
        message, outcome = ask(request.user, question, config, session, fresh=fresh)
        return Response(answer_payload(request, message, outcome, config))


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, cls=DjangoJSONEncoder)}\n\n"


class EventStreamRenderer(BaseRenderer):
    """Lets a request that accepts only text/event-stream through content negotiation, which otherwise answers 406.
    The stream itself is not rendered; this only renders errors such as a 400 or 429, as JSON text."""

    media_type = "text/event-stream"
    format = "event-stream"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return json.dumps(data, cls=DjangoJSONEncoder).encode()


class QueryStreamView(APIView):
    """query/, as server-sent events: "text" events carry the answer sentence by sentence as it is written, then one
    "done" event carries everything query/ returns. The answer in "done" is the one to keep: the checks after
    generation can change it. A failure after streaming has begun arrives as an "error" event."""

    throttle_scope = "query"
    renderer_classes = [JSONRenderer, EventStreamRenderer]

    def post(self, request):
        question, session, config, fresh = _parse(request)
        request_id = request.request_id

        def events():
            # The body is produced after the middleware has returned, so the request ID is set again here.
            token = current_request_id.set(request_id)
            try:
                for kind, value in ask_streaming(request.user, question, config, session, fresh=fresh):
                    if kind == "text":
                        yield _event("text", {"text": value})
                    else:
                        yield _event("done", answer_payload(request, *value, config))
            except ServiceUnavailable as exc:
                yield _event("error", {"error": str(exc)})
            except Exception:
                logger.exception("Streaming answer failed")
                yield _event("error", {"error": "Internal server error"})
            finally:
                current_request_id.reset(token)

        response = StreamingHttpResponse(events(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # tells nginx not to hold the stream back
        return response


class ChatSessionViewSet(DestroyModelMixin, ReadOnlyModelViewSet):
    """A user's own conversations, most recent first. Anyone else's answers 404."""

    def get_serializer_class(self):
        return ChatSessionListSerializer if self.action == "list" else ChatSessionSerializer

    def get_queryset(self):
        sessions = ChatSession.objects.filter(user=self.request.user).order_by("-updated_at")
        if self.action == "list":
            return sessions.annotate(message_count=Count("messages"))
        return sessions.prefetch_related(
            "messages__retrieval_logs", "messages__llm_runs", "messages__performance", "messages__feedback"
        )


class DownloadView(APIView):
    """GET messages/<id>/download/<extension>/: the answer as a file, for the person who asked. ?inline=1 serves the
    HTML version for a preview. The page cannot run scripts or load anything, whatever the answer contains."""

    throttle_scope = "download"

    def get(self, request, message_id, extension):
        if extension not in downloads.DOWNLOADS:
            return Response({"error": f"Cannot make a .{extension} file."}, status=404)
        message = get_object_or_404(ChatMessage, id=message_id, session__user=request.user)
        content, content_type, name = downloads.build(message, extension)
        response = HttpResponse(content, content_type=content_type)
        inline = extension == "html" and request.query_params.get("inline")
        response["Content-Disposition"] = f'{"inline" if inline else "attachment"}; filename="{name}"'
        response["X-Content-Type-Options"] = "nosniff"
        if extension == "html":
            response["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; img-src data:"
        return response


class FeedbackView(APIView):
    """Rate an answer you asked for: POST {"rating": "up" | "down", "comment": ""} sets it, DELETE clears it.
    Anyone else's answer is a 404."""

    def post(self, request, message_id):
        message = get_object_or_404(ChatMessage, id=message_id, session__user=request.user)
        serializer = FeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        feedback, _ = AnswerFeedback.objects.update_or_create(
            message=message, user=request.user, defaults=serializer.validated_data
        )
        metrics.FEEDBACK.labels(feedback.rating).inc()
        return Response({"rating": feedback.rating, "comment": feedback.comment})

    def delete(self, request, message_id):
        message = get_object_or_404(ChatMessage, id=message_id, session__user=request.user)
        AnswerFeedback.objects.filter(message=message, user=request.user).delete()
        return Response(status=204)


class ReviewView(APIView):
    """GET review/?kind=&days=30&limit=&offset=: chat answers to look at again, newest first. kind is thumbs_down,
    flagged, declined or failed; empty means any of them. Analysts and admins only: it shows other people's
    questions, though not who asked them."""

    permission_classes = [IsAnalystOrAdmin]

    def get(self, request):
        try:
            days = min(max(int(request.query_params.get("days", 30)), 1), 365)
        except ValueError:
            days = 30
        kind = request.query_params.get("kind", "")
        paginator = LimitOffsetPagination()
        page = paginator.paginate_queryset(review.queue(days, kind), request, view=self)
        return Response(
            {
                "count": paginator.count,
                "next": paginator.get_next_link(),
                "previous": paginator.get_previous_link(),
                "days": days,
                "counts": review.counts(days),
                "labels": review.LABELS,
                "results": [review.as_item(message) for message in page],
            }
        )


class ReviewDatasetView(APIView):
    """POST review/dataset/ {"name", "items": [{"message_id", "reference_answer", "answerable"}]}: a new version of
    the named dataset, made from reviewed answers."""

    permission_classes = [IsAnalystOrAdmin]

    def post(self, request):
        serializer = ReviewDatasetSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        dataset = review.create_dataset(data["name"], data["items"], request.user)
        AuditLog.objects.create(
            user=request.user,
            action="dataset_from_review",
            target=f"dataset:{dataset.id}",
            metadata={"messages": [item["message_id"] for item in data["items"]]},
        )
        return Response(
            {"id": dataset.id, "name": dataset.name, "version": dataset.version, "case_count": len(data["items"])},
            status=201,
        )
