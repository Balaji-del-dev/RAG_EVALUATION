import json
import logging
from dataclasses import asdict

from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Count
from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from rest_framework.mixins import DestroyModelMixin
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
)
from apps.chatbot.services import downloads, formats
from apps.chatbot.services.chat_service import ask, ask_streaming
from apps.common.exceptions import ServiceUnavailable
from apps.common.middleware import current_request_id
from apps.documents.services.ingestion import default_config
from apps.monitoring.services import metrics
from apps.security.services.guardrails import text_for_user

logger = logging.getLogger(__name__)


def _parse(request):
    serializer = QueryRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    data = serializer.validated_data
    # Filtering by user means another user's session id answers 404.
    session = get_object_or_404(ChatSession, pk=data["session"], user=request.user) if "session" in data else None
    config = data.get("rag_config") or (session and session.rag_config) or default_config()
    return data["question"], session, config


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
        },
        "usage": {
            "model": outcome.model,
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
        },
    }


class QueryView(APIView):
    throttle_scope = "query"

    def post(self, request):
        question, session, config = _parse(request)
        message, outcome = ask(request.user, question, config, session)
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
        question, session, config = _parse(request)
        request_id = request.request_id

        def events():
            # The body is produced after the middleware has returned, so the request ID is set again here.
            token = current_request_id.set(request_id)
            try:
                for kind, value in ask_streaming(request.user, question, config, session):
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
