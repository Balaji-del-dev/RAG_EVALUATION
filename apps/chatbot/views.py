from dataclasses import asdict

from django.db.models import Count
from django.shortcuts import get_object_or_404
from rest_framework.mixins import DestroyModelMixin
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ReadOnlyModelViewSet

from apps.chatbot.models import ChatSession
from apps.chatbot.serializers import ChatSessionListSerializer, ChatSessionSerializer, QueryRequestSerializer
from apps.chatbot.services.chat_service import ask
from apps.documents.services.ingestion import default_config
from apps.security.services.guardrails import text_for_user


class QueryView(APIView):
    def post(self, request):
        serializer = QueryRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        # Filtering by user means another user's session id answers 404.
        session = get_object_or_404(ChatSession, pk=data["session"], user=request.user) if "session" in data else None
        config = data.get("rag_config") or (session and session.rag_config) or default_config()
        message, outcome = ask(request.user, data["question"], config, session)
        return Response(
            {
                "message_id": message.id,
                "session": message.session_id,
                "rag_config": config.id,
                "question": message.question,
                "answer": message.answer,
                "abstained": message.abstained,
                "decline_reason": message.decline_reason,
                "quoted": message.quoted,
                "standalone_question": message.standalone_question,
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
        )


class ChatSessionViewSet(DestroyModelMixin, ReadOnlyModelViewSet):
    """A user's own conversations, most recent first. Anyone else's answers 404."""

    def get_serializer_class(self):
        return ChatSessionListSerializer if self.action == "list" else ChatSessionSerializer

    def get_queryset(self):
        sessions = ChatSession.objects.filter(user=self.request.user).order_by("-updated_at")
        if self.action == "list":
            return sessions.annotate(message_count=Count("messages"))
        return sessions.prefetch_related("messages__retrieval_logs", "messages__llm_runs", "messages__performance")
