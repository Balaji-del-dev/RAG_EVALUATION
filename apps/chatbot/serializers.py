from rest_framework import serializers

from apps.chatbot.models import ChatMessage, ChatSession
from apps.chatbot.services import formats
from apps.documents.models import Document, DocumentChunk
from apps.retrieval.models import RagConfig
from apps.security.services.guardrails import text_for_user


class QueryRequestSerializer(serializers.Serializer):
    # Generous on purpose. The input guard rejects anything over its own limit, and that is logged as a block
    # and counted by the security metrics. A long attack should be a recorded block, not a validation error.
    question = serializers.CharField(max_length=20000)
    rag_config = serializers.PrimaryKeyRelatedField(queryset=RagConfig.objects.all(), required=False)
    session = serializers.IntegerField(required=False)
    # An installed Ollama chat model for this question only, such as "ollama_chat/qwen2.5:3b". Empty: the config's.
    llm_model = serializers.CharField(required=False, allow_blank=True, max_length=255)
    # Skip the answer cache: "Answer again" on a cached answer.
    fresh = serializers.BooleanField(required=False, default=False)


class ChatMessageSerializer(serializers.ModelSerializer):
    """A past message, with what the chat screen shows under it. Sources are rechecked against the reader's
    current permissions: access removed since the question was asked is removed here too."""

    findings = serializers.JSONField(source="guard_findings")
    sources = serializers.SerializerMethodField()
    stages = serializers.SerializerMethodField()
    timings_ms = serializers.SerializerMethodField()
    usage = serializers.SerializerMethodField()
    feedback = serializers.SerializerMethodField()
    format = serializers.SerializerMethodField()

    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "question",
            "standalone_question",
            "answer_format",
            "format",
            "answer",
            "citations",
            "abstained",
            "decline_reason",
            "quoted",
            "blocked",
            "block_reason",
            "findings",
            "created_at",
            "sources",
            "stages",
            "timings_ms",
            "usage",
            "request_id",
            "feedback",
            "checks",
            "cached",
        ]

    @staticmethod
    def _log(message):
        logs = list(message.retrieval_logs.all())
        return logs[0] if logs else None

    def get_stages(self, message):
        log = self._log(message)
        return {name: hits for name, hits in log.stages.items() if name != "timings_ms"} if log else {}

    def get_sources(self, message):
        log = self._log(message)
        if not log or not log.retrieved_chunk_ids:
            return []
        user = self.context["request"].user
        readable = Document.objects.accessible_to(user).values("id")
        rows = DocumentChunk.objects.filter(id__in=log.retrieved_chunk_ids, document_id__in=readable).select_related(
            "document"
        )
        by_id = {row.id: row for row in rows}
        scores = {}
        for stage in ("bm25", "vector", "rrf", "rerank", "mmr"):
            scores.update({hit["chunk_id"]: hit["score"] for hit in log.stages.get(stage, [])})
        return [
            {
                "chunk_id": chunk_id,
                "document_id": by_id[chunk_id].document_id,
                "filename": by_id[chunk_id].document.filename,
                "page_number": by_id[chunk_id].page_number,
                "text": text_for_user(user, by_id[chunk_id].chunk_text),
                "score": scores.get(chunk_id, 0.0),
            }
            for chunk_id in log.retrieved_chunk_ids
            if chunk_id in by_id
        ]

    def get_format(self, message):
        return formats.describe(message.answer_format)

    def get_feedback(self, message):
        user = self.context["request"].user
        return next((item.rating for item in message.feedback.all() if item.user_id == user.id), None)

    def get_timings_ms(self, message):
        log = self._log(message)
        timings = dict(log.stages.get("timings_ms", {})) if log else {}
        performance = getattr(message, "performance", None)
        if performance:
            timings.update(
                retrieval=performance.retrieval_ms,
                generation=performance.generation_ms,
                total=performance.total_ms,
                first_token=performance.first_token_ms,
            )
        return timings

    def get_usage(self, message):
        runs = list(message.llm_runs.all())
        if not runs:
            return {"model": "", "prompt_tokens": 0, "completion_tokens": 0}
        return {
            "model": runs[0].model_name,
            "prompt_tokens": runs[0].prompt_tokens,
            "completion_tokens": runs[0].completion_tokens,
            "tokens_per_second": getattr(getattr(message, "performance", None), "tokens_per_second", 0.0),
        }


class FeedbackSerializer(serializers.Serializer):
    rating = serializers.ChoiceField(choices=["up", "down"])
    comment = serializers.CharField(required=False, allow_blank=True, max_length=2000, default="")


class ChatSessionListSerializer(serializers.ModelSerializer):
    """For the history list, which only needs titles and dates."""

    message_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = ChatSession
        fields = ["id", "title", "created_at", "updated_at", "message_count"]


class ChatSessionSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)

    class Meta:
        model = ChatSession
        fields = ["id", "title", "rag_config", "created_at", "updated_at", "messages"]


class ReviewCaseSerializer(serializers.Serializer):
    message_id = serializers.IntegerField()
    reference_answer = serializers.CharField(required=False, allow_blank=True, max_length=20000, default="")
    # False: the right behaviour is to decline, because the documents do not answer it.
    answerable = serializers.BooleanField(required=False, default=True)


class ReviewDatasetSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    items = ReviewCaseSerializer(many=True, allow_empty=False, max_length=500)
