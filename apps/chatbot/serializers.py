from rest_framework import serializers

from apps.chatbot.models import ChatMessage, ChatSession
from apps.documents.models import Document, DocumentChunk
from apps.retrieval.models import RagConfig
from apps.security.services.guardrails import text_for_user


class QueryRequestSerializer(serializers.Serializer):
    # Generous on purpose. The input guard rejects anything over its own limit, and that is logged as a block
    # and counted by the security metrics. A long attack should be a recorded block, not a validation error.
    question = serializers.CharField(max_length=20000)
    rag_config = serializers.PrimaryKeyRelatedField(queryset=RagConfig.objects.all(), required=False)
    session = serializers.IntegerField(required=False)


class ChatMessageSerializer(serializers.ModelSerializer):
    """A past message, with what the chat screen shows under it. Sources are rechecked against the reader's
    current permissions: access removed since the question was asked is removed here too."""

    findings = serializers.JSONField(source="guard_findings")
    sources = serializers.SerializerMethodField()
    stages = serializers.SerializerMethodField()
    timings_ms = serializers.SerializerMethodField()
    usage = serializers.SerializerMethodField()

    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "question",
            "standalone_question",
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
        rows = DocumentChunk.objects.filter(id__in=log.retrieved_chunk_ids, document_id__in=readable).select_related("document")
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

    def get_timings_ms(self, message):
        log = self._log(message)
        timings = dict(log.stages.get("timings_ms", {})) if log else {}
        performance = getattr(message, "performance", None)
        if performance:
            timings.update(retrieval=performance.retrieval_ms, generation=performance.generation_ms, total=performance.total_ms)
        return timings

    def get_usage(self, message):
        runs = list(message.llm_runs.all())
        if not runs:
            return {"model": "", "prompt_tokens": 0, "completion_tokens": 0}
        return {"model": runs[0].model_name, "prompt_tokens": runs[0].prompt_tokens, "completion_tokens": runs[0].completion_tokens}


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
