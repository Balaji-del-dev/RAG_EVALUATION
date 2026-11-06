from rest_framework import serializers

from apps.chatbot.services.generation import PROMPTS
from apps.retrieval.models import RagConfig, RetrievalLog


class RagConfigSerializer(serializers.ModelSerializer):
    index_key = serializers.CharField(read_only=True)

    class Meta:
        model = RagConfig
        fields = "__all__"

    def validate_prompt_version(self, value):
        if value not in PROMPTS:
            raise serializers.ValidationError(f"Unknown prompt version. Available: {', '.join(sorted(PROMPTS))}")
        return value

    def validate_mmr_lambda(self, value):
        if not 0 <= value <= 1:
            raise serializers.ValidationError("Must be between 0 and 1")
        return value

    def validate(self, attrs):
        size = attrs.get("chunk_size", getattr(self.instance, "chunk_size", 220))
        overlap = attrs.get("chunk_overlap", getattr(self.instance, "chunk_overlap", 40))
        if overlap >= size:
            raise serializers.ValidationError({"chunk_overlap": "Must be smaller than chunk_size"})
        return attrs


class RetrievalLogSerializer(serializers.ModelSerializer):
    question = serializers.CharField(source="message.question", read_only=True)

    class Meta:
        model = RetrievalLog
        fields = ["id", "message", "question", "rag_config", "retrieved_chunk_ids", "stages", "created_at"]


class RetrieveRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=2000)
    rag_config = serializers.PrimaryKeyRelatedField(queryset=RagConfig.objects.all(), required=False)
