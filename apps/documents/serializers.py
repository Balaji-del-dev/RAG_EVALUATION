from rest_framework import serializers

from apps.documents.models import Document, DocumentChunk


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.IntegerField(read_only=True)
    owner = serializers.CharField(source="owner.username", read_only=True, default=None)

    class Meta:
        model = Document
        # file_path is left out: storage keys are an internal detail.
        fields = [
            "id",
            "filename",
            "content_type",
            "size_bytes",
            "status",
            "error",
            "page_count",
            "chunk_count",
            "visibility",
            "owner",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


class UploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    visibility = serializers.ChoiceField(choices=Document.Visibility.choices, default=Document.Visibility.PUBLIC)


class ChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ["id", "index_key", "chunk_index", "chunk_text", "token_count", "page_number", "metadata"]
