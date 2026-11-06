from django.contrib import admin

from apps.documents.models import Document, DocumentChunk, DocumentPage, EmbeddingJob


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["filename", "status", "visibility", "owner", "page_count", "created_at"]
    list_filter = ["status", "visibility"]
    search_fields = ["filename"]
    filter_horizontal = ["allowed_users", "allowed_groups"]


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ["document", "index_key", "chunk_index", "token_count", "page_number"]
    list_filter = ["index_key"]
    search_fields = ["chunk_text"]


admin.site.register(DocumentPage)
admin.site.register(EmbeddingJob)
