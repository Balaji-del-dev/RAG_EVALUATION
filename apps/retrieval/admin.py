from django.contrib import admin

from apps.retrieval.models import RagConfig, RetrievalLog


@admin.register(RagConfig)
class RagConfigAdmin(admin.ModelAdmin):
    list_display = ["name", "is_default", "retrieval_mode", "chunk_strategy", "embedding_model", "llm_model", "top_k"]


admin.site.register(RetrievalLog)
