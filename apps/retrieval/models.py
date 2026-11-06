import hashlib

from django.db import models
from django.db.models import Q

from apps.common.models import TimeStampedModel


class RagConfig(TimeStampedModel):
    """Everything that defines one RAG experiment. Comparing runs means comparing these rows."""

    class Mode(models.TextChoices):
        VECTOR = "vector", "Vector only"
        BM25 = "bm25", "BM25 only"
        HYBRID = "hybrid", "Hybrid (RRF)"

    class ChunkStrategy(models.TextChoices):
        RECURSIVE = "recursive", "Recursive"
        STRUCTURE = "structure", "Structure-based"
        SEMANTIC = "semantic", "Semantic"

    name = models.CharField(max_length=120, unique=True)
    description = models.TextField(blank=True)
    is_default = models.BooleanField(default=False)

    # Index: changing any of these needs its own chunks and vectors.
    embedding_model = models.CharField(max_length=255, default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_strategy = models.CharField(max_length=16, choices=ChunkStrategy.choices, default=ChunkStrategy.RECURSIVE)
    chunk_size = models.IntegerField(default=220, help_text="Approximate tokens")
    chunk_overlap = models.IntegerField(default=40)

    # Retrieval
    retrieval_mode = models.CharField(max_length=16, choices=Mode.choices, default=Mode.HYBRID)
    candidate_k = models.IntegerField(default=20, help_text="Candidates fetched from each retriever")
    rrf_k = models.IntegerField(default=60)
    reranker = models.CharField(
        max_length=255, blank=True, default="cross-encoder/ms-marco-MiniLM-L6-v2", help_text="Blank disables reranking"
    )
    rerank_top_n = models.IntegerField(default=10)
    # Off by default. On the sample data, "explain leave policy" (answerable) scored -5.9 and a question
    # about revenue (not in the documents) scored -4.6, so no cutoff separates the two. Let the LLM decide.
    rerank_min_score = models.FloatField(
        null=True, blank=True, default=None, help_text="Optional: below this, the top chunk counts as no evidence"
    )
    use_mmr = models.BooleanField(default=True)
    mmr_lambda = models.FloatField(default=0.7)
    top_k = models.IntegerField(default=5)

    # Generation
    llm_model = models.CharField(max_length=255, default="ollama_chat/qwen2.5:1.5b")
    temperature = models.FloatField(default=0.0)
    max_tokens = models.IntegerField(default=256)
    prompt_version = models.CharField(max_length=16, default="v3")
    input_cost_per_1k = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    output_cost_per_1k = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    cost_currency = models.CharField(max_length=8, default="INR")

    class Meta:
        db_table = "rag_configs"
        ordering = ["name"]
        constraints = [
            models.UniqueConstraint(fields=["is_default"], condition=Q(is_default=True), name="one_default_rag_config"),
            models.CheckConstraint(condition=Q(chunk_overlap__lt=models.F("chunk_size")), name="overlap_below_size"),
        ]

    def __str__(self):
        return self.name

    @property
    def index_key(self) -> str:
        spec = f"{self.chunk_strategy}|{self.chunk_size}|{self.chunk_overlap}|{self.embedding_model}"
        return hashlib.sha1(spec.encode(), usedforsecurity=False).hexdigest()[:12]

    @property
    def index_name(self) -> str:
        return f"rageval_{self.index_key}"


class RetrievalLog(models.Model):
    message = models.ForeignKey("chatbot.ChatMessage", on_delete=models.CASCADE, related_name="retrieval_logs")
    rag_config = models.ForeignKey(RagConfig, null=True, on_delete=models.SET_NULL)
    retrieved_chunk_ids = models.JSONField(default=list)
    stages = models.JSONField(default=dict, help_text="Chunk ids and scores after each retrieval stage")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "retrieval_logs"
        ordering = ["-created_at"]
