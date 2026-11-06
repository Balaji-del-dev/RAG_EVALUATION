from django.conf import settings
from django.contrib.auth.models import Group
from django.db import models
from django.db.models import Q

from apps.common.constants import JobStatus
from apps.common.models import TimeStampedModel


class DocumentQuerySet(models.QuerySet):
    def accessible_to(self, user):
        """Documents this user may read. Every retrieval path filters through this."""
        if user is None or not user.is_authenticated:
            return self.none()
        if user.sees_all_documents:
            return self
        return self.filter(
            Q(visibility=Document.Visibility.PUBLIC)
            | Q(owner=user)
            | Q(allowed_users=user)
            | Q(allowed_groups__in=user.groups.all())
        ).distinct()


class Document(TimeStampedModel):
    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    class Visibility(models.TextChoices):
        PUBLIC = "public", "Everyone signed in"
        RESTRICTED = "restricted", "Only allowed users and groups"

    filename = models.CharField(max_length=255)
    file_path = models.CharField(max_length=512, help_text="Object key in MinIO")
    content_type = models.CharField(max_length=128, blank=True)
    size_bytes = models.BigIntegerField(default=0)
    sha256 = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.UPLOADED, db_index=True)
    error = models.TextField(blank=True)
    page_count = models.IntegerField(default=0)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="documents"
    )
    visibility = models.CharField(max_length=16, choices=Visibility.choices, default=Visibility.PUBLIC)
    allowed_users = models.ManyToManyField(settings.AUTH_USER_MODEL, blank=True, related_name="shared_documents")
    allowed_groups = models.ManyToManyField(Group, blank=True, related_name="shared_documents")

    objects = DocumentQuerySet.as_manager()

    class Meta:
        db_table = "documents"
        ordering = ["-created_at"]

    def __str__(self):
        return self.filename


class DocumentPage(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="pages")
    page_number = models.IntegerField()
    content = models.TextField()

    class Meta:
        db_table = "document_pages"
        ordering = ["document_id", "page_number"]
        constraints = [models.UniqueConstraint(fields=["document", "page_number"], name="uniq_document_page")]


class DocumentChunk(models.Model):
    """One chunk of one document under one index.

    `index_key` identifies a chunking and embedding combination (see RagConfig.index_key), so
    configurations that chunk differently can be compared without overwriting each other. The
    vectors live in Qdrant and the BM25 postings in OpenSearch, both keyed by this row's id.
    """

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    index_key = models.CharField(max_length=16, db_index=True)
    chunk_index = models.IntegerField()
    chunk_text = models.TextField()
    token_count = models.IntegerField()
    page_number = models.IntegerField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "document_chunks"
        ordering = ["document_id", "chunk_index"]
        constraints = [
            models.UniqueConstraint(fields=["document", "index_key", "chunk_index"], name="uniq_document_chunk")
        ]


class EmbeddingJob(TimeStampedModel):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="embedding_jobs")
    index_key = models.CharField(max_length=16)
    status = models.CharField(max_length=16, choices=JobStatus.choices, default=JobStatus.PENDING, db_index=True)
    model_name = models.CharField(max_length=255)
    chunk_count = models.IntegerField(default=0)
    duration_ms = models.FloatField(default=0)
    error = models.TextField(blank=True)

    class Meta:
        db_table = "embedding_jobs"
        ordering = ["-created_at"]
