"""Ingestion pipeline: file -> pages -> chunks -> embeddings -> Qdrant + BM25 index.

Chunk rows are the source of truth; both search indexes can be rebuilt from them. A document can have
several chunk sets, one per index_key, so that differently chunked configurations can sit side by side.
"""

import hashlib
import logging
import time
import uuid
from pathlib import PurePath

from django.conf import settings
from django.db import transaction

from apps.common.constants import JobStatus
from apps.common.exceptions import InvalidDocument
from apps.common.utils import count_tokens
from apps.documents.models import Document, DocumentChunk, DocumentPage, EmbeddingJob
from apps.documents.services import embeddings, storage
from apps.documents.services.chunking import chunk_pages
from apps.documents.services.parser import ParsedPage, parse_document
from apps.retrieval.models import RagConfig
from apps.retrieval.services import bm25_search, vector_search
from apps.security.models import AuditLog
from apps.security.services.guardrails import mask_secrets

logger = logging.getLogger(__name__)


def create_document(*, data: bytes, filename: str, content_type: str, owner, visibility: str) -> Document:
    """Validate an upload and store the original file. Parsing happens later, in a background task."""
    extension = PurePath(filename).suffix.lower()
    if extension not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_UPLOAD_EXTENSIONS))
        raise InvalidDocument(f"Unsupported file type {extension or '(none)'}. Allowed: {allowed}")
    if not data:
        raise InvalidDocument("The file is empty")
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise InvalidDocument(f"The file is larger than {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    digest = hashlib.sha256(data).hexdigest()
    existing = Document.objects.filter(sha256=digest).first()
    if existing:
        raise InvalidDocument(f"This file has already been uploaded as '{existing.filename}'")

    key = f"{uuid.uuid4().hex}{extension}"
    storage.put_object(key, data, content_type or "application/octet-stream")
    return Document.objects.create(
        filename=PurePath(filename).name,
        file_path=key,
        content_type=content_type,
        size_bytes=len(data),
        sha256=digest,
        owner=owner,
        visibility=visibility,
    )


def default_config() -> RagConfig:
    config = RagConfig.objects.filter(is_default=True).first() or RagConfig.objects.order_by("id").first()
    if config is None:
        config = RagConfig.objects.create(name="Default", is_default=True)
    return config


def _without_secrets(document: Document, pages: list[ParsedPage]) -> list[ParsedPage]:
    """Credentials never belong in an index, whoever may read the document. Masking them here means that no
    search, prompt or screen can reveal them. The original file in storage is unchanged."""
    cleaned, found = [], set()
    for page in pages:
        text, types = mask_secrets(page.content)
        cleaned.append(ParsedPage(page.page_number, text))
        found.update(types)
    if found:
        AuditLog.objects.create(
            action="secrets_masked", target=f"document:{document.id}", metadata={"types": sorted(found)}
        )
    return cleaned


def _configs_to_index(document: Document, primary: RagConfig) -> list[RagConfig]:
    """The primary config, plus every other index holding this document, so that none is left with old text."""
    existing = set(document.chunks.values_list("index_key", flat=True)) - {primary.index_key}
    targets = {primary.index_key: primary}
    for config in RagConfig.objects.all():
        if config.index_key in existing:
            targets.setdefault(config.index_key, config)
    return list(targets.values())


def process_document(document_id: int, config: RagConfig | None = None) -> Document:
    """Parse the file and build the index for the default configuration."""
    document = Document.objects.get(pk=document_id)
    document.status = Document.Status.PROCESSING
    document.error = ""
    document.save(update_fields=["status", "error", "updated_at"])
    try:
        pages = _without_secrets(document, parse_document(storage.get_object(document.file_path), document.filename))
        with transaction.atomic():
            document.pages.all().delete()
            DocumentPage.objects.bulk_create(
                DocumentPage(document=document, page_number=p.page_number, content=p.content) for p in pages
            )
            document.page_count = len(pages)
            document.save(update_fields=["page_count", "updated_at"])
        for target in _configs_to_index(document, config or default_config()):
            index_document(document, target)
    except Exception as exc:
        logger.exception("Processing failed for document %s", document_id)
        document.status = Document.Status.FAILED
        document.error = str(exc)[:2000]
        document.save(update_fields=["status", "error", "updated_at"])
        raise
    document.status = Document.Status.READY
    document.save(update_fields=["status", "updated_at"])
    return document


def index_document(document: Document, config: RagConfig) -> int:
    """Build this document's chunks and index entries for one configuration. Safe to re-run."""
    started = time.perf_counter()
    job = EmbeddingJob.objects.create(
        document=document, index_key=config.index_key, model_name=config.embedding_model, status=JobStatus.RUNNING
    )
    try:
        pages = [ParsedPage(p.page_number, p.content) for p in document.pages.all()]
        embed_fn = lambda texts: embeddings.embed_texts(texts, config.embedding_model)  # noqa: E731
        drafts = chunk_pages(pages, config, embed_fn=embed_fn)

        with transaction.atomic():
            _delete_index_entries(document, config)
            chunks = DocumentChunk.objects.bulk_create(
                DocumentChunk(
                    document=document,
                    index_key=config.index_key,
                    chunk_index=index,
                    chunk_text=draft.text,
                    token_count=count_tokens(draft.text),
                    page_number=draft.page_number,
                    metadata=draft.metadata,
                )
                for index, draft in enumerate(drafts)
            )

        vectors = embeddings.embed_texts([c.chunk_text for c in chunks], config.embedding_model)
        vector_search.upsert(config, chunks, vectors)
        bm25_search.backend().index(config, chunks)
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)[:2000]
        job.duration_ms = (time.perf_counter() - started) * 1000
        job.save()
        raise
    job.status = JobStatus.COMPLETED
    job.chunk_count = len(chunks)
    job.duration_ms = (time.perf_counter() - started) * 1000
    job.save()
    return len(chunks)


def _delete_index_entries(document: Document, config: RagConfig) -> None:
    vector_search.delete_document(config, document.id)
    bm25_search.backend().delete_document(config, document.id)
    document.chunks.filter(index_key=config.index_key).delete()


def ensure_index(config: RagConfig) -> int:
    """Index every ready document that has no chunks under this configuration. Returns how many."""
    missing = Document.objects.filter(status=Document.Status.READY).exclude(chunks__index_key=config.index_key)
    count = 0
    for document in missing:
        index_document(document, config)
        count += 1
    return count


def delete_document(document: Document) -> None:
    """Remove a document from every index it is in, then from storage and SQL."""
    index_keys = set(document.chunks.values_list("index_key", flat=True))
    for index_key in index_keys:
        config = next((c for c in RagConfig.objects.all() if c.index_key == index_key), None)
        if config:
            vector_search.delete_document(config, document.id)
            bm25_search.backend().delete_document(config, document.id)
    storage.delete_object(document.file_path)
    document.delete()
