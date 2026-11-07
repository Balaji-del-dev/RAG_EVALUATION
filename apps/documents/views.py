from datetime import timedelta

from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.generics import ListAPIView, RetrieveDestroyAPIView
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import IsAnalystOrAdmin, IsAnalystOrReadOnly
from apps.common.tasks import enqueue
from apps.documents.models import Document, DocumentChunk
from apps.documents.serializers import ChunkSerializer, DocumentSerializer, UploadSerializer
from apps.documents.services import ingestion
from apps.documents.services.ingestion import default_config
from apps.documents.tasks import process_document_task
from apps.security.models import AuditLog
from apps.security.services.guardrails import text_for_user

STALE_PROCESSING_AFTER = timedelta(minutes=15)


def _visible_documents(user):
    # A document the user may not read answers 404, not 403, so its existence is not revealed.
    index_key = default_config().index_key
    return Document.objects.accessible_to(user).annotate(
        chunk_count=Count("chunks", filter=Q(chunks__index_key=index_key), distinct=True)
    )


class DocumentUploadView(APIView):
    throttle_scope = "upload"
    permission_classes = [IsAnalystOrAdmin]
    parser_classes = [MultiPartParser]

    def post(self, request):
        serializer = UploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        upload = serializer.validated_data["file"]
        document = ingestion.create_document(
            data=upload.read(),
            filename=upload.name,
            content_type=upload.content_type or "",
            owner=request.user,
            visibility=serializer.validated_data["visibility"],
        )
        AuditLog.objects.create(user=request.user, action="document_uploaded", target=f"document:{document.id}")
        enqueue(process_document_task, document.id)
        document.chunk_count = 0
        return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)


class DocumentListView(ListAPIView):
    serializer_class = DocumentSerializer
    filterset_fields = ["status", "visibility"]

    def get_queryset(self):
        return _visible_documents(self.request.user)


class DocumentDetailView(RetrieveDestroyAPIView):
    serializer_class = DocumentSerializer
    permission_classes = [IsAnalystOrReadOnly]

    def get_queryset(self):
        return _visible_documents(self.request.user)

    def perform_destroy(self, instance):
        AuditLog.objects.create(user=self.request.user, action="document_deleted", target=instance.filename)
        ingestion.delete_document(instance)


class DocumentProcessView(APIView):
    permission_classes = [IsAnalystOrAdmin]

    def post(self, request, pk):
        document = get_object_or_404(Document.objects.accessible_to(request.user), pk=pk)
        # Past this age the worker probably died mid-job, so allow a retry instead of blocking forever.
        recently_started = timezone.now() - document.updated_at < STALE_PROCESSING_AFTER
        if document.status in (Document.Status.UPLOADED, Document.Status.PROCESSING) and recently_started:
            return Response({"error": "This document is already queued or processing"}, status=status.HTTP_409_CONFLICT)
        enqueue(process_document_task, document.id)
        # Show "Queued" straight away. Otherwise nothing visibly changes until the worker gets to it.
        document.status = Document.Status.UPLOADED
        document.error = ""
        document.save(update_fields=["status", "error", "updated_at"])
        AuditLog.objects.create(user=request.user, action="document_reprocessed", target=f"document:{document.id}")
        return Response({"id": document.id, "status": "queued"}, status=status.HTTP_202_ACCEPTED)


class DocumentStatusView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, pk):
        document = get_object_or_404(Document.objects.accessible_to(request.user), pk=pk)
        return Response(
            {
                "id": document.id,
                "status": document.status,
                "error": document.error,
                "page_count": document.page_count,
                "jobs": list(
                    document.embedding_jobs.values("index_key", "status", "model_name", "chunk_count", "duration_ms")[
                        :10
                    ]
                ),
            }
        )


class DocumentChunksView(ListAPIView):
    serializer_class = ChunkSerializer

    def get_queryset(self):
        document = get_object_or_404(Document.objects.accessible_to(self.request.user), pk=self.kwargs["pk"])
        return DocumentChunk.objects.filter(document=document, index_key=default_config().index_key)

    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        for chunk in response.data["results"]:
            chunk["chunk_text"] = text_for_user(request.user, chunk["chunk_text"])
        return response
