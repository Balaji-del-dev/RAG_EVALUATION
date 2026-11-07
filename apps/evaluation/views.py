from django.db.models import Count, ProtectedError
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.mixins import DestroyModelMixin
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet, ReadOnlyModelViewSet

from apps.common.constants import JobStatus
from apps.common.permissions import IsAnalystOrAdmin, IsAnalystOrReadOnly
from apps.common.tasks import enqueue
from apps.evaluation.models import EvaluationDataset, EvaluationResult, EvaluationRun, MetricDefinition
from apps.evaluation.serializers import (
    DatasetCreateSerializer,
    DatasetDetailSerializer,
    DatasetUploadSerializer,
    EvaluationDatasetSerializer,
    EvaluationErrorSerializer,
    EvaluationResultSerializer,
    EvaluationRunSerializer,
    MetricDefinitionSerializer,
)
from apps.evaluation.services import dataset_manager
from apps.evaluation.services.run_executor import sync_metric_definitions
from apps.evaluation.tasks import run_evaluation_task
from apps.security.models import AuditLog


class DatasetViewSet(DestroyModelMixin, ReadOnlyModelViewSet):
    """Datasets can be created and deleted, but not edited. A run's scores only mean something if its questions
    stay fixed, so a dataset that any run has used cannot be deleted either."""

    permission_classes = [IsAnalystOrReadOnly]

    def get_queryset(self):
        return EvaluationDataset.objects.annotate(case_count=Count("cases")).prefetch_related("cases")

    def get_serializer_class(self):
        return DatasetDetailSerializer if self.action == "retrieve" else EvaluationDatasetSerializer

    def destroy(self, request, *args, **kwargs):
        try:
            return super().destroy(request, *args, **kwargs)
        except ProtectedError:
            return Response({"error": "Runs use this dataset. Delete them first."}, status=status.HTTP_409_CONFLICT)

    def _created(self, dataset):
        dataset.case_count = dataset.cases.count()
        return Response(DatasetDetailSerializer(dataset).data, status=status.HTTP_201_CREATED)

    def create(self, request):
        serializer = DatasetCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        return self._created(dataset_manager.create_dataset(**serializer.validated_data, user=request.user))

    @action(detail=False, methods=["post"], parser_classes=[MultiPartParser])
    def upload(self, request):
        serializer = DatasetUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        upload = serializer.validated_data.pop("file")
        cases = dataset_manager.parse_dataset_file(upload.read(), upload.name)
        return self._created(
            dataset_manager.create_dataset(**serializer.validated_data, cases=cases, user=request.user)
        )


class EvaluationRunViewSet(ModelViewSet):
    serializer_class = EvaluationRunSerializer
    permission_classes = [IsAnalystOrReadOnly]
    filterset_fields = ["status", "run_type", "rag_config", "dataset"]
    http_method_names = ["get", "post", "delete", "head", "options"]
    queryset = EvaluationRun.objects.select_related("rag_config", "dataset")

    def perform_create(self, serializer):
        run = serializer.save(created_by=self.request.user)
        AuditLog.objects.create(user=self.request.user, action="evaluation_started", target=f"run:{run.id}")
        try:
            enqueue(run_evaluation_task, run.id)
        except Exception as exc:
            run.status = JobStatus.FAILED
            run.error = str(exc)
            run.save(update_fields=["status", "error", "updated_at"])
            raise

    def destroy(self, request, *args, **kwargs):
        if self.get_object().status == JobStatus.RUNNING:
            return Response({"error": "This run is still in progress"}, status=status.HTTP_409_CONFLICT)
        return super().destroy(request, *args, **kwargs)

    @action(detail=True, permission_classes=[IsAnalystOrAdmin])
    def results(self, request, pk=None):
        queryset = self.get_object().results.select_related("case").prefetch_related("scores__metric")
        page = self.paginate_queryset(queryset)
        return self.get_paginated_response(EvaluationResultSerializer(page, many=True).data)

    @action(detail=True, permission_classes=[IsAnalystOrAdmin])
    def errors(self, request, pk=None):
        return Response(EvaluationErrorSerializer(self.get_object().errors.all(), many=True).data)


class EvaluationResultDetailView(RetrieveAPIView):
    serializer_class = EvaluationResultSerializer
    permission_classes = [IsAnalystOrAdmin]
    queryset = EvaluationResult.objects.select_related("case").prefetch_related("scores__metric")


class MetricDefinitionListView(ListAPIView):
    serializer_class = MetricDefinitionSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = None

    def get_queryset(self):
        if not MetricDefinition.objects.exists():
            sync_metric_definitions()
        return MetricDefinition.objects.all()
