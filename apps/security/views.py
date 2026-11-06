from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from apps.common.permissions import IsAdminRole, IsAnalystOrAdmin, IsAnalystOrReadOnly
from apps.common.tasks import enqueue
from apps.evaluation.models import EvaluationRun
from apps.evaluation.serializers import EvaluationRunSerializer
from apps.evaluation.tasks import run_evaluation_task
from apps.retrieval.models import RagConfig
from apps.security.models import AuditLog, SecurityResult, SecurityTest
from apps.security.serializers import (
    AuditLogSerializer,
    SecurityResultSerializer,
    SecurityRunRequestSerializer,
    SecurityTestSerializer,
)


class SecurityTestViewSet(ModelViewSet):
    queryset = SecurityTest.objects.all()
    serializer_class = SecurityTestSerializer
    permission_classes = [IsAnalystOrReadOnly]
    filterset_fields = ["attack_type", "enabled"]


class SecurityResultListView(ListAPIView):
    serializer_class = SecurityResultSerializer
    permission_classes = [IsAnalystOrAdmin]
    filterset_fields = ["run", "passed", "severity", "test__attack_type"]
    queryset = SecurityResult.objects.select_related("test")


class SecurityRunView(APIView):
    permission_classes = [IsAnalystOrAdmin]

    def post(self, request):
        serializer = SecurityRunRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        config = get_object_or_404(RagConfig, pk=serializer.validated_data["rag_config"])
        run = EvaluationRun.objects.create(
            name=serializer.validated_data["name"],
            run_type=EvaluationRun.RunType.SECURITY,
            rag_config=config,
            created_by=request.user,
        )
        AuditLog.objects.create(user=request.user, action="security_run_started", target=f"run:{run.id}")
        enqueue(run_evaluation_task, run.id)
        return Response(EvaluationRunSerializer(run).data, status=status.HTTP_201_CREATED)


class AuditLogListView(ListAPIView):
    serializer_class = AuditLogSerializer
    permission_classes = [IsAdminRole]
    filterset_fields = ["action"]
    queryset = AuditLog.objects.select_related("user")
