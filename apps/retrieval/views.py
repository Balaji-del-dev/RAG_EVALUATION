from dataclasses import asdict

from rest_framework.generics import ListAPIView
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet

from apps.common.constants import Role
from apps.common.permissions import IsAnalystOrReadOnly, has_role
from apps.documents.services.ingestion import default_config
from apps.retrieval.models import RagConfig, RetrievalLog
from apps.retrieval.serializers import RagConfigSerializer, RetrievalLogSerializer, RetrieveRequestSerializer
from apps.retrieval.services.hybrid_search import retrieve
from apps.security.services.access_control import allowed_document_ids
from apps.security.services.guardrails import text_for_user


class RagConfigViewSet(ModelViewSet):
    queryset = RagConfig.objects.all()
    serializer_class = RagConfigSerializer
    permission_classes = [IsAnalystOrReadOnly]


class RetrieveView(APIView):
    throttle_scope = "query"
    """Retrieval without the LLM: for inspecting each stage, and for load tests that target retrieval."""

    def post(self, request):
        serializer = RetrieveRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        config = serializer.validated_data.get("rag_config") or default_config()
        result = retrieve(serializer.validated_data["question"], config, allowed_document_ids(request.user))
        return Response(
            {
                "chunks": [
                    {**asdict(chunk), "text": text_for_user(request.user, chunk.text)} for chunk in result.chunks
                ],
                "stages": result.stages,
                "timings_ms": result.timings_ms,
                "below_relevance_floor": result.below_relevance_floor,
            }
        )


class RetrievalLogListView(ListAPIView):
    serializer_class = RetrievalLogSerializer
    filterset_fields = ["rag_config"]

    def get_queryset(self):
        logs = RetrievalLog.objects.select_related("message__session")
        if has_role(self.request.user, Role.ADMIN):
            return logs
        return logs.filter(message__session__user=self.request.user)
