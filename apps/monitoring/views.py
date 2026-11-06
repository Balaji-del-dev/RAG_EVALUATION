from rest_framework.response import Response
from rest_framework.views import APIView

from apps.monitoring.services.health import service_health
from apps.monitoring.services.latency import live_metrics
from apps.monitoring.services.token_tracker import usage_by_model


class MetricsView(APIView):
    def get(self, request):
        try:
            days = min(max(int(request.query_params.get("days", 7)), 1), 365)
        except ValueError:
            days = 7
        return Response({**live_metrics(days), "by_model": usage_by_model()})


class HealthView(APIView):
    def get(self, request):
        return Response(service_health())
