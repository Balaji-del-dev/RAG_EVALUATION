from rest_framework.response import Response
from rest_framework.views import APIView

from apps.monitoring.services.health import service_health
from apps.monitoring.services.latency import live_metrics, slo_report
from apps.monitoring.services.token_tracker import usage_by_model


def _days(request) -> int:
    try:
        return min(max(int(request.query_params.get("days", 7)), 1), 365)
    except ValueError:
        return 7


class MetricsView(APIView):
    def get(self, request):
        metrics = live_metrics(_days(request))
        return Response({**metrics, "slo": slo_report(metrics), "by_model": usage_by_model()})


class SloView(APIView):
    """GET slo/?days=7: the live-chat objectives, their targets, and whether the period met them."""

    def get(self, request):
        metrics = live_metrics(_days(request))
        return Response({"days": metrics["days"], "requests": metrics["requests"], **slo_report(metrics)})


class HealthView(APIView):
    def get(self, request):
        return Response(service_health())
