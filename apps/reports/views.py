from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import IsAnalystOrAdmin
from apps.evaluation.models import EvaluationRun
from apps.reports.models import Report
from apps.reports.services import comparison, report_generator


class SummaryView(APIView):
    def get(self, request):
        return Response(report_generator.dashboard_summary())


class CompareView(APIView):
    def get(self, request):
        raw = request.query_params.get("runs", "")
        try:
            run_ids = [int(part) for part in raw.split(",") if part.strip()]
        except ValueError:
            return Response({"error": "runs must be comma-separated ids"}, status=status.HTTP_400_BAD_REQUEST)
        if len(run_ids) < 2:
            return Response({"error": "Choose at least two runs to compare"}, status=status.HTTP_400_BAD_REQUEST)
        return Response(comparison.compare_runs(run_ids))


class RunExportView(APIView):
    # Exports include answers and contexts drawn from every document, so viewers cannot download them.
    permission_classes = [IsAnalystOrAdmin]

    def get(self, request, pk):
        run = get_object_or_404(EvaluationRun, pk=pk)
        # "type", not "format": DRF reserves the ?format= parameter for choosing a renderer.
        export_type = request.query_params.get("type", "json")
        if export_type == "csv":
            report_generator.record_export(run, Report.Format.CSV, request.user)
            response = HttpResponse(report_generator.run_report_csv(run), content_type="text/csv")
            response["Content-Disposition"] = f'attachment; filename="run-{run.id}.csv"'
            return response
        if export_type == "json":
            report_generator.record_export(run, Report.Format.JSON, request.user)
            return Response(report_generator.run_report(run))
        return Response({"error": "type must be json or csv"}, status=status.HTTP_400_BAD_REQUEST)
