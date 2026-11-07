from django.contrib import admin
from django.http import HttpResponseForbidden
from django.shortcuts import render
from django.urls import include, path, re_path
from django.views.decorators.cache import never_cache
from django_prometheus.exports import ExportToDjangoView

LOCAL_ADDRESSES = {"127.0.0.1", "::1"}


def metrics(request):
    """For Prometheus to scrape. This machine only: the numbers describe how the system is used."""
    if request.META.get("REMOTE_ADDR") not in LOCAL_ADDRESSES:
        return HttpResponseForbidden("Metrics are only served to this machine.")
    return ExportToDjangoView(request)


@never_cache
def react_app(request):
    """The T in MVT: Django renders the built React app's index.html for every page route.
    Never cached, because its file names change with every frontend build."""
    return render(request, "index.html")


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/accounts/", include("apps.accounts.urls")),
    path("api/documents/", include("apps.documents.urls")),
    path("api/rag/", include("apps.chatbot.urls")),
    path("api/rag/", include("apps.retrieval.urls")),
    path("api/evaluation/", include("apps.evaluation.urls")),
    path("api/security/", include("apps.security.urls")),
    path("api/monitoring/", include("apps.monitoring.urls")),
    path("api/reports/", include("apps.reports.urls")),
    path("metrics", metrics),
    re_path(r"^(?!api/|admin/|static/).*$", react_app),
]
