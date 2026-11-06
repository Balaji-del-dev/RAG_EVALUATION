from django.contrib import admin
from django.shortcuts import render
from django.urls import include, path, re_path
from django.views.decorators.cache import never_cache


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
    re_path(r"^(?!api/|admin/|static/).*$", react_app),
]
