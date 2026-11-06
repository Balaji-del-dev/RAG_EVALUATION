from django.urls import path
from rest_framework.routers import SimpleRouter

from apps.evaluation import views

router = SimpleRouter()
router.register("datasets", views.DatasetViewSet, basename="evaluation-dataset")
router.register("runs", views.EvaluationRunViewSet, basename="evaluation-run")

urlpatterns = [
    path("results/<int:pk>/", views.EvaluationResultDetailView.as_view()),
    path("metrics/", views.MetricDefinitionListView.as_view()),
    *router.urls,
]
