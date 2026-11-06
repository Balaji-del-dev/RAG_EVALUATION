from django.urls import path

from apps.monitoring import views

urlpatterns = [
    path("metrics/", views.MetricsView.as_view()),
    path("health/", views.HealthView.as_view()),
]
