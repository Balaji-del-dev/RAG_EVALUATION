from django.urls import path

from apps.reports import views

urlpatterns = [
    path("summary/", views.SummaryView.as_view()),
    path("compare/", views.CompareView.as_view()),
    path("runs/<int:pk>/export/", views.RunExportView.as_view()),
]
