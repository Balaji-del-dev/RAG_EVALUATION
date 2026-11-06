from django.urls import path

from apps.documents import views

urlpatterns = [
    path("upload/", views.DocumentUploadView.as_view()),
    path("", views.DocumentListView.as_view()),
    path("<int:pk>/", views.DocumentDetailView.as_view()),
    path("<int:pk>/process/", views.DocumentProcessView.as_view()),
    path("<int:pk>/status/", views.DocumentStatusView.as_view()),
    path("<int:pk>/chunks/", views.DocumentChunksView.as_view()),
]
