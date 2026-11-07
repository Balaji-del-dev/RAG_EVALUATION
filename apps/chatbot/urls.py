from django.urls import path
from rest_framework.routers import SimpleRouter

from apps.chatbot import views

router = SimpleRouter()
router.register("sessions", views.ChatSessionViewSet, basename="chat-session")

urlpatterns = [
    path("query/", views.QueryView.as_view()),
    path("query/stream/", views.QueryStreamView.as_view()),
    path("messages/<int:message_id>/feedback/", views.FeedbackView.as_view()),
    path("messages/<int:message_id>/download/<str:extension>/", views.DownloadView.as_view()),
    *router.urls,
]
