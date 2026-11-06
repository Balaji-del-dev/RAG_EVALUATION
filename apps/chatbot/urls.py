from django.urls import path
from rest_framework.routers import SimpleRouter

from apps.chatbot import views

router = SimpleRouter()
router.register("sessions", views.ChatSessionViewSet, basename="chat-session")

urlpatterns = [path("query/", views.QueryView.as_view()), *router.urls]
