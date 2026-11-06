from django.urls import path
from rest_framework.routers import SimpleRouter

from apps.retrieval import views

router = SimpleRouter()
router.register("configs", views.RagConfigViewSet)

urlpatterns = [
    path("retrieve/", views.RetrieveView.as_view()),
    path("retrieval-logs/", views.RetrievalLogListView.as_view()),
    *router.urls,
]
