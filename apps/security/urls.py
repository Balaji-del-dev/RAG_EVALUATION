from django.urls import path
from rest_framework.routers import SimpleRouter

from apps.security import views

router = SimpleRouter()
router.register("tests", views.SecurityTestViewSet)

urlpatterns = [
    path("results/", views.SecurityResultListView.as_view()),
    path("runs/", views.SecurityRunView.as_view()),
    path("audit/", views.AuditLogListView.as_view()),
    *router.urls,
]
