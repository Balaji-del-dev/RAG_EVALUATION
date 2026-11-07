from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView

from apps.accounts import views

urlpatterns = [
    path("csrf/", views.CsrfView.as_view()),
    path("login/", views.LoginView.as_view()),
    path("logout/", views.LogoutView.as_view()),
    path("me/", views.MeView.as_view()),
    path("token/", views.TokenView.as_view()),
    path("token/refresh/", TokenRefreshView.as_view()),
]
