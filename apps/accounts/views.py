from django.contrib.auth import authenticate, login, logout
from django.middleware.csrf import get_token
from rest_framework import status
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView

from apps.accounts.serializers import LoginSerializer, UserSerializer
from apps.security.models import AuditLog


class CsrfView(APIView):
    """Sets the CSRF cookie, which the React app sends back on every state-changing request."""

    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"csrfToken": get_token(request)})


class LoginView(APIView):
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "login"

    def post(self, request):
        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        username = serializer.validated_data["username"]
        # The underlying request, so that django-axes can mark it as locked out for its middleware.
        user = authenticate(request._request, username=username, password=serializer.validated_data["password"])
        if user is None:
            AuditLog.objects.create(action="login_failed", target=username[:255])
            return Response({"error": "Incorrect username or password"}, status=status.HTTP_400_BAD_REQUEST)
        login(request, user)
        AuditLog.objects.create(user=user, action="login")
        return Response(UserSerializer(user).data)


class TokenView(TokenObtainPairView):
    """A JWT pair for API clients: send "Authorization: Bearer <access>". Same throttle and lockout as signing in."""

    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "login"

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        try:
            serializer.is_valid(raise_exception=True)
        except AuthenticationFailed:
            AuditLog.objects.create(action="login_failed", target=str(request.data.get("username", ""))[:255])
            raise
        AuditLog.objects.create(user=serializer.user, action="token_issued")
        return Response(serializer.validated_data)


class LogoutView(APIView):
    def post(self, request):
        AuditLog.objects.create(user=request.user, action="logout")
        logout(request)
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)
