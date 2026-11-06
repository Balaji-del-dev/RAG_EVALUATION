from django.contrib.auth import authenticate, login, logout
from django.middleware.csrf import get_token
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

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
        user = authenticate(request, username=username, password=serializer.validated_data["password"])
        if user is None:
            AuditLog.objects.create(action="login_failed", target=username[:255])
            return Response({"error": "Incorrect username or password"}, status=status.HTTP_400_BAD_REQUEST)
        login(request, user)
        AuditLog.objects.create(user=user, action="login")
        return Response(UserSerializer(user).data)


class LogoutView(APIView):
    def post(self, request):
        AuditLog.objects.create(user=request.user, action="logout")
        logout(request)
        return Response(status=status.HTTP_204_NO_CONTENT)


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)
