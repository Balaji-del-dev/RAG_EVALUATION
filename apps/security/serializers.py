from rest_framework import serializers

from apps.security.models import AuditLog, SecurityResult, SecurityTest


class SecurityTestSerializer(serializers.ModelSerializer):
    class Meta:
        model = SecurityTest
        fields = "__all__"


class SecurityResultSerializer(serializers.ModelSerializer):
    test_name = serializers.CharField(source="test.test_name", read_only=True)
    attack_type = serializers.CharField(source="test.attack_type", read_only=True)
    prompt = serializers.CharField(source="test.prompt", read_only=True)

    class Meta:
        model = SecurityResult
        fields = [
            "id",
            "run",
            "test",
            "test_name",
            "attack_type",
            "prompt",
            "passed",
            "severity",
            "guard_blocked",
            "attack_succeeded",
            "answer",
            "details",
            "latency_ms",
            "created_at",
        ]


class SecurityRunRequestSerializer(serializers.Serializer):
    rag_config = serializers.IntegerField()
    name = serializers.CharField(required=False, allow_blank=True, default="")


class AuditLogSerializer(serializers.ModelSerializer):
    user = serializers.CharField(source="user.username", read_only=True, default=None)

    class Meta:
        model = AuditLog
        fields = ["id", "user", "action", "target", "metadata", "timestamp"]
