from django.conf import settings
from django.db import models


class SecurityTest(models.Model):
    class AttackType(models.TextChoices):
        PROMPT_INJECTION = "prompt_injection", "Prompt injection"
        JAILBREAK = "jailbreak", "Jailbreak"
        PII = "pii", "PII extraction"
        SENSITIVE_DATA = "sensitive_data", "Sensitive data extraction"
        ACCESS_CONTROL = "access_control", "Access control"
        TOXICITY = "toxicity", "Toxic output"
        BENIGN = "benign", "Benign control"

    class Severity(models.TextChoices):
        LOW = "low", "Low"
        MEDIUM = "medium", "Medium"
        HIGH = "high", "High"
        CRITICAL = "critical", "Critical"

    test_name = models.CharField(max_length=120, unique=True)
    attack_type = models.CharField(max_length=32, choices=AttackType.choices, db_index=True)
    prompt = models.TextField()
    expected_block = models.BooleanField(default=True, help_text="Should the guards stop this request?")
    success_markers = models.JSONField(
        default=list, blank=True, help_text="If any of these appears in the final answer, the attack succeeded"
    )
    severity = models.CharField(max_length=16, choices=Severity.choices, default=Severity.HIGH)
    enabled = models.BooleanField(default=True)

    class Meta:
        db_table = "security_tests"
        ordering = ["attack_type", "test_name"]

    def __str__(self):
        return self.test_name


class SecurityResult(models.Model):
    run = models.ForeignKey("evaluation.EvaluationRun", on_delete=models.CASCADE, related_name="security_results")
    test = models.ForeignKey(SecurityTest, on_delete=models.PROTECT, related_name="results")
    passed = models.BooleanField()
    severity = models.CharField(max_length=16, choices=SecurityTest.Severity.choices)
    guard_blocked = models.BooleanField(default=False)
    attack_succeeded = models.BooleanField(default=False)
    answer = models.TextField(blank=True)
    details = models.JSONField(default=dict, blank=True)
    latency_ms = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "security_results"
        ordering = ["id"]
        constraints = [models.UniqueConstraint(fields=["run", "test"], name="uniq_run_security_test")]


class AuditLog(models.Model):
    """Security event history. Rows are never updated; the admin exposes them read-only."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    action = models.CharField(max_length=64, db_index=True)
    target = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "audit_logs"
        ordering = ["-timestamp"]
