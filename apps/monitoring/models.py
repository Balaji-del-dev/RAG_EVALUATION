"""Production-traffic telemetry, one row per chat message. Evaluation runs keep their timings on
EvaluationResult instead, so a test run never skews the live numbers."""

from django.db import models


class PerformanceLog(models.Model):
    message = models.OneToOneField("chatbot.ChatMessage", on_delete=models.CASCADE, related_name="performance")
    retrieval_ms = models.FloatField(default=0)
    generation_ms = models.FloatField(default=0)
    total_ms = models.FloatField(default=0)
    # Time until the first words of the answer were on screen.
    first_token_ms = models.FloatField(default=0)
    tokens_per_second = models.FloatField(default=0)
    succeeded = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "performance_logs"
        ordering = ["-created_at"]


class TokenUsage(models.Model):
    message = models.OneToOneField("chatbot.ChatMessage", on_delete=models.CASCADE, related_name="token_usage")
    model_name = models.CharField(max_length=255)
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "token_usage"
        ordering = ["-created_at"]


class CostRecord(models.Model):
    message = models.OneToOneField("chatbot.ChatMessage", on_delete=models.CASCADE, related_name="cost")
    input_cost = models.DecimalField(max_digits=12, decimal_places=6, default=0)
    output_cost = models.DecimalField(max_digits=12, decimal_places=6, default=0)
    currency = models.CharField(max_length=8, default="INR")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "cost_records"
        ordering = ["-created_at"]
