from django.conf import settings
from django.db import models


class Report(models.Model):
    """A record of each export, so there is a history of what left the system and who took it."""

    class Format(models.TextChoices):
        JSON = "json", "JSON"
        CSV = "csv", "CSV"

    run = models.ForeignKey("evaluation.EvaluationRun", on_delete=models.CASCADE, related_name="reports")
    format = models.CharField(max_length=8, choices=Format.choices)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "reports"
        ordering = ["-created_at"]
