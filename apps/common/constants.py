from django.db import models


class JobStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class Role(models.TextChoices):
    ADMIN = "admin", "Admin"
    ANALYST = "analyst", "Analyst"
    VIEWER = "viewer", "Viewer"


class MetricCategory(models.TextChoices):
    RETRIEVAL = "retrieval", "Retrieval"
    CONTEXT = "context", "Context"
    ANSWER = "answer", "Answer"
    RELIABILITY = "reliability", "Reliability"
    SECURITY = "security", "Security"
    PERFORMANCE = "performance", "Performance"
