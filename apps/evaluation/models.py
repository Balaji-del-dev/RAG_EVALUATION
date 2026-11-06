from django.conf import settings
from django.db import models
from django.db.models import Q

from apps.common.constants import JobStatus, MetricCategory
from apps.common.models import TimeStampedModel


class EvaluationDataset(TimeStampedModel):
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=32, default="1")
    description = models.TextField(blank=True)
    documents = models.ManyToManyField(
        "documents.Document",
        blank=True,
        related_name="datasets",
        help_text="The documents these questions were written against. Empty means every ready document.",
    )
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)

    class Meta:
        db_table = "evaluation_datasets"
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["name", "version"], name="uniq_dataset_version")]

    def __str__(self):
        return f"{self.name} v{self.version}"


class EvaluationCase(TimeStampedModel):
    dataset = models.ForeignKey(EvaluationDataset, on_delete=models.CASCADE, related_name="cases")
    question = models.TextField()
    reference_answer = models.TextField(blank=True)
    answerable = models.BooleanField(default=True, help_text="False if the correct behaviour is to abstain")
    evidence = models.JSONField(
        default=list,
        blank=True,
        help_text="Text passages that answer the question. Chunk relevance is derived from these, "
        "so the gold labels survive re-chunking.",
    )

    class Meta:
        db_table = "evaluation_cases"
        ordering = ["id"]


class EvaluationRun(TimeStampedModel):
    class RunType(models.TextChoices):
        FULL = "full", "Quality and security"
        QUALITY = "quality", "Quality only"
        SECURITY = "security", "Security only"

    name = models.CharField(max_length=255, blank=True)
    run_type = models.CharField(max_length=16, choices=RunType.choices, default=RunType.QUALITY)
    dataset = models.ForeignKey(EvaluationDataset, null=True, blank=True, on_delete=models.PROTECT, related_name="runs")
    rag_config = models.ForeignKey("retrieval.RagConfig", on_delete=models.PROTECT, related_name="runs")
    status = models.CharField(max_length=16, choices=JobStatus.choices, default=JobStatus.PENDING, db_index=True)
    k = models.IntegerField(default=5)
    frameworks = models.JSONField(default=list, blank=True, help_text='Optional adapters: "ragas", "deepeval"')
    selected_metrics = models.JSONField(default=list, blank=True, help_text="Empty means all metrics")
    snapshot = models.JSONField(default=dict, blank=True, help_text="Everything needed to reproduce this run")
    summary = models.JSONField(default=dict, blank=True)
    total_cases = models.IntegerField(default=0)
    completed_cases = models.IntegerField(default=0)
    failed_cases = models.IntegerField(default=0)
    progress_stage = models.CharField(max_length=64, blank=True)
    worker = models.CharField(max_length=255, blank=True, help_text="The Celery worker executing this run")
    error = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)

    class Meta:
        db_table = "evaluation_runs"
        ordering = ["-created_at"]
        constraints = [
            models.CheckConstraint(
                condition=Q(run_type="security") | Q(dataset__isnull=False), name="quality_runs_need_dataset"
            )
        ]

    def __str__(self):
        return self.name or f"Run {self.pk}"


class EvaluationResult(TimeStampedModel):
    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="results")
    case = models.ForeignKey(EvaluationCase, on_delete=models.PROTECT, related_name="results")
    answer = models.TextField(blank=True)
    contexts = models.JSONField(default=list, blank=True)
    retrieved_chunk_ids = models.JSONField(default=list, blank=True)
    relevant_chunk_ids = models.JSONField(default=list, blank=True)
    citations = models.JSONField(default=list, blank=True)
    abstained = models.BooleanField(default=False)
    blocked = models.BooleanField(default=False)
    retrieval_ms = models.FloatField(default=0)
    generation_ms = models.FloatField(default=0)
    total_ms = models.FloatField(default=0)
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    cost = models.DecimalField(max_digits=12, decimal_places=6, default=0)

    class Meta:
        db_table = "evaluation_results"
        ordering = ["id"]
        constraints = [models.UniqueConstraint(fields=["run", "case"], name="uniq_run_case")]


class MetricDefinition(models.Model):
    class Scope(models.TextChoices):
        CASE = "case", "Scored per question"
        RUN = "run", "Scored per run"

    name = models.CharField(max_length=64, unique=True)
    display_name = models.CharField(max_length=64)
    order = models.IntegerField(default=0, help_text="Position in the learning roadmap, 1 to 37")
    priority = models.CharField(max_length=32, blank=True)
    question = models.TextField(blank=True, help_text="What this metric tells you, in plain language")
    category = models.CharField(max_length=16, choices=MetricCategory.choices, db_index=True)
    formula = models.TextField()
    description = models.TextField(blank=True)
    scope = models.CharField(max_length=8, choices=Scope.choices, default=Scope.CASE)
    higher_is_better = models.BooleanField(default=True)
    unit = models.CharField(max_length=16, blank=True)

    class Meta:
        db_table = "metric_definitions"
        ordering = ["order"]

    def __str__(self):
        return self.name


class MetricScore(models.Model):
    """One score. `result` is null for run-level metrics such as p95 latency."""

    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="scores")
    result = models.ForeignKey(EvaluationResult, null=True, blank=True, on_delete=models.CASCADE, related_name="scores")
    metric = models.ForeignKey(MetricDefinition, on_delete=models.PROTECT, related_name="scores")
    framework = models.CharField(max_length=16, default="builtin")
    score = models.FloatField()
    details = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "metric_scores"
        indexes = [models.Index(fields=["run", "metric"])]
        constraints = [
            models.UniqueConstraint(
                fields=["result", "metric", "framework"],
                condition=Q(result__isnull=False),
                name="uniq_case_metric_score",
            ),
            models.UniqueConstraint(
                fields=["run", "metric", "framework"],
                condition=Q(result__isnull=True),
                name="uniq_run_metric_score",
            ),
        ]


class EvaluationError(models.Model):
    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="errors")
    case = models.ForeignKey(EvaluationCase, null=True, blank=True, on_delete=models.SET_NULL)
    stage = models.CharField(max_length=64, blank=True)
    error_message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "evaluation_errors"
        ordering = ["id"]
