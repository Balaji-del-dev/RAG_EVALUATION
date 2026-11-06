from django.contrib import admin

from apps.evaluation.models import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationError,
    EvaluationResult,
    EvaluationRun,
    MetricDefinition,
    MetricScore,
)


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    list_display = ["id", "name", "run_type", "rag_config", "dataset", "status", "created_at"]
    list_filter = ["status", "run_type"]


@admin.register(MetricDefinition)
class MetricDefinitionAdmin(admin.ModelAdmin):
    list_display = ["name", "category", "scope", "formula", "higher_is_better"]
    list_filter = ["category", "scope"]


for model in (EvaluationDataset, EvaluationCase, EvaluationResult, MetricScore, EvaluationError):
    admin.site.register(model)
