from rest_framework import serializers

from apps.evaluation.models import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationError,
    EvaluationResult,
    EvaluationRun,
    MetricDefinition,
    MetricScore,
)
from apps.evaluation.services.evaluator_registry import load_all_metrics


class EvaluationCaseSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationCase
        fields = ["id", "question", "reference_answer", "answerable", "evidence"]


class EvaluationDatasetSerializer(serializers.ModelSerializer):
    case_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = EvaluationDataset
        fields = ["id", "name", "version", "description", "case_count", "created_at"]


class DatasetDetailSerializer(EvaluationDatasetSerializer):
    cases = EvaluationCaseSerializer(many=True, read_only=True)

    class Meta(EvaluationDatasetSerializer.Meta):
        fields = [*EvaluationDatasetSerializer.Meta.fields, "cases"]


class DatasetCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=255)
    version = serializers.CharField(max_length=32, default="1")
    description = serializers.CharField(required=False, allow_blank=True, default="")
    cases = serializers.ListField(child=serializers.DictField(), allow_empty=False)


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    name = serializers.CharField(max_length=255)
    version = serializers.CharField(max_length=32, default="1")
    description = serializers.CharField(required=False, allow_blank=True, default="")


class EvaluationRunSerializer(serializers.ModelSerializer):
    rag_config_name = serializers.CharField(source="rag_config.name", read_only=True)
    dataset_name = serializers.CharField(source="dataset.name", read_only=True, default=None)

    class Meta:
        model = EvaluationRun
        fields = [
            "id",
            "name",
            "run_type",
            "dataset",
            "dataset_name",
            "rag_config",
            "rag_config_name",
            "status",
            "k",
            "frameworks",
            "selected_metrics",
            "snapshot",
            "summary",
            "total_cases",
            "completed_cases",
            "failed_cases",
            "progress_stage",
            "error",
            "started_at",
            "finished_at",
            "created_at",
        ]
        read_only_fields = [
            "status",
            "snapshot",
            "summary",
            "total_cases",
            "completed_cases",
            "failed_cases",
            "progress_stage",
            "error",
            "started_at",
            "finished_at",
            "created_at",
        ]

    def validate_frameworks(self, value):
        unknown = set(value) - {"ragas", "deepeval"}
        if unknown:
            raise serializers.ValidationError(f"Unknown frameworks: {sorted(unknown)}")
        return value

    def validate_selected_metrics(self, value):
        unknown = set(value) - set(load_all_metrics().evaluators)
        if unknown:
            raise serializers.ValidationError(f"Unknown metrics: {sorted(unknown)}")
        return value

    def validate_k(self, value):
        if value < 1:
            raise serializers.ValidationError("Must be at least 1")
        return value

    def validate(self, attrs):
        if attrs.get("run_type", EvaluationRun.RunType.QUALITY) != EvaluationRun.RunType.SECURITY and not attrs.get(
            "dataset"
        ):
            raise serializers.ValidationError({"dataset": "Required for quality and full runs"})
        return attrs


class MetricDefinitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = MetricDefinition
        fields = [
            "id",
            "name",
            "display_name",
            "order",
            "priority",
            "question",
            "category",
            "formula",
            "description",
            "scope",
            "higher_is_better",
            "unit",
        ]


class MetricScoreSerializer(serializers.ModelSerializer):
    metric = serializers.CharField(source="metric.name")
    display_name = serializers.CharField(source="metric.display_name")
    category = serializers.CharField(source="metric.category")
    order = serializers.IntegerField(source="metric.order")
    higher_is_better = serializers.BooleanField(source="metric.higher_is_better")

    class Meta:
        model = MetricScore
        fields = ["metric", "display_name", "category", "order", "higher_is_better", "framework", "score", "details"]


class EvaluationResultSerializer(serializers.ModelSerializer):
    question = serializers.CharField(source="case.question")
    reference_answer = serializers.CharField(source="case.reference_answer")
    answerable = serializers.BooleanField(source="case.answerable")
    scores = MetricScoreSerializer(many=True, read_only=True)

    class Meta:
        model = EvaluationResult
        fields = [
            "id",
            "run",
            "case",
            "question",
            "reference_answer",
            "answerable",
            "answer",
            "contexts",
            "retrieved_chunk_ids",
            "relevant_chunk_ids",
            "citations",
            "abstained",
            "blocked",
            "retrieval_ms",
            "generation_ms",
            "total_ms",
            "prompt_tokens",
            "completion_tokens",
            "cost",
            "scores",
        ]


class EvaluationErrorSerializer(serializers.ModelSerializer):
    class Meta:
        model = EvaluationError
        fields = ["id", "case", "stage", "error_message", "created_at"]
