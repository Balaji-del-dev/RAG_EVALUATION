from django.db.models import Count, Sum

from apps.monitoring.models import TokenUsage


def usage_by_model() -> list[dict]:
    return list(
        TokenUsage.objects.values("model_name")
        .annotate(requests=Count("id"), input_tokens=Sum("input_tokens"), output_tokens=Sum("output_tokens"))
        .order_by("-requests")
    )
