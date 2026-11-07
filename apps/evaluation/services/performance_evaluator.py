"""Performance metrics (6), scored once per run from the timings saved on each result."""

from apps.common.constants import MetricCategory
from apps.common.utils import percentile, safe_divide
from apps.evaluation.services.evaluator_registry import BaseMetricEvaluator, MetricValue, registry


def latency_value(values: list[float]) -> MetricValue | None:
    if not values:
        return None
    return MetricValue(
        percentile(values, 95),
        {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "count": len(values),
        },
    )


class _PerformanceMetric(BaseMetricEvaluator):
    category = MetricCategory.PERFORMANCE
    scope = "run"
    higher_is_better = False


@registry.register
class RetrievalLatency(_PerformanceMetric):
    name = "retrieval_latency"
    formula = "95th percentile of (retrieval end − retrieval start)"
    unit = "ms"

    def evaluate(self, ctx):
        return latency_value([r.retrieval_ms for r in ctx.results if not r.blocked])


@registry.register
class EndToEndLatency(_PerformanceMetric):
    name = "e2e_latency"
    formula = "95th percentile of (response end − request start)"
    unit = "ms"

    def evaluate(self, ctx):
        return latency_value([r.total_ms for r in ctx.results])


@registry.register
class TokenConsumption(_PerformanceMetric):
    name = "token_consumption"
    formula = "(input + output tokens) ÷ questions"
    unit = "tokens"

    def evaluate(self, ctx):
        if not ctx.results:
            return None
        prompt = sum(r.prompt_tokens for r in ctx.results)
        completion = sum(r.completion_tokens for r in ctx.results)
        return MetricValue(
            (prompt + completion) / len(ctx.results), {"input_tokens": prompt, "output_tokens": completion}
        )


@registry.register
class CostPerQuery(_PerformanceMetric):
    name = "cost_per_query"
    formula = "(input cost + output cost) ÷ questions"
    unit = "currency"

    def evaluate(self, ctx):
        if not ctx.results:
            return None
        return float(sum(r.cost for r in ctx.results)) / len(ctx.results)


@registry.register
class Throughput(_PerformanceMetric):
    name = "throughput"
    formula = "successful questions ÷ total seconds spent answering them"
    description = "Sequential here, one question at a time. Use the Locust file to measure concurrent load."
    higher_is_better = True
    unit = "req/s"

    def evaluate(self, ctx):
        seconds = sum(r.total_ms for r in ctx.results) / 1000
        return len(ctx.results) / seconds if seconds else None


@registry.register
class ErrorRate(_PerformanceMetric):
    name = "error_rate"
    formula = "failed questions ÷ all questions"

    def evaluate(self, ctx):
        total = len(ctx.results) + ctx.failed_cases
        if not total:
            return None
        return safe_divide(ctx.failed_cases, total)
