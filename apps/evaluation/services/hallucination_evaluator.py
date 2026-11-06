"""Reliability metrics (5): does the system make things up, and does it admit when it does not know?"""

from apps.common.constants import MetricCategory
from apps.common.utils import safe_divide
from apps.evaluation.services import nli
from apps.evaluation.services.evaluator_registry import BaseMetricEvaluator, MetricValue, registry


class _ReliabilityMetric(BaseMetricEvaluator):
    category = MetricCategory.RELIABILITY


@registry.register
class HallucinationRate(_ReliabilityMetric):
    name = "hallucination_rate"
    formula = "answer claims not entailed by the context ÷ all answer claims"
    description = "The inverse of faithfulness, stored separately because this is the number people ask for."
    higher_is_better = False

    def evaluate(self, ctx):
        verdicts = ctx.answer_vs_context
        if not verdicts:
            return None
        unsupported = [v.claim for v in verdicts if not v.supported]
        return MetricValue(len(unsupported) / len(verdicts), {"unsupported_claims": unsupported})


@registry.register
class AbstentionAccuracy(_ReliabilityMetric):
    name = "abstention_accuracy"
    formula = "1 if the system declined exactly when the question was unanswerable, else 0"
    description = "Answering an unanswerable question and refusing an answerable one both score 0."

    def evaluate(self, ctx):
        return 1.0 if ctx.declined == (not ctx.answerable) else 0.0


@registry.register
class ContradictionRate(_ReliabilityMetric):
    name = "contradiction_rate"
    formula = "answer claims contradicted by the context ÷ all answer claims"
    description = "Worse than an unsupported claim: the answer says the opposite of the evidence."
    higher_is_better = False

    def evaluate(self, ctx):
        verdicts = ctx.answer_vs_context
        if not verdicts:
            return None
        contradicted = [v.claim for v in verdicts if v.contradicted]
        return MetricValue(len(contradicted) / len(verdicts), {"contradicted_claims": contradicted})


@registry.register
class CitationAccuracy(_ReliabilityMetric):
    name = "citation_accuracy"
    formula = "citations whose source entails the cited sentence ÷ all citations"
    description = "A citation to a source that does not exist, or does not say this, counts as wrong."

    def evaluate(self, ctx):
        cited = [(claim, marker) for claim in ctx.answer_claims for marker in claim.markers]
        if not cited:
            return None
        correct, wrong = 0, []
        for claim, marker in cited:
            if 1 <= marker <= len(ctx.contexts) and nli.judge_claims([claim.text], [ctx.contexts[marker - 1]])[0].supported:
                correct += 1
            else:
                wrong.append({"claim": claim.text, "marker": f"S{marker}"})
        return MetricValue(correct / len(cited), {"wrong_citations": wrong})


@registry.register
class CitationCompleteness(_ReliabilityMetric):
    name = "citation_completeness"
    formula = "answer claims with at least one valid citation ÷ all answer claims"
    description = "Whether every claim can be traced to a source. Whether the source supports it is Citation Accuracy."

    def evaluate(self, ctx):
        if not ctx.answer_claims:
            return None
        cited = sum(any(1 <= m <= len(ctx.contexts) for m in claim.markers) for claim in ctx.answer_claims)
        return safe_divide(cited, len(ctx.answer_claims))
