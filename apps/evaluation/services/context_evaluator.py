"""Context metrics (4): is the text handed to the LLM any good, regardless of what the LLM does with it?"""

from django.conf import settings

from apps.common.constants import MetricCategory
from apps.common.utils import safe_divide
from apps.documents.services import embeddings
from apps.evaluation.services.claims import split_sentences
from apps.evaluation.services.dataset_manager import evidence_tokens
from apps.evaluation.services.evaluator_registry import BaseMetricEvaluator, MetricValue, registry
from apps.retrieval.services.bm25_search import analyze


def average_precision(relevance: list[bool]) -> float:
    """Σ (Precision@i × rel(i)) ÷ number of relevant items. Rewards relevant chunks ranked early."""
    hits, total = 0, 0.0
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            hits += 1
            total += hits / rank
    return safe_divide(total, hits)


@registry.register
class ContextPrecision(BaseMetricEvaluator):
    name = "context_precision"
    category = MetricCategory.CONTEXT
    formula = "Σ (Precision@i × rel(i)) ÷ relevant chunks retrieved"
    description = "Whether the relevant chunks are ranked ahead of the irrelevant ones."

    def evaluate(self, ctx):
        if not ctx.relevant_ids or not ctx.retrieved_ids:
            return None if not ctx.relevant_ids else 0.0
        relevance = [chunk_id in ctx.relevant_ids for chunk_id in ctx.retrieved_ids[: ctx.k]]
        return average_precision(relevance)


@registry.register
class ContextRecall(BaseMetricEvaluator):
    name = "context_recall"
    category = MetricCategory.CONTEXT
    formula = "reference claims entailed by the context ÷ all reference claims"
    description = "Whether the context holds what is needed to give the reference answer."

    def evaluate(self, ctx):
        if not ctx.reference_claims or not ctx.answerable:
            return None
        verdicts = ctx.reference_vs_context
        return MetricValue(
            safe_divide(sum(v.supported for v in verdicts), len(verdicts)),
            {"unsupported_reference_claims": [v.claim for v in verdicts if not v.supported]},
        )


@registry.register
class ContextRelevancy(BaseMetricEvaluator):
    name = "context_relevancy"
    category = MetricCategory.CONTEXT
    formula = "context sentences with cosine(question, sentence) ≥ threshold ÷ all context sentences"
    description = "How much of the context is on topic, rather than noise the LLM must ignore."

    def evaluate(self, ctx):
        sentences = [sentence for context in ctx.contexts for sentence in split_sentences(context)]
        if not sentences:
            return None
        model = settings.EVALUATION["JUDGE_EMBEDDING_MODEL"]
        vectors = embeddings.embed_texts([ctx.question, *sentences], model)
        threshold = settings.EVALUATION["CONTEXT_RELEVANCY_THRESHOLD"]
        question = vectors[0]
        relevant = sum(1 for vector in vectors[1:] if sum(a * b for a, b in zip(question, vector, strict=True)) >= threshold)
        return safe_divide(relevant, len(sentences))


@registry.register
class ContextCoverage(BaseMetricEvaluator):
    name = "context_coverage"
    category = MetricCategory.CONTEXT
    formula = "gold evidence passages found in the context ÷ all gold evidence passages"
    description = "Checked against hand-labelled evidence, without any model."

    def evaluate(self, ctx):
        if not ctx.evidence:
            return None
        threshold = settings.EVALUATION["EVIDENCE_MATCH_THRESHOLD"]
        context_terms = set(analyze(" ".join(ctx.contexts)))
        covered = 0
        for passage in ctx.evidence:
            terms = evidence_tokens(passage)
            if terms and len(terms & context_terms) / len(terms) >= threshold:
                covered += 1
        return safe_divide(covered, len(ctx.evidence))
