"""Retrieval metrics (6). Pure ranking math over chunk ids, with no models involved."""

import math

from apps.common.constants import MetricCategory
from apps.common.utils import safe_divide
from apps.evaluation.services.evaluator_registry import BaseMetricEvaluator, registry


def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    top = retrieved[:k]
    return safe_divide(sum(1 for item in top if item in relevant), len(top))


def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    return safe_divide(sum(1 for item in retrieved[:k] if item in relevant), len(relevant))


def f1_at_k(retrieved: list, relevant: set, k: int) -> float:
    precision, recall = precision_at_k(retrieved, relevant, k), recall_at_k(retrieved, relevant, k)
    return safe_divide(2 * precision * recall, precision + recall)


def hit_rate_at_k(retrieved: list, relevant: set, k: int) -> float:
    return 1.0 if any(item in relevant for item in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list, relevant: set) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, relevant: set, k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return safe_divide(dcg, idcg)


class _RetrievalMetric(BaseMetricEvaluator):
    category = MetricCategory.RETRIEVAL

    def evaluate(self, ctx):
        # An unanswerable question has no gold chunks, so ranking quality means nothing for it.
        if not ctx.relevant_ids:
            return None
        return self.compute(ctx.retrieved_ids, ctx.relevant_ids, ctx.k)


@registry.register
class PrecisionAtK(_RetrievalMetric):
    name = "precision_at_k"
    formula = "relevant chunks in top K ÷ chunks in top K"
    description = "How much of what was retrieved is useful."

    def compute(self, retrieved, relevant, k):
        return precision_at_k(retrieved, relevant, k)


@registry.register
class RecallAtK(_RetrievalMetric):
    name = "recall_at_k"
    formula = "relevant chunks in top K ÷ all relevant chunks"
    description = "How much of what is useful was retrieved."

    def compute(self, retrieved, relevant, k):
        return recall_at_k(retrieved, relevant, k)


@registry.register
class F1AtK(_RetrievalMetric):
    name = "f1_at_k"
    formula = "2 × Precision@K × Recall@K ÷ (Precision@K + Recall@K)"
    description = "Harmonic mean, which is low if either precision or recall is low."

    def compute(self, retrieved, relevant, k):
        return f1_at_k(retrieved, relevant, k)


@registry.register
class HitRateAtK(_RetrievalMetric):
    name = "hit_rate_at_k"
    formula = "1 if any relevant chunk is in the top K, else 0"
    description = "Whether the retriever found anything useful at all."

    def compute(self, retrieved, relevant, k):
        return hit_rate_at_k(retrieved, relevant, k)


@registry.register
class MeanReciprocalRank(_RetrievalMetric):
    name = "mrr"
    formula = "1 ÷ rank of the first relevant chunk (averaged over questions)"
    description = "How high the first useful chunk appears."

    def compute(self, retrieved, relevant, k):
        return reciprocal_rank(retrieved, relevant)


@registry.register
class NDCGAtK(_RetrievalMetric):
    name = "ndcg_at_k"
    formula = "DCG ÷ IDCG, where DCG = Σ rel(i) ÷ log2(i + 1)"
    description = "Ranking quality: relevant chunks count for more the higher they rank."

    def compute(self, retrieved, relevant, k):
        return ndcg_at_k(retrieved, relevant, k)
