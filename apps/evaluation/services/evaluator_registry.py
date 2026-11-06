"""The evaluation engine's core: one base class, one registry, and no per-metric views.

A case-level metric gets an EvalContext (one question). A run-level metric gets a RunContext (the whole
run). Returning None means "not applicable here": for example Recall@K on a question with no gold
evidence. Such scores are not stored, because a false zero would drag the average down.
"""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from django.conf import settings

from apps.common.constants import MetricCategory
from apps.documents.services import embeddings
from apps.evaluation.services import nli
from apps.evaluation.services.claims import Claim, split_claims, strip_citations


@dataclass
class MetricValue:
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class EvalContext:
    question: str = ""
    answer: str = ""
    reference: str = ""
    contexts: list[str] = field(default_factory=list)
    retrieved_ids: list[int] = field(default_factory=list)
    relevant_ids: set[int] = field(default_factory=set)
    evidence: list[str] = field(default_factory=list)
    answerable: bool = True
    abstained: bool = False
    blocked: bool = False
    k: int = 5

    @property
    def declined(self) -> bool:
        return self.abstained or self.blocked

    @cached_property
    def answer_claims(self) -> list[Claim]:
        return [] if self.declined else split_claims(self.answer)

    @cached_property
    def reference_claims(self) -> list[Claim]:
        return split_claims(self.reference)

    @cached_property
    def clean_answer(self) -> str:
        return strip_citations(self.answer)

    # The NLI verdicts are shared: several metrics need the same judgements, so each is computed once.
    @cached_property
    def answer_vs_context(self) -> list[nli.ClaimVerdict]:
        return nli.judge_claims([c.text for c in self.answer_claims], self.contexts)

    @cached_property
    def reference_vs_context(self) -> list[nli.ClaimVerdict]:
        return nli.judge_claims([c.text for c in self.reference_claims], self.contexts)

    @cached_property
    def answer_vs_reference(self) -> list[nli.ClaimVerdict]:
        return nli.judge_claims([c.text for c in self.answer_claims], [self.reference])

    @cached_property
    def reference_vs_answer(self) -> list[nli.ClaimVerdict]:
        return nli.judge_claims([c.text for c in self.reference_claims], [self.clean_answer])

    def cosine(self, first: str, second: str) -> float:
        """Similarity with the fixed judge model. If it followed the RagConfig's embedding model, scores
        from configurations using different models could not be compared."""
        model = settings.EVALUATION["JUDGE_EMBEDDING_MODEL"]
        vectors = np.array(embeddings.embed_texts([first, second], model))
        return float(np.clip(vectors[0] @ vectors[1], 0.0, 1.0))


@dataclass
class RunContext:
    run: object
    results: list = field(default_factory=list)
    security_results: list = field(default_factory=list)
    failed_cases: int = 0


class BaseMetricEvaluator:
    name: str = ""
    category: str = ""
    formula: str = ""
    description: str = ""
    scope: str = "case"
    higher_is_better: bool = True
    unit: str = ""

    def evaluate(self, ctx) -> float | MetricValue | None:
        raise NotImplementedError


class EvaluatorRegistry:
    def __init__(self):
        self.evaluators: dict[str, BaseMetricEvaluator] = {}

    def register(self, cls):
        """Class decorator. Registering the same name twice is a bug, so it fails loudly."""
        instance = cls()
        if instance.name in self.evaluators:
            raise ValueError(f"Metric {instance.name} is already registered")
        self.evaluators[instance.name] = instance
        return cls

    def evaluate(self, name: str, ctx) -> MetricValue | None:
        value = self.evaluators[name].evaluate(ctx)
        if value is None or isinstance(value, MetricValue):
            return value
        return MetricValue(float(value))

    def for_scope(self, scope: str, selected: list[str] | None = None) -> list[BaseMetricEvaluator]:
        return [e for e in self.evaluators.values() if e.scope == scope and (not selected or e.name in selected)]

    def categories(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {category.value: [] for category in MetricCategory}
        for evaluator in self.evaluators.values():
            grouped[evaluator.category].append(evaluator.name)
        return grouped


registry = EvaluatorRegistry()


def load_all_metrics() -> EvaluatorRegistry:
    """Import every metric module, so that their @registry.register decorators run."""
    from apps.evaluation.services import (  # noqa: F401
        answer_evaluator,
        context_evaluator,
        hallucination_evaluator,
        performance_evaluator,
        retrieval_evaluator,
        security_evaluator,
    )

    return registry
