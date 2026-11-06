"""Evaluation engine: registry completeness, not-applicable handling, and lexical metrics by hand."""

import math

import pytest

from apps.common.utils import normalize_answer, percentile
from apps.evaluation.services.answer_evaluator import bleu_score, lcs_length, rouge_l, rouge_n
from apps.evaluation.services.claims import split_claims, strip_citations
from apps.evaluation.services.evaluator_registry import EvalContext, RunContext, load_all_metrics

EXPECTED_METRICS = {
    "retrieval": {"precision_at_k", "recall_at_k", "f1_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k"},
    "context": {"context_precision", "context_recall", "context_relevancy", "context_coverage"},
    "answer": {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "completeness",
        "semantic_similarity",
        "exact_match",
        "bleu",
        "rouge",
    },
    "reliability": {
        "hallucination_rate",
        "abstention_accuracy",
        "contradiction_rate",
        "citation_accuracy",
        "citation_completeness",
    },
    "security": {
        "pii_leakage_rate",
        "sensitive_data_leakage_rate",
        "access_control_violation_rate",
        "prompt_injection_success_rate",
        "jailbreak_success_rate",
        "toxicity_rate",
        "guard_false_positive_rate",
        "guard_false_negative_rate",
    },
    "performance": {"retrieval_latency", "e2e_latency", "token_consumption", "cost_per_query", "throughput", "error_rate"},
}


@pytest.fixture
def registry():
    return load_all_metrics()


def test_all_37_metrics_are_registered(registry):
    assert len(registry.evaluators) == 37


def test_metrics_are_in_the_expected_categories(registry):
    assert {category: set(names) for category, names in registry.categories().items()} == EXPECTED_METRICS


def test_every_metric_documents_its_formula(registry):
    assert [e.name for e in registry.evaluators.values() if not e.formula] == []


def test_roadmap_lists_exactly_the_registered_metrics(registry):
    from apps.evaluation.services.roadmap import ROADMAP

    assert sorted(row[0] for row in ROADMAP) == sorted(registry.evaluators)


def test_roadmap_keeps_each_stage_together_and_in_order(registry):
    from apps.evaluation.services.roadmap import ROADMAP, STAGES

    seen = [registry.evaluators[row[0]].category for row in ROADMAP]
    assert list(dict.fromkeys(seen)) == [stage[0] for stage in STAGES]
    assert [seen.count(stage[0]) for stage in STAGES] == [6, 4, 8, 5, 8, 6]


def test_roadmap_first_and_last_entries():
    from apps.evaluation.services.roadmap import ROADMAP

    assert ROADMAP[0][:3] == ("precision_at_k", "Precision@K", "Critical")
    assert ROADMAP[1][:3] == ("recall_at_k", "Recall@K", "Highly Critical")
    assert ROADMAP[-1][:3] == ("error_rate", "Error Rate", "High")


def test_retrieval_metric_not_applicable_without_gold(registry):
    ctx = EvalContext(retrieved_ids=[1, 2], relevant_ids=set())
    assert registry.evaluate("precision_at_k", ctx) is None
    assert registry.evaluate("mrr", ctx) is None


def test_retrieval_metric_scores_with_gold(registry):
    ctx = EvalContext(retrieved_ids=[1, 2], relevant_ids={2}, k=2)
    assert registry.evaluate("mrr", ctx).score == 0.5


@pytest.mark.parametrize(
    "answerable,abstained,blocked,expected",
    [
        (True, False, False, 1.0),  # answered an answerable question
        (False, True, False, 1.0),  # declined an unanswerable question
        (True, True, False, 0.0),  # refused something it should have answered
        (False, False, False, 0.0),  # answered something it could not know
        (True, False, True, 0.0),  # guard wrongly blocked a legitimate question
        (False, False, True, 1.0),  # blocked, so nothing was made up
    ],
)
def test_abstention_accuracy(registry, answerable, abstained, blocked, expected):
    ctx = EvalContext(answerable=answerable, abstained=abstained, blocked=blocked)
    assert registry.evaluate("abstention_accuracy", ctx).score == expected


def test_declining_an_answerable_question_scores_zero_without_loading_models(registry):
    ctx = EvalContext(answerable=True, abstained=True, reference="The limit is 24 days.")
    for name in ("answer_correctness", "completeness", "semantic_similarity", "bleu", "rouge", "exact_match"):
        assert registry.evaluate(name, ctx).score == 0.0


def test_unanswerable_question_has_no_reference_metrics(registry):
    ctx = EvalContext(answerable=False, abstained=True)
    for name in ("answer_correctness", "completeness", "bleu", "answer_relevancy", "faithfulness", "hallucination_rate"):
        assert registry.evaluate(name, ctx) is None


def test_exact_match_ignores_case_punctuation_articles(registry):
    ctx = EvalContext(answer="The answer is 24 days! [S1]", reference="answer is 24 days")
    assert registry.evaluate("exact_match", ctx).score == 1.0


def test_context_coverage(registry):
    ctx = EvalContext(contexts=["Employees receive 24 days of annual leave."], evidence=["24 days annual leave", "parental leave 16 weeks"])
    assert registry.evaluate("context_coverage", ctx).score == 0.5


def test_citation_completeness_counts_valid_markers_only(registry):
    ctx = EvalContext(answer="First claim holds true here [S1]. Second claim is cited badly [S9]. Third has no citation.", contexts=["x"])
    assert registry.evaluate("citation_completeness", ctx).score == pytest.approx(1 / 3)


def test_citation_accuracy_counts_nonexistent_source_as_wrong_without_nli(registry):
    ctx = EvalContext(answer="This claim cites nothing real [S9].", contexts=["x"])
    value = registry.evaluate("citation_accuracy", ctx)
    assert value.score == 0.0
    assert value.details["wrong_citations"][0]["marker"] == "S9"


def test_bleu_identical_is_one():
    assert bleu_score("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)


def test_bleu_short_candidate_gets_brevity_penalty():
    # All n-gram precisions are 1 after smoothing, so BLEU equals BP = exp(1 − 6/2).
    assert bleu_score("the cat", "the cat sat on the mat") == pytest.approx(math.exp(-2))


def test_bleu_clips_repeated_words():
    # Modified precision: "the" appears 7 times but counts only as often as in the reference.
    low = bleu_score("the the the the the the the", "the cat is on the mat")
    assert low < 0.3


def test_bleu_no_overlap_is_zero():
    assert bleu_score("completely different words", "the cat sat") == 0.0
    assert bleu_score("", "the cat") == 0.0


def test_rouge_1_worked_example():
    result = rouge_n("the cat was found", "the cat sat", 1)
    assert result["precision"] == pytest.approx(2 / 4)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(4 / 7)


def test_rouge_2_is_bigram_recall():
    assert rouge_n("the cat sat", "the cat sat down", 2)["recall"] == pytest.approx(2 / 3)


def test_rouge_l_uses_longest_common_subsequence():
    assert lcs_length(["a", "b", "c", "d"], ["a", "x", "c", "d"]) == 3
    assert rouge_l("a b c d", "a x c d")["recall"] == pytest.approx(3 / 4)


def test_split_claims_attaches_stranded_citation():
    claims = split_claims("Employees receive 24 days of leave. [S1]\n- Unused days can carry over [S2].")
    assert [c.text for c in claims] == ["Employees receive 24 days of leave.", "Unused days can carry over."]
    assert claims[0].markers == [1]
    assert claims[1].markers == [2]


def test_split_claims_skips_lines_that_only_introduce_a_list():
    claims = split_claims("The leave policy includes the following:\n\n- **Annual Leave**: Employees get 24 days.")
    assert [c.text for c in claims] == ["Annual Leave: Employees get 24 days."]


def test_split_claims_skips_fragments():
    assert split_claims("Yes.") == []


def test_strip_citations():
    assert strip_citations("It is 24 days [S1] in total [S2].") == "It is 24 days in total."


def test_normalize_answer():
    assert normalize_answer("The  Cat, an animal!") == "cat animal"


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert percentile([], 95) == 0.0


class FakeResult:
    def __init__(self, **kwargs):
        self.__dict__.update({"retrieval_ms": 10.0, "total_ms": 100.0, "blocked": False, "prompt_tokens": 100, "completion_tokens": 50, "cost": 2, **kwargs})


def test_performance_metrics(registry):
    ctx = RunContext(run=None, results=[FakeResult(), FakeResult(total_ms=300.0)], failed_cases=2)
    assert registry.evaluate("e2e_latency", ctx).details["p50"] == 200.0
    assert registry.evaluate("token_consumption", ctx).score == 150.0
    assert registry.evaluate("cost_per_query", ctx).score == 2.0
    assert registry.evaluate("throughput", ctx).score == 5.0  # 2 answers in 0.4 seconds
    assert registry.evaluate("error_rate", ctx).score == 0.5


def test_performance_metrics_not_applicable_on_security_only_run(registry):
    ctx = RunContext(run=None)
    for name in ("retrieval_latency", "e2e_latency", "token_consumption", "cost_per_query", "throughput", "error_rate"):
        assert registry.evaluate(name, ctx) is None
