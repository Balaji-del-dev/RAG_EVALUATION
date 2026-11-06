"""Answer metrics (8): semantic ones use NLI or embeddings, the lexical ones are implemented from their
definitions (BLEU, ROUGE), so that the numbers can be checked by hand."""

import math
from collections import Counter

from apps.common.constants import MetricCategory
from apps.common.utils import normalize_answer, safe_divide, word_tokens
from apps.evaluation.services import nli
from apps.evaluation.services.evaluator_registry import BaseMetricEvaluator, MetricValue, registry

CORRECTNESS_FACTUAL_WEIGHT = 0.75


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(candidate: str, reference: str, max_n: int = 4) -> float:
    """BLEU = BP × exp(mean of log modified n-gram precisions), with add-one smoothing for n > 1 so that
    a short answer missing all 4-grams does not score exactly 0 (Lin & Och, 2004).
    BP = 1 if the candidate is at least as long as the reference, else exp(1 − r/c)."""
    candidate_tokens, reference_tokens = word_tokens(candidate), word_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    log_precisions = []
    for n in range(1, max_n + 1):
        candidate_grams, reference_grams = ngrams(candidate_tokens, n), ngrams(reference_tokens, n)
        matches = sum(min(count, reference_grams[gram]) for gram, count in candidate_grams.items())
        total = sum(candidate_grams.values())
        if n > 1:
            matches, total = matches + 1, total + 1
        if matches == 0 or total == 0:
            return 0.0
        log_precisions.append(math.log(matches / total))
    c, r = len(candidate_tokens), len(reference_tokens)
    brevity = 1.0 if c >= r else math.exp(1 - r / c)
    return brevity * math.exp(sum(log_precisions) / max_n)


def rouge_n(candidate: str, reference: str, n: int) -> dict:
    """ROUGE-N: precision, recall and F1 of the n-grams shared with the reference."""
    candidate_grams, reference_grams = ngrams(word_tokens(candidate), n), ngrams(word_tokens(reference), n)
    overlap = sum((candidate_grams & reference_grams).values())
    precision = safe_divide(overlap, sum(candidate_grams.values()))
    recall = safe_divide(overlap, sum(reference_grams.values()))
    return {"precision": precision, "recall": recall, "f1": safe_divide(2 * precision * recall, precision + recall)}


def lcs_length(first: list[str], second: list[str]) -> int:
    previous = [0] * (len(second) + 1)
    for a in first:
        current = [0]
        for j, b in enumerate(second, start=1):
            current.append(previous[j - 1] + 1 if a == b else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> dict:
    candidate_tokens, reference_tokens = word_tokens(candidate), word_tokens(reference)
    overlap = lcs_length(candidate_tokens, reference_tokens)
    precision = safe_divide(overlap, len(candidate_tokens))
    recall = safe_divide(overlap, len(reference_tokens))
    return {"precision": precision, "recall": recall, "f1": safe_divide(2 * precision * recall, precision + recall)}


class _AnswerMetric(BaseMetricEvaluator):
    category = MetricCategory.ANSWER


class _ComparedToReference(_AnswerMetric):
    """Base for metrics comparing the answer with a reference. Declining an answerable question scores 0.
    Unanswerable questions have no reference, so they are not scored here (see Abstention Accuracy)."""

    def evaluate(self, ctx):
        if not ctx.answerable or not ctx.reference:
            return None
        if ctx.declined:
            return 0.0
        return self.compute(ctx)


@registry.register
class Faithfulness(_AnswerMetric):
    name = "faithfulness"
    formula = "answer claims entailed by the context ÷ all answer claims"
    description = "Whether the answer stays within the retrieved evidence."

    def evaluate(self, ctx):
        verdicts = ctx.answer_vs_context
        if not verdicts:
            return None
        return MetricValue(
            safe_divide(sum(v.supported for v in verdicts), len(verdicts)),
            {"claims": [{"claim": v.claim, "entailment": round(v.entailment, 3)} for v in verdicts]},
        )


@registry.register
class AnswerRelevancy(_AnswerMetric):
    name = "answer_relevancy"
    formula = "cosine(question, answer)"
    description = "Whether the answer addresses the question that was asked."

    def evaluate(self, ctx):
        if ctx.declined:
            return 0.0 if ctx.answerable else None
        return ctx.cosine(ctx.question, ctx.clean_answer)


@registry.register
class SemanticSimilarity(_ComparedToReference):
    name = "semantic_similarity"
    formula = "cosine(answer embedding, reference embedding)"
    description = "Same meaning as the reference, even in different words."

    def compute(self, ctx):
        return ctx.cosine(ctx.clean_answer, ctx.reference)


@registry.register
class AnswerCorrectness(_ComparedToReference):
    name = "answer_correctness"
    formula = "0.75 × claim F1 + 0.25 × semantic similarity"
    description = (
        "Claim F1. TP: answer claims that match the reference. FP: claims that match neither the reference nor the "
        "context, because a true extra detail is not an error. FN: reference claims missing from the answer."
    )

    def compute(self, ctx):
        reference_claims = [claim.text for claim in ctx.reference_claims]
        in_context = {verdict.claim: verdict.supported for verdict in ctx.answer_vs_context}
        tp = fp = 0
        for verdict in ctx.answer_vs_reference:
            # A longer sentence that contains a reference fact is right, even if the reference does not entail it.
            if verdict.supported or any(v.supported for v in nli.judge_claims(reference_claims, [verdict.claim])):
                tp += 1
            elif not in_context.get(verdict.claim, False):
                fp += 1
        fn = sum(not verdict.supported for verdict in ctx.reference_vs_answer)
        f1 = safe_divide(tp, tp + 0.5 * (fp + fn))
        similarity = ctx.cosine(ctx.clean_answer, ctx.reference)
        score = CORRECTNESS_FACTUAL_WEIGHT * f1 + (1 - CORRECTNESS_FACTUAL_WEIGHT) * similarity
        return MetricValue(score, {"tp": tp, "fp": fp, "fn": fn, "f1": f1, "similarity": similarity})


@registry.register
class Completeness(_ComparedToReference):
    name = "completeness"
    formula = "reference claims entailed by the answer ÷ all reference claims"
    description = "Whether the answer includes everything the reference says."

    def compute(self, ctx):
        verdicts = ctx.reference_vs_answer
        return MetricValue(
            safe_divide(sum(v.supported for v in verdicts), len(verdicts)),
            {"missing": [v.claim for v in verdicts if not v.supported]},
        )


@registry.register
class ExactMatch(_ComparedToReference):
    name = "exact_match"
    formula = "1 if normalized answer = normalized reference, else 0"
    description = "Case, punctuation and articles are ignored. Expect this near 0 for free-text answers."

    def compute(self, ctx):
        return 1.0 if normalize_answer(ctx.clean_answer) == normalize_answer(ctx.reference) else 0.0


@registry.register
class Bleu(_ComparedToReference):
    name = "bleu"
    formula = "BP × exp(¼ Σ log pₙ), n = 1…4, pₙ = clipped n-gram precision"
    description = "N-gram precision against the reference, penalizing short answers."

    def compute(self, ctx):
        return bleu_score(ctx.clean_answer, ctx.reference)


@registry.register
class Rouge(_ComparedToReference):
    name = "rouge"
    formula = "ROUGE-2 recall = shared bigrams ÷ reference bigrams"
    description = "N-gram recall against the reference. Details also hold ROUGE-1 and ROUGE-L."

    def compute(self, ctx):
        details = {
            "rouge_1": rouge_n(ctx.clean_answer, ctx.reference, 1),
            "rouge_2": rouge_n(ctx.clean_answer, ctx.reference, 2),
            "rouge_l": rouge_l(ctx.clean_answer, ctx.reference),
        }
        return MetricValue(details["rouge_2"]["recall"], details)
