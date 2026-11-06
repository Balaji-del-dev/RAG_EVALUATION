"""Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank(d)), summed over every ranked list containing d.

It uses ranks, not scores, so it can merge BM25 scores and cosine similarities without normalizing them.
"""


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
