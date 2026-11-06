"""Maximal Marginal Relevance: pick chunks that match the query but do not repeat each other.

MMR = argmax over unselected d of [ λ·sim(q, d) − (1 − λ)·max over selected s of sim(d, s) ]

Vectors are expected to be normalized, so a dot product is cosine similarity.
"""

import numpy as np


def maximal_marginal_relevance(query_vector, document_vectors, top_k: int, lambda_: float = 0.7) -> list[int]:
    """Return the positions of the selected documents, in selection order."""
    if len(document_vectors) == 0 or top_k <= 0:
        return []
    documents = np.asarray(document_vectors, dtype=float)
    relevance = documents @ np.asarray(query_vector, dtype=float)
    selected = [int(np.argmax(relevance))]
    remaining = set(range(len(documents))) - set(selected)
    while remaining and len(selected) < top_k:
        candidates = sorted(remaining)
        redundancy = (documents[candidates] @ documents[selected].T).max(axis=1)
        scores = lambda_ * relevance[candidates] - (1 - lambda_) * redundancy
        best = candidates[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return selected
