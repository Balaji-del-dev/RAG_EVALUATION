"""Cross-encoder reranking: reads the query and chunk together, so it is slower but more accurate
than the bi-encoder used for first-stage search."""

import threading
from functools import lru_cache

_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")


def score_pairs(query: str, texts: list[str], model_name: str) -> list[float]:
    if not texts:
        return []
    with _lock:
        model = _load(model_name)
    return [float(score) for score in model.predict([(query, text) for text in texts], show_progress_bar=False)]
