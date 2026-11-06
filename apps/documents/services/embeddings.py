"""Sentence Transformers embeddings. Models load on first use, because they cost hundreds of MB."""

import hashlib
import threading
from functools import lru_cache

from django.core.cache import cache

QUERY_CACHE_SECONDS = 24 * 3600
BATCH_SIZE = 16
# BGE models expect this instruction on queries (not on passages).
QUERY_PREFIXES = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}
_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _model(model_name: str):
    with _lock:
        return _load(model_name)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model(model_name).encode(
        texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    )
    return vectors.tolist()


def embed_query(text: str, model_name: str) -> list[float]:
    key = "qemb:" + hashlib.sha256(f"{model_name}|{text}".encode()).hexdigest()
    cached = cache.get(key)
    if cached is not None:
        return cached
    vector = embed_texts([QUERY_PREFIXES.get(model_name, "") + text], model_name)[0]
    cache.set(key, vector, QUERY_CACHE_SECONDS)
    return vector


def embedding_dimension(model_name: str) -> int:
    return _model(model_name).get_sentence_embedding_dimension()
