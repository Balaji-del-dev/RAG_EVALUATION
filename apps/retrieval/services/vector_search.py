"""Qdrant vector search. Permissions are applied as a server-side filter, before any result exists."""

from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from apps.common.exceptions import ServiceUnavailable
from config.env import env

QDRANT_ERRORS = (ResponseHandlingException, UnexpectedResponse, ConnectionError, TimeoutError)
UPSERT_BATCH = 128


@dataclass
class Hit:
    chunk_id: int
    score: float


@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    return QdrantClient(url=env.QDRANT_URL, timeout=30, check_compatibility=False)


def ping() -> bool:
    try:
        _client().get_collections()
        return True
    except QDRANT_ERRORS:
        return False


def _ensure_collection(name: str, dimension: int) -> None:
    client = _client()
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=VectorParams(size=dimension, distance=Distance.COSINE))
        client.create_payload_index(name, "document_id", PayloadSchemaType.INTEGER)


def upsert(config, chunks, vectors) -> None:
    if not chunks:
        return
    try:
        _ensure_collection(config.index_name, len(vectors[0]))
        points = [
            PointStruct(id=chunk.id, vector=vector, payload={"document_id": chunk.document_id})
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), UPSERT_BATCH):
            _client().upsert(config.index_name, points[start : start + UPSERT_BATCH], wait=True)
    except QDRANT_ERRORS as exc:
        raise ServiceUnavailable(f"Qdrant write failed: {exc}") from exc


def search(config, query_vector, k: int, allowed_document_ids: list[int]) -> list[Hit]:
    # An empty list means "may read nothing". Without this check, no filter would mean "read all".
    if not allowed_document_ids:
        return []
    try:
        if not _client().collection_exists(config.index_name):
            return []
        response = _client().query_points(
            config.index_name,
            query=query_vector,
            limit=k,
            query_filter=Filter(must=[FieldCondition(key="document_id", match=MatchAny(any=allowed_document_ids))]),
        )
    except QDRANT_ERRORS as exc:
        raise ServiceUnavailable(f"Qdrant search failed: {exc}") from exc
    return [Hit(int(point.id), float(point.score)) for point in response.points]


def delete_document(config, document_id: int) -> None:
    try:
        if _client().collection_exists(config.index_name):
            _client().delete(
                config.index_name,
                FilterSelector(
                    filter=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
                ),
                wait=True,
            )
    except QDRANT_ERRORS as exc:
        raise ServiceUnavailable(f"Qdrant delete failed: {exc}") from exc


def drop_collection(name: str) -> None:
    try:
        _client().delete_collection(name)
    except QDRANT_ERRORS as exc:
        raise ServiceUnavailable(f"Qdrant delete failed: {exc}") from exc
