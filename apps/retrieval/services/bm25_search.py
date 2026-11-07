"""BM25 retrieval with two interchangeable backends, selected by BM25_BACKEND.

opensearch  The production backend.
local       Okapi BM25 in Python over the chunk rows in Postgres. It needs no extra service, which
            matters on a small machine, and it shows the formula. It reads every chunk of an index
            per query, so it only suits small corpora.
"""

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import PurePath

from apps.common.exceptions import ServiceUnavailable
from apps.retrieval.services.vector_search import Hit
from config.env import env

K1 = 1.2
B = 0.75
TERM_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "should",
        "such",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


def searchable_text(filename: str, text: str) -> str:
    """A chunk with the name of its document. The work history in a resume never says whose it is, so without this
    "balamurugan experience" finds an email that repeats the name and says nothing about experience."""
    title = re.sub(r"[\W_]+", " ", PurePath(filename).stem).strip()
    return f"{title}\n{text}"


def analyze(text: str) -> list[str]:
    return [term for term in TERM_RE.findall(text.lower()) if term not in STOPWORDS]


def okapi_scores(query_terms, documents) -> list[float]:
    """Score tokenized documents against a query.

    score(D) = Σ IDF(t) · f(t,D)·(k1+1) / (f(t,D) + k1·(1 − b + b·|D|/avgdl))
    IDF(t)   = ln(1 + (N − n(t) + 0.5) / (n(t) + 0.5))
    """
    count = len(documents)
    if not count:
        return []
    average_length = sum(len(d) for d in documents) / count or 1.0
    document_frequency = Counter(term for d in documents for term in set(d))
    scores = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            seen_in = document_frequency[term]
            idf = math.log(1 + (count - seen_in + 0.5) / (seen_in + 0.5))
            score += idf * frequency * (K1 + 1) / (frequency + K1 * (1 - B + B * len(document) / average_length))
        scores.append(score)
    return scores


class LocalBM25:
    name = "local"

    def ping(self) -> bool:
        return True

    def index(self, config, chunks) -> None:
        """Nothing to do: the chunk rows in Postgres are the index."""

    def delete_document(self, config, document_id: int) -> None:
        """Nothing to do: deleting the chunk rows removes them from search."""

    def search(self, config, query: str, k: int, allowed_document_ids: list[int]) -> list[Hit]:
        from apps.documents.models import DocumentChunk

        if not allowed_document_ids:
            return []
        rows = list(
            DocumentChunk.objects.filter(index_key=config.index_key, document_id__in=allowed_document_ids).values_list(
                "id", "chunk_text", "document__filename"
            )
        )
        scores = okapi_scores(analyze(query), [analyze(searchable_text(filename, text)) for _, text, filename in rows])
        ranked = sorted(zip(rows, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [Hit(row[0], score) for row, score in ranked[:k] if score > 0]


class OpenSearchBM25:
    name = "opensearch"
    MAPPINGS = {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {"chunk_text": {"type": "text", "analyzer": "english"}, "document_id": {"type": "integer"}}
        },
    }

    def __init__(self):
        from opensearchpy import OpenSearch

        auth = (env.OPENSEARCH_USER, env.OPENSEARCH_PASSWORD) if env.OPENSEARCH_PASSWORD else None
        self.client = OpenSearch(
            hosts=[env.OPENSEARCH_URL],
            http_auth=auth,
            verify_certs=env.OPENSEARCH_VERIFY_CERTS,
            ssl_show_warn=False,
            timeout=30,
        )

    @staticmethod
    def _errors():
        from opensearchpy.exceptions import OpenSearchException

        return (OpenSearchException, ConnectionError, TimeoutError)

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except self._errors():
            return False

    def index(self, config, chunks) -> None:
        from opensearchpy.helpers import bulk

        if not chunks:
            return
        try:
            if not self.client.indices.exists(index=config.index_name):
                self.client.indices.create(index=config.index_name, body=self.MAPPINGS)
            actions = (
                {
                    "_index": config.index_name,
                    "_id": chunk.id,
                    "chunk_text": chunk.chunk_text,
                    "document_id": chunk.document_id,
                }
                for chunk in chunks
            )
            bulk(self.client, actions, refresh="wait_for")
        except self._errors() as exc:
            raise ServiceUnavailable(f"OpenSearch write failed: {exc}") from exc

    def delete_document(self, config, document_id: int) -> None:
        try:
            if self.client.indices.exists(index=config.index_name):
                self.client.delete_by_query(
                    index=config.index_name,
                    body={"query": {"term": {"document_id": document_id}}},
                    refresh=True,
                    conflicts="proceed",
                )
        except self._errors() as exc:
            raise ServiceUnavailable(f"OpenSearch delete failed: {exc}") from exc

    def search(self, config, query: str, k: int, allowed_document_ids: list[int]) -> list[Hit]:
        if not allowed_document_ids:
            return []
        body = {
            "size": k,
            "_source": False,
            "query": {
                "bool": {
                    "must": {"match": {"chunk_text": query}},
                    "filter": {"terms": {"document_id": allowed_document_ids}},
                }
            },
        }
        try:
            if not self.client.indices.exists(index=config.index_name):
                return []
            response = self.client.search(index=config.index_name, body=body)
        except self._errors() as exc:
            raise ServiceUnavailable(f"OpenSearch search failed: {exc}") from exc
        return [Hit(int(hit["_id"]), float(hit["_score"])) for hit in response["hits"]["hits"]]


@lru_cache(maxsize=1)
def backend():
    return OpenSearchBM25() if env.BM25_BACKEND == "opensearch" else LocalBM25()
