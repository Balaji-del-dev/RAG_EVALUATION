"""The retrieval pipeline: BM25 + vector -> RRF -> permission re-check -> rerank -> MMR -> top K."""

import time
from dataclasses import dataclass, field

from apps.documents.models import DocumentChunk
from apps.documents.services import embeddings
from apps.monitoring.services.tracing import span
from apps.retrieval.models import RagConfig
from apps.retrieval.services import bm25_search, mmr, reranking, rrf, vector_search


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    page_number: int | None
    text: str
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    # True when the reranker judged even the best chunk irrelevant to the question.
    below_relevance_floor: bool = False

    @property
    def chunk_ids(self) -> list[int]:
        return [chunk.chunk_id for chunk in self.chunks]


def _timed(result: RetrievalResult, stage: str, started: float) -> None:
    result.timings_ms[stage] = round((time.perf_counter() - started) * 1000, 2)


def retrieve(query: str, config: RagConfig, allowed_document_ids: list[int]) -> RetrievalResult:
    """Return the best chunks that the caller may read.

    `allowed_document_ids` has no default on purpose: a caller must state its permissions. They are
    applied inside both search engines, and again here against SQL, so a stale index entry cannot
    reach the LLM.
    """
    result = RetrievalResult()
    allowed = set(allowed_document_ids)
    if not allowed:
        return result

    ranked_lists = []
    query_vector = None
    if config.retrieval_mode in (RagConfig.Mode.VECTOR, RagConfig.Mode.HYBRID):
        started = time.perf_counter()
        with span("retrieval.vector"):
            query_vector = embeddings.embed_query(query, config.embedding_model)
            hits = vector_search.search(config, query_vector, config.candidate_k, sorted(allowed))
        _timed(result, "vector", started)
        ranked_lists.append([hit.chunk_id for hit in hits])
        result.stages["vector"] = [{"chunk_id": hit.chunk_id, "score": round(hit.score, 4)} for hit in hits]

    if config.retrieval_mode in (RagConfig.Mode.BM25, RagConfig.Mode.HYBRID):
        started = time.perf_counter()
        with span("retrieval.bm25"):
            hits = bm25_search.backend().search(config, query, config.candidate_k, sorted(allowed))
        _timed(result, "bm25", started)
        ranked_lists.append([hit.chunk_id for hit in hits])
        result.stages["bm25"] = [{"chunk_id": hit.chunk_id, "score": round(hit.score, 4)} for hit in hits]

    fused = rrf.reciprocal_rank_fusion(ranked_lists, config.rrf_k)
    result.stages["rrf"] = [{"chunk_id": chunk_id, "score": round(score, 5)} for chunk_id, score in fused]
    if not fused:
        return result

    # Second enforcement point: load only rows whose document is still allowed and still indexed.
    rows = (
        DocumentChunk.objects.filter(
            id__in=[chunk_id for chunk_id, _ in fused], index_key=config.index_key, document_id__in=allowed
        )
        .select_related("document")
    )
    by_id = {row.id: row for row in rows}
    candidates = [(by_id[chunk_id], score) for chunk_id, score in fused if chunk_id in by_id]

    if config.reranker and candidates:
        started = time.perf_counter()
        with span("retrieval.rerank"):
            texts = [bm25_search.searchable_text(row.document.filename, row.chunk_text) for row, _ in candidates]
            scores = reranking.score_pairs(query, texts, config.reranker)
        _timed(result, "rerank", started)
        candidates = sorted(
            ((row, score) for (row, _), score in zip(candidates, scores, strict=True)),
            key=lambda pair: pair[1],
            reverse=True,
        )[: config.rerank_top_n]
        result.stages["rerank"] = [{"chunk_id": row.id, "score": round(score, 4)} for row, score in candidates]
        if config.rerank_min_score is not None and candidates and candidates[0][1] < config.rerank_min_score:
            result.below_relevance_floor = True

    if config.use_mmr and len(candidates) > config.top_k:
        started = time.perf_counter()
        with span("retrieval.mmr"):
            if query_vector is None:
                query_vector = embeddings.embed_query(query, config.embedding_model)
            vectors = embeddings.embed_texts([row.chunk_text for row, _ in candidates], config.embedding_model)
            order = mmr.maximal_marginal_relevance(query_vector, vectors, config.top_k, config.mmr_lambda)
        candidates = [candidates[position] for position in order]
        _timed(result, "mmr", started)
        result.stages["mmr"] = [{"chunk_id": row.id, "score": round(score, 4)} for row, score in candidates]
        # MMR chooses a varied set. Within it the relevance order is the better one, and S1 should be the best source.
        candidates.sort(key=lambda pair: pair[1], reverse=True)
    else:
        candidates = candidates[: config.top_k]

    result.chunks = [
        RetrievedChunk(row.id, row.document_id, row.document.filename, row.page_number, row.chunk_text, score)
        for row, score in candidates
    ]
    return result
