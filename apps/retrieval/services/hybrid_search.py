"""The retrieval pipeline: BM25 + vector -> RRF -> permission re-check -> rerank -> MMR -> top K."""

import contextvars
import copy
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from apps.common.exceptions import ServiceUnavailable
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
    # What happened that the reader should know: a fallback, a widened search.
    notes: list[str] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[int]:
        return [chunk.chunk_id for chunk in self.chunks]


# A table read from a PDF comes out as all its labels, then all its values, sometimes split between two chunks:
# "Days In Month (A) / Arrear Days (B) / LOPR Days(C) / LOP Days (D) / Net Days Worked" and then "31.0 / 0.0 / 0.0 / 0.0
# / 31.0". Given the labels without the values, the model reported a rupee amount from elsewhere as the LOP days.
MIN_TABLE_ROWS = 3
MAX_LABEL_LENGTH = 40
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?")


def _is_number(line: str) -> bool:
    return bool(NUMBER_RE.fullmatch(line.strip()))


def _is_label(line: str) -> bool:
    text = line.strip()
    return bool(text) and len(text) <= MAX_LABEL_LENGTH and not re.search(r"\d", text)


def pair_table_values(text: str) -> str:
    """Rejoin a run of labels with the run of values that follows it, when the counts match: "LOP Days (D): 0.0".
    A label line starting with "(" continues the one before, as in "Net Days Worked / (E = A + B + C - D)"."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        labels, j = [], i
        while j < len(lines) and _is_label(lines[j]):
            if lines[j].strip().startswith("(") and labels:
                labels[-1] += " " + lines[j].strip()
            else:
                labels.append(lines[j].strip())
            j += 1
        values, k = [], j
        while k < len(lines) and _is_number(lines[k]):
            values.append(lines[k].strip())
            k += 1
        # The values must visibly end. A run cut off by the end of a chunk may be missing some, and pairing what
        # is left would put numbers against the wrong labels.
        if len(labels) >= MIN_TABLE_ROWS and len(values) == len(labels) and k < len(lines):
            out.extend(f"{label}: {value}" for label, value in zip(labels, values, strict=True))
            i = k
        else:
            # The whole run of labels, or nothing: retrying from inside it paired "Basic Salary" with a year-to-date
            # figure once the remaining count happened to match.
            out.extend(lines[i : max(j, i + 1)])
            i = max(j, i + 1)
    return "\n".join(out)


def join_split_table(text: str, next_text: str) -> str:
    """A chunk ending with a table's labels gets the next chunk's values, without the overlap the two share."""
    first, second = text.split("\n"), next_text.split("\n")
    if len(first) < MIN_TABLE_ROWS or not all(_is_label(line) for line in first[-MIN_TABLE_ROWS:]):
        return text
    overlap = next((k for k in range(min(len(first), len(second)), 0, -1) if first[-k:] == second[:k]), 0)
    rest = second[overlap:]
    first_value = next((line for line in rest if not _is_label(line)), "")
    return "\n".join(first + rest) if _is_number(first_value) else text


def without_overlap(first: str, second: str) -> str:
    """The second chunk's text without the lines it repeats from the end of the first."""
    a, b = first.split("\n"), second.split("\n")
    overlap = next((k for k in range(min(len(a), len(b)), 0, -1) if a[-k:] == b[:k]), 0)
    return "\n".join(b[overlap:])


def text_with_tables(row: DocumentChunk, index_key: str) -> str:
    """The chunk's text, with a table split from the next chunk joined back and its labels paired with values."""
    text = row.chunk_text
    lines = text.split("\n")
    if len(lines) >= MIN_TABLE_ROWS and all(_is_label(line) for line in lines[-MIN_TABLE_ROWS:]):
        following = (
            DocumentChunk.objects.filter(
                document_id=row.document_id, index_key=index_key, chunk_index=row.chunk_index + 1
            )
            .values_list("chunk_text", flat=True)
            .first()
        )
        if following:
            text = join_split_table(text, following)
    return pair_table_values(text)


logger = logging.getLogger(__name__)
# A cross-encoder score below this means even the best match is weak. Good matches here score 3 to 10.
WEAK_MATCH = 0.0
RETRY_DELAY_SECONDS = 0.5


def _timed(result: RetrievalResult, stage: str, started: float) -> None:
    result.timings_ms[stage] = round((time.perf_counter() - started) * 1000, 2)


def retrieve(
    query: str, config: RagConfig, allowed_document_ids: list[int], widen_if_weak: bool = False
) -> RetrievalResult:
    """The best chunks the caller may read. With widen_if_weak (the chat sets it; evaluation runs do not, so their
    scores stay comparable), a search whose best match scores below WEAK_MATCH runs again with twice the candidates,
    and the better of the two is kept."""
    result = _retrieve_once(query, config, allowed_document_ids)
    best = _best_score(result)
    if not widen_if_weak or best is None or best >= WEAK_MATCH:
        return result
    wider = copy.copy(config)
    wider.candidate_k, wider.rerank_top_n = config.candidate_k * 2, config.rerank_top_n * 2
    second = _retrieve_once(query, wider, allowed_document_ids)
    if (_best_score(second) or best) > best:
        second.notes.append(f"The best match was weak, so the search was widened to {wider.candidate_k} candidates.")
        return second
    return result


def _best_score(result: "RetrievalResult") -> float | None:
    ranked = result.stages.get("rerank") or []
    return ranked[0]["score"] if ranked else None


def _with_one_retry(search):
    """Search indexes can drop a connection now and then; one quick retry covers that."""
    try:
        return search()
    except ServiceUnavailable:
        time.sleep(RETRY_DELAY_SECONDS)
        return search()


# Two threads: one search at a time per web worker thread is the normal load, and embedding is CPU-bound anyway.
_VECTOR_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vector-search")


def _vector_hits(query: str, config: RagConfig, allowed: list[int]):
    """The query's embedding, its vector-search hits, and how long each step took."""
    timings = {}
    started = time.perf_counter()
    with span("retrieval.embed"):
        query_vector = embeddings.embed_query(query, config.embedding_model)
    timings["embed"] = round((time.perf_counter() - started) * 1000, 2)
    with span("retrieval.vector"):
        hits = _with_one_retry(lambda: vector_search.search(config, query_vector, config.candidate_k, allowed))
    timings["vector"] = round((time.perf_counter() - started) * 1000, 2)
    return query_vector, hits, timings


def _retrieve_once(query: str, config: RagConfig, allowed_document_ids: list[int]) -> "RetrievalResult":
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
    hybrid = config.retrieval_mode == RagConfig.Mode.HYBRID
    use_vector = config.retrieval_mode in (RagConfig.Mode.VECTOR, RagConfig.Mode.HYBRID)
    use_bm25 = config.retrieval_mode in (RagConfig.Mode.BM25, RagConfig.Mode.HYBRID)

    # In a hybrid search the two engines run at the same time: the embedding and Qdrant call in a worker thread,
    # the keyword search here. The keyword search stays in this thread because it may read the database, and a
    # thread would open a connection of its own that nothing closes.
    vector_job = None
    if use_vector:
        context = contextvars.copy_context()  # keeps the request ID and trace parent in the worker's spans
        vector_job = _VECTOR_POOL.submit(context.run, _vector_hits, query, config, sorted(allowed))

    bm25_hits, bm25_error = [], None
    if use_bm25:
        started = time.perf_counter()
        try:
            with span("retrieval.bm25"):
                bm25_hits = _with_one_retry(
                    lambda: bm25_search.backend().search(config, query, config.candidate_k, sorted(allowed))
                )
        except ServiceUnavailable as exc:
            bm25_error = exc
        _timed(result, "bm25", started)

    if vector_job is not None:
        try:
            query_vector, hits, timings = vector_job.result()
            result.timings_ms.update(timings)
        except ServiceUnavailable as exc:
            # In a hybrid search, keywords alone still answer. A vector-only search has nothing to fall back to.
            if not hybrid or bm25_error:
                raise
            logger.warning("Vector search failed, continuing with keywords only: %s", exc)
            result.notes.append("The vector index did not answer, so this used keyword search only.")
            hits = []
        # Listed first, as before, so that fusion breaks ties the same way.
        ranked_lists.append([hit.chunk_id for hit in hits])
        result.stages["vector"] = [{"chunk_id": hit.chunk_id, "score": round(hit.score, 4)} for hit in hits]

    if use_bm25:
        if bm25_error:
            if not hybrid:
                raise bm25_error  # nothing left to search with
            logger.warning("Keyword search failed, continuing with vectors only: %s", bm25_error)
            result.notes.append("The keyword index did not answer, so this used vector search only.")
        ranked_lists.append([hit.chunk_id for hit in bm25_hits])
        result.stages["bm25"] = [{"chunk_id": hit.chunk_id, "score": round(hit.score, 4)} for hit in bm25_hits]

    fused = rrf.reciprocal_rank_fusion(ranked_lists, config.rrf_k)
    result.stages["rrf"] = [{"chunk_id": chunk_id, "score": round(score, 5)} for chunk_id, score in fused]
    if not fused:
        return result

    # Second enforcement point: load only rows whose document is still allowed and still indexed.
    rows = DocumentChunk.objects.filter(
        id__in=[chunk_id for chunk_id, _ in fused], index_key=config.index_key, document_id__in=allowed
    ).select_related("document")
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
        RetrievedChunk(
            row.id,
            row.document_id,
            row.document.filename,
            row.page_number,
            text_with_tables(row, config.index_key),
            score,
        )
        for row, score in candidates
    ]
    return result
