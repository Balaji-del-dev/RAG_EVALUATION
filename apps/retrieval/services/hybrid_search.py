"""The retrieval pipeline: BM25 + vector -> RRF -> permission re-check -> rerank -> MMR -> top K."""

import re
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
