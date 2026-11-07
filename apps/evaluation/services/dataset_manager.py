"""Dataset import and gold-label resolution.

Gold labels are evidence passages, not chunk ids. Chunk ids change whenever a document is re-chunked, which
would make comparing chunking strategies impossible. Instead, a chunk counts as relevant if it contains
enough of a passage's key terms.
"""

import csv
import io
import json
from pathlib import PurePath

from django.conf import settings
from django.db import transaction

from apps.common.exceptions import InvalidDataset
from apps.documents.models import DocumentChunk
from apps.evaluation.models import EvaluationCase, EvaluationDataset
from apps.retrieval.services.bm25_search import analyze

# If no chunk holds a whole passage (it was split across a boundary), accept the best chunk above this share.
PARTIAL_MATCH_FLOOR = 0.5


def evidence_tokens(text: str) -> set[str]:
    return set(analyze(text))


def relevant_chunk_ids(evidence: list[str], index_key: str, document_ids: list[int] | None = None) -> set[int]:
    if not evidence:
        return set()
    rows = DocumentChunk.objects.filter(index_key=index_key)
    if document_ids is not None:
        rows = rows.filter(document_id__in=document_ids)
    chunks = [(chunk_id, evidence_tokens(text)) for chunk_id, text in rows.values_list("id", "chunk_text")]
    threshold = settings.EVALUATION["EVIDENCE_MATCH_THRESHOLD"]
    relevant: set[int] = set()
    for passage in evidence:
        terms = evidence_tokens(passage)
        if not terms:
            continue
        overlaps = [(chunk_id, len(terms & tokens) / len(terms)) for chunk_id, tokens in chunks]
        matched = {chunk_id for chunk_id, share in overlaps if share >= threshold}
        if not matched and overlaps:
            best_id, best_share = max(overlaps, key=lambda pair: pair[1])
            if best_share >= PARTIAL_MATCH_FLOOR:
                matched = {best_id}
        relevant |= matched
    return relevant


def _clean_case(raw: dict, position: int) -> dict:
    question = str(raw.get("question", "")).strip()
    if not question:
        raise InvalidDataset(f"Case {position}: the question is missing")
    answerable = raw.get("answerable", True)
    if isinstance(answerable, str):
        answerable = answerable.strip().lower() not in ("false", "0", "no")
    evidence = raw.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [part.strip() for part in evidence.split("||") if part.strip()]
    reference = str(raw.get("reference_answer", "")).strip()
    if answerable and not reference:
        raise InvalidDataset(f"Case {position}: an answerable question needs a reference_answer")
    return {"question": question, "reference_answer": reference, "answerable": bool(answerable), "evidence": evidence}


DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg"}


def parse_dataset_file(data: bytes, filename: str) -> list[dict]:
    """JSON: a list of cases, or an object with a "cases" list. CSV: question, reference_answer, answerable,
    evidence columns, with several evidence passages separated by ||.

    The type is checked before anything is read. A Word file dropped here used to fail with "'utf-8' codec can't
    decode byte 0x90", which says nothing about the actual mistake."""
    extension = PurePath(filename).suffix.lower()
    if extension in DOCUMENT_EXTENSIONS:
        raise InvalidDataset(
            f"{filename} is a document, not a dataset. To ask questions about it, upload it on the Documents page. "
            "A dataset is a list of test questions with their expected answers, in a .json or .csv file."
        )
    if extension not in {".json", ".csv"}:
        raise InvalidDataset(f"{filename} is not a dataset file. Upload a .json or .csv file.")
    if extension == ".json":
        try:
            parsed = json.loads(data.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise InvalidDataset(f"{filename} is not UTF-8 text. Save it as UTF-8 and upload it again.") from exc
        except json.JSONDecodeError as exc:
            raise InvalidDataset(
                f"{filename} is not valid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}."
            ) from exc
        if isinstance(parsed, dict):
            if not isinstance(parsed.get("cases"), list):
                raise InvalidDataset(f'{filename} is a JSON object without a "cases" list.')
            return parsed["cases"]
        if not isinstance(parsed, list):
            raise InvalidDataset(f"{filename} must hold a list of cases.")
        return parsed
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Excel on Windows saves "CSV" in the Windows code page, not UTF-8.
        text = data.decode("cp1252", errors="replace")
    try:
        return list(csv.DictReader(io.StringIO(text)))
    except csv.Error as exc:
        raise InvalidDataset(f"{filename} is not a readable CSV file: {exc}") from exc


@transaction.atomic
def create_dataset(*, name: str, version: str, description: str, cases: list[dict], user=None) -> EvaluationDataset:
    if not isinstance(cases, list) or not cases:
        raise InvalidDataset("A dataset needs at least one case")
    if EvaluationDataset.objects.filter(name=name, version=version).exists():
        raise InvalidDataset(
            f"Dataset '{name}' version {version} already exists. Datasets are immutable: use a new version."
        )
    cleaned = [_clean_case(raw, position) for position, raw in enumerate(cases, start=1)]
    dataset = EvaluationDataset.objects.create(name=name, version=version, description=description, created_by=user)
    EvaluationCase.objects.bulk_create(EvaluationCase(dataset=dataset, **case) for case in cleaned)
    return dataset
