"""Exact figures read from documents by rule, for questions that ask for the same figure across several of them:
"exact salary data of balamurugan thangadurai in table format".

Asked that, the search found all four payslips but only their first chunk, the address block, and the model
declined: the figures were in other chunks. Even with the right chunks, the small model misread payslip tables. So
when a question names a figure that several documents label the same way, each document is read whole and the
number after the label is taken, one row per document, each row citing its own document.
"""

import re
from dataclasses import dataclass

from apps.chatbot.services.generation import months_in
from apps.documents.models import DocumentChunk
from apps.retrieval.services.hybrid_search import RetrievedChunk, pair_table_values, without_overlap

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass(frozen=True)
class Field:
    label: str
    pattern: re.Pattern  # the label as it appears in the documents, followed by its number
    asked_by: tuple[str, ...]  # words in a question that ask for it


def _field(label: str, text_label: str, asked_by: tuple[str, ...]) -> Field:
    # The label, then separators only (a colon, "Rs.", a line break), then the number. Anything else in between
    # means the number belongs to something else, and the field is left out rather than guessed.
    pattern = re.compile(rf"{text_label}[\s:|/-]*(?:Rs\.?\s*)?(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
    return Field(label, pattern, asked_by)


FIELDS = [
    _field(
        "Net pay", r"\bnet\s+pay\b", ("salary", "net pay", "net salary", "take home", "take-home", "in hand", "in-hand")
    ),
    _field("Total earnings", r"\btotal\s+earnings\b", ("salary", "gross", "earnings", "ctc")),
    _field("Total deductions", r"\btotal\s+deductions\b", ("salary", "deductions", "deducted")),
    _field("LOP days", r"\blop\s+days\b(?:\s*\(d\))?", ("lop", "loss of pay", "leave without pay")),
]


@dataclass
class FieldTable:
    markdown: str
    chunks: list[RetrievedChunk]  # one per row, in row order, so that [S1] is the first row's document


def asked_fields(question: str) -> list[Field]:
    text = question.lower()
    return [field for field in FIELDS if any(re.search(rf"\b{re.escape(word)}\b", text) for word in field.asked_by)]


def _document_text(rows: list[DocumentChunk]) -> str:
    text = rows[0].chunk_text
    for previous, row in zip(rows, rows[1:], strict=False):
        text += "\n" + without_overlap(previous.chunk_text, row.chunk_text)
    return pair_table_values(text)


def _period(text: str, filename: str) -> tuple[tuple[int, int], str]:
    """(sort key, label) such as ((2026, 5), "May 2026"), from the document's opening or its file name."""
    head = text[:400]
    months = sorted(months_in(head)) or sorted(months_in(filename.replace("_", " ")))
    year = re.search(r"\b(20\d\d)\b", head) or re.search(r"(?:^|[^\d])(20\d\d|\d\d)(?:[^\d]|$)", filename)
    if months:
        full = int(year.group(1)) if year else 0
        full = full + 2000 if 0 < full < 100 else full
        return (full, months[0]), f"{MONTH_NAMES[months[0] - 1]} {full}" if full else MONTH_NAMES[months[0] - 1]
    return (9999, 99), filename.rsplit(".", 1)[0]


def field_table(
    question: str, retrieved: list[RetrievedChunk], document_ids: list[int], index_key: str
) -> FieldTable | None:
    """The asked-for figures from every document in scope that labels them, if the search itself mostly found such
    documents and there are at least two. One document's figures are left to the normal answer."""
    fields = asked_fields(question)
    if not fields or not retrieved:
        return None
    rows = list(
        DocumentChunk.objects.filter(index_key=index_key, document_id__in=document_ids)
        .select_related("document")
        .order_by("document_id", "chunk_index")
    )
    by_document: dict[int, list[DocumentChunk]] = {}
    for row in rows:
        by_document.setdefault(row.document_id, []).append(row)
    found = []
    for parts in by_document.values():
        text = _document_text(parts)
        values = {field.label: match.group(1) for field in fields if (match := field.pattern.search(text))}
        if values:
            # The source shown for the row is the chunk holding its first figure.
            first = min((field.pattern.search(text).start(), field) for field in fields if field.label in values)[1]
            cited = next((part for part in parts if first.pattern.search(pair_table_values(part.chunk_text))), parts[0])
            found.append((_period(text, parts[0].document.filename), values, cited))
    labelled = {cited.document_id for _, _, cited in found}
    if len(found) < 2 or sum(chunk.document_id in labelled for chunk in retrieved) * 2 < len(retrieved):
        return None
    found.sort(key=lambda item: item[0][0])
    columns = [field.label for field in fields if any(field.label in values for _, values, _ in found)]
    lines = ["| Document | " + " | ".join(columns) + " | Source |", "|" + "---|" * (len(columns) + 2)]
    chunks = []
    for number, ((_, period), values, cited) in enumerate(found, start=1):
        cells = [values.get(column, "not in this document") for column in columns]
        lines.append(f"| {period} | " + " | ".join(cells) + f" | [S{number}] |")
        chunks.append(
            RetrievedChunk(
                cited.id, cited.document_id, cited.document.filename, cited.page_number, cited.chunk_text, 1.0
            )
        )
    return FieldTable("\n".join(lines), chunks)
