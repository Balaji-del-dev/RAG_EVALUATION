"""Parsing and chunking, using real generated PDF, DOCX and CSV files."""

import io
from types import SimpleNamespace

import pytest

from apps.common.exceptions import InvalidDocument
from apps.common.utils import count_tokens
from apps.documents.services.chunking import chunk_pages, chunk_recursive, chunk_semantic, chunk_structure
from apps.documents.services.parser import ParsedPage, clean_text, parse_document


def make_pdf(pages: list[str]) -> bytes:
    import pymupdf

    with pymupdf.open() as pdf:
        for text in pages:
            pdf.new_page().insert_text((72, 72), text)
        return pdf.tobytes()


def make_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading("Leave Policy", level=1)
    doc.add_paragraph("Employees get 24 days.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Grade"
    table.cell(0, 1).text = "L3"
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def test_pdf_keeps_page_numbers():
    pages = parse_document(make_pdf(["First page text", "Second page text"]), "a.pdf")
    assert [p.page_number for p in pages] == [1, 2]
    assert "Second page" in pages[1].content


def test_pdf_blank_pages_are_dropped():
    pages = parse_document(make_pdf(["Real text", ""]), "a.pdf")
    assert [p.page_number for p in pages] == [1]


def test_docx_keeps_headings_and_tables():
    content = parse_document(make_docx(), "a.docx")[0].content
    assert "# Leave Policy" in content
    assert "24 days" in content
    assert "Grade | L3" in content


def test_csv_rows_become_labelled_text():
    pages = parse_document(b"name,tier\nHalden,Premium\n", "a.csv")
    assert pages[0].content == "name: Halden; tier: Premium"


def test_csv_handles_byte_order_mark():
    assert parse_document(b"\xef\xbb\xbfname\nHalden\n", "a.csv")[0].content == "name: Halden"


def test_unsupported_type_is_rejected():
    with pytest.raises(InvalidDocument, match="Unsupported"):
        parse_document(b"x", "a.exe")


def test_corrupt_pdf_gives_a_clear_error():
    with pytest.raises(InvalidDocument, match="Could not parse"):
        parse_document(b"this is not a pdf", "a.pdf")


def test_empty_text_is_rejected():
    with pytest.raises(InvalidDocument, match="No text"):
        parse_document(b"   \n ", "a.txt")


def test_clean_text():
    assert clean_text("a  b\x00\r\n\r\n\r\n\r\nc") == "a b\n\nc"
    assert clean_text("inter-\nnational") == "international"


LONG_TEXT = " ".join(f"Sentence number {i} talks about a separate subject." for i in range(60))


def test_recursive_respects_size_and_overlaps():
    drafts = chunk_recursive([ParsedPage(3, LONG_TEXT)], size=50, overlap=10)
    assert len(drafts) > 1
    assert all(count_tokens(d.text) <= 50 for d in drafts)
    assert all(d.page_number == 3 for d in drafts)
    # Overlap: the end of one chunk reappears at the start of the next.
    assert drafts[0].text.split()[-1] in drafts[1].text


def test_structure_splits_at_headings_and_repeats_them():
    page = ParsedPage(1, "# Leave\n\n" + LONG_TEXT + "\n\n# Travel\n\nEconomy class only.")
    drafts = chunk_structure([page], size=50, overlap=5)
    assert {d.metadata["section"] for d in drafts} == {"Leave", "Travel"}
    assert all("Leave" in d.text for d in drafts if d.metadata["section"] == "Leave")
    travel = [d for d in drafts if d.metadata["section"] == "Travel"]
    assert len(travel) == 1
    assert "Economy" in travel[0].text


def test_structure_does_not_lose_text_before_first_heading():
    drafts = chunk_structure([ParsedPage(1, "Intro words here.\n\n# Section\n\nBody text.")], 50, 5)
    assert "Intro words here." in drafts[0].text


def fake_embed(sentences):
    """Two topics with orthogonal vectors: a real break exists exactly where the topic changes."""
    return [[1.0, 0.0] if "leave" in s else [0.0, 1.0] for s in sentences]


def test_semantic_breaks_where_the_topic_changes():
    text = "Annual leave is 24 days. Sick leave is 12 days. Parental leave is 16 weeks. Backups run nightly. Backups last 35 days."
    drafts = chunk_semantic([ParsedPage(1, text)], size=100, embed_fn=fake_embed)
    assert len(drafts) >= 2
    assert all(not ("leave" in d.text and "Backups" in d.text) for d in drafts)


def test_semantic_respects_the_size_limit():
    same_topic = " ".join(f"Annual leave rule number {i} applies here." for i in range(40))
    drafts = chunk_semantic([ParsedPage(1, same_topic)], size=30, embed_fn=lambda s: [[1.0, 0.0] for _ in s])
    assert len(drafts) > 1
    assert all(count_tokens(d.text) <= 30 for d in drafts)


def test_chunk_pages_dispatches_on_strategy():
    page = [ParsedPage(1, "# A\n\nSome words.")]
    config = SimpleNamespace(chunk_strategy="structure", chunk_size=50, chunk_overlap=5)
    assert chunk_pages(page, config)[0].metadata["section"] == "A"
