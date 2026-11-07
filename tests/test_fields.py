"""Exact figures across documents, read by rule: the user's "exact salary data ... in table format"."""

import pytest

from apps.chatbot.services import fields
from apps.documents.models import Document, DocumentChunk
from apps.retrieval.services.hybrid_search import RetrievedChunk

pytestmark = pytest.mark.django_db

# The real payslip layout: the address block first, the figures in later chunks, labels before values.
PAGES = {
    "Feb": [
        "PAYSLIP FOR THE MONTH OF FEBRUARY 2026\nEmp Name\nBalamurugan Thangadurai",
        "Total Earnings\n66,000.00\n66,000.00 Total Deductions\n2,366.00\n2,366.00\nNet Pay : Rs. 63,634.00",
        "Days In Month (A)\nArrear Days (B)\nLOPR Days(C)\nLOP Days (D)\nNet Days Worked\n28.0\n3.0\n0.0\n0.0\n31.0\nPrivate",
    ],
    "May": [
        "PAYSLIP FOR THE MONTH OF MAY 2026\nEmp Name\nBalamurugan Thangadurai",
        "Total Earnings\n60,093.00\n141,759.00 Total Deductions\n2,534.00\n5,634.00\nNet Pay : Rs. 57,559.00",
        "Days In Month (A)\nArrear Days (B)\nLOPR Days(C)\nLOP Days (D)\nNet Days Worked\n31.0\n0.0\n0.0\n0.0\n31.0\nPrivate",
    ],
}


@pytest.fixture
def payslips():
    documents = {}
    for month, chunks in PAGES.items():
        document = Document.objects.create(
            filename=f"Pay Slip_{month}_2026.pdf", file_path=month, sha256=month, status="ready"
        )
        for index, text in enumerate(chunks):
            DocumentChunk.objects.create(
                document=document, index_key="k", chunk_index=index, chunk_text=text, token_count=9
            )
        documents[month] = document
    policy = Document.objects.create(filename="hr_policy.docx", file_path="p", sha256="p", status="ready")
    DocumentChunk.objects.create(
        document=policy, index_key="k", chunk_index=0, chunk_text="Salary is paid monthly.", token_count=4
    )
    documents["policy"] = policy
    return documents


def _found(documents, *names):
    """What the search found: the address chunk of each payslip named, as in the real failure."""
    chunks = []
    for name in names:
        row = DocumentChunk.objects.filter(document=documents[name]).order_by("chunk_index").first()
        chunks.append(RetrievedChunk(row.id, row.document_id, row.document.filename, 1, row.chunk_text, -1.2))
    return chunks


def test_salary_across_payslips_is_read_exactly(payslips):
    ids = [document.id for document in payslips.values()]
    table = fields.field_table(
        "exact salary data of balamurugan thangadurai in table", _found(payslips, "May", "Feb"), ids, "k"
    )
    assert table.markdown.splitlines() == [
        "| Document | Net pay | Total earnings | Total deductions | Source |",
        "|---|---|---|---|---|",
        "| Feb 2026 | 63,634.00 | 66,000.00 | 2,366.00 | [S1] |",
        "| May 2026 | 57,559.00 | 60,093.00 | 2,534.00 | [S2] |",
    ]
    assert [chunk.document_id for chunk in table.chunks] == [payslips["Feb"].id, payslips["May"].id]
    assert "Net Pay" in table.chunks[0].text  # the source shown is the chunk holding the figure


def test_lop_comes_from_the_paired_table_not_the_first_number(payslips):
    ids = [payslips["Feb"].id, payslips["May"].id]
    table = fields.field_table("lop days for each month", _found(payslips, "Feb", "May"), ids, "k")
    assert "| Feb 2026 | 0.0 |" in table.markdown and "| May 2026 | 0.0 |" in table.markdown


def test_one_document_is_left_to_the_normal_answer(payslips):
    assert fields.field_table("may salary", _found(payslips, "May"), [payslips["May"].id], "k") is None


def test_a_policy_question_that_found_the_policy_is_not_hijacked(payslips):
    found = _found(payslips, "policy") + _found(payslips, "policy")
    ids = [document.id for document in payslips.values()]
    assert fields.field_table("what is the salary policy", found, ids, "k") is None


def test_a_question_that_names_no_figure_is_not_touched(payslips):
    ids = [document.id for document in payslips.values()]
    assert fields.field_table("who is balamurugan", _found(payslips, "May", "Feb"), ids, "k") is None
