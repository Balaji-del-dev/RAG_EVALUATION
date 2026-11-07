"""Answer formats: recognising the request, converting by rule, the retry when the model loops, and file downloads."""

import json
import shutil
from datetime import datetime
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import yaml
from rest_framework.test import APIClient

from apps.chatbot.services import downloads, formats, generation
from apps.chatbot.services.llm_gateway import LLMResponse
from apps.retrieval.services.hybrid_search import RetrievedChunk

ALL_PHRASES = [(fmt.key, phrase) for fmt in formats.FORMATS for phrase in fmt.phrases]


@pytest.mark.parametrize("key,phrase", ALL_PHRASES)
def test_every_phrase_is_recognised_with_a_cue(key, phrase):
    fmt, rest = formats.detect(f"give the leave policy as a {phrase}")
    assert fmt is not None and fmt.key == key
    assert "leave policy" in rest


@pytest.mark.parametrize("key,phrase", [(fmt.key, phrase) for fmt in formats.FORMATS for phrase in fmt.free])
def test_phrases_that_mean_only_a_layout_need_no_cue(key, phrase):
    assert formats.detect(f"leave policy {phrase}")[0].key == key


@pytest.mark.parametrize(
    "question",
    [
        "What is on the dining table?",
        "summarise this pdf",
        "what is json?",
        "explain the flowchart in the document",
        "what is the code of conduct?",
        "List the GenAI questions",
        "what are the steps to apply for leave",
        "what is the timeline of the project",
        "is excel installed on company laptops",
        "how many points does the appraisal have",
    ],
)
def test_a_word_used_for_its_meaning_is_not_a_format(question):
    assert formats.detect(question) == (None, question)


def test_the_most_specific_phrase_wins():
    assert formats.detect("show the leave types as a comparison table")[0].key == "comparison_table"
    assert formats.detect("give it as an html table")[0].key == "html_table"
    assert formats.detect("draw a decision tree for approvals")[0].key == "decision_tree"


def test_a_message_that_is_only_a_format_request_leaves_nothing_to_search():
    assert formats.detect("download this as pdf") == (formats.BY_KEY["pdf"], "")


TABLE = "| Leave | Days | Source |\n|---|---|---|\n| Earned | 18 | [S1] |\n| Sick | 12 | [S2] |"


def test_a_table_becomes_json_with_its_sources():
    data = json.loads(formats.convert(TABLE, "json").split("\n", 1)[1].rsplit("```", 1)[0])
    assert data == [
        {"Leave": "Earned", "Days": "18", "sources": ["S1"]},
        {"Leave": "Sick", "Days": "12", "sources": ["S2"]},
    ]


def test_a_table_becomes_yaml_xml_and_csv():
    assert yaml.safe_load(formats.convert(TABLE, "yaml")[8:-4])[1]["Days"] == "12"
    root = ElementTree.fromstring(formats.convert(TABLE, "xml")[7:-4])
    assert [item.findtext("Leave") for item in root] == ["Earned", "Sick"]
    assert formats.to_csv(TABLE).splitlines() == ["Leave,Days,Source", "Earned,18,", "Sick,12,"]


def test_a_list_and_plain_text_become_json_too():
    assert formats.structured("- Lock screens [S1]\n- Report phishing [S2]") == [
        {"text": "Lock screens", "sources": ["S1"]},
        {"text": "Report phishing", "sources": ["S2"]},
    ]
    assert formats.structured("Earned leave is 18 days [S1].") == {
        "answer": "Earned leave is 18 days .",
        "sources": ["S1"],
    }


def test_a_model_stuck_in_a_loop_is_detected():
    assert formats.is_degenerate("```html\n" * 40)  # the real reply
    assert not formats.is_degenerate(TABLE)
    assert not formats.is_degenerate("- one\n- two\n- three\n- four\n- five\n- six")


def test_a_longer_format_gets_a_smaller_source_budget():
    assert generation.context_budget("") == generation.MAX_CONTEXT_TOKENS  # what evaluation runs use
    assert generation.context_budget("table") < generation.MAX_CONTEXT_TOKENS
    assert generation.context_budget("table") >= generation.MIN_CONTEXT_TOKENS


class SequenceLLM:
    """Gives each reply in turn and records what it was asked."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return LLMResponse(self.replies.pop(0), "fake", 10, 5, 1.0)


def _generate(monkeypatch, layout, *replies):
    fake = SequenceLLM(*replies)
    monkeypatch.setattr(generation.llm_gateway, "complete", fake.complete)
    config = SimpleNamespace(prompt_version="v3", llm_model="fake", temperature=0.0, max_tokens=256)
    chunk = RetrievedChunk(1, 1, "hr.docx", 1, "Earned leave is 18 days. Sick leave is 12 days.", 1.0)
    return generation.generate("leave types", [chunk], config, layout=layout), fake


def test_a_looping_answer_is_asked_again_as_plain_markdown(monkeypatch):
    answer, fake = _generate(monkeypatch, "html", "```html\n" * 40, "Earned leave is 18 days [S1].")
    assert answer.text == "Earned leave is 18 days [S1]." and not answer.abstained
    assert "Format:" in fake.calls[0][0][1]["content"] and "Format:" not in fake.calls[1][0][1]["content"]
    assert answer.llm.prompt_tokens == 20  # both calls are counted


def test_a_loop_without_a_format_is_withheld(monkeypatch):
    answer, _ = _generate(monkeypatch, "", "Leave\n" * 12)
    assert answer.abstained and answer.degenerate


def test_json_is_converted_by_rule_from_the_models_table(monkeypatch):
    answer, fake = _generate(monkeypatch, "json", TABLE)
    assert answer.text.startswith("```json\n[")
    assert json.loads(answer.text[8:-4])[0]["Days"] == "18"
    # [S2] is not a source it was given: kept without a chunk, so citation accuracy counts it against the model.
    assert [(citation["marker"], bool(citation["chunk_id"])) for citation in answer.citations] == [
        ("S1", True),
        ("S2", False),
    ]
    assert fake.calls[0][1]["max_tokens"] == 512  # a table needs more room than the default 256


def _message(answer=TABLE, answer_format=""):
    return SimpleNamespace(
        question="Leave types at VVDN?",
        answer=answer,
        answer_format=answer_format,
        citations=[{"marker": "S1", "chunk_id": 1, "filename": "hr.docx", "page_number": 2}],
        created_at=datetime(2026, 11, 7, 9, 30),
    )


@pytest.mark.parametrize("extension", ["md", "txt", "html", "csv", "json", "yaml", "xml", "pptx"])
def test_plain_downloads(extension):
    content, content_type, name = downloads.build(_message(), extension)
    assert name == f"leave-types-at-vvdn.{extension}" and content
    if extension == "pptx":
        assert content[:2] == b"PK"
    elif extension == "csv":
        assert content.startswith(b"\xef\xbb\xbfLeave,Days,Source")
    elif extension == "json":
        assert json.loads(content)[0]["Leave"] == "Earned"
    elif extension == "html":
        assert b"<table>" in content and b"hr.docx, page 2" in content


def test_html_in_an_answer_is_shown_as_text_not_run():
    content, _, _ = downloads.build(_message("Hello <script>alert(1)</script>"), "html")
    assert b"<script>" not in content and b"&lt;script&gt;" in content


@pytest.mark.skipif(not shutil.which("soffice"), reason="LibreOffice is not installed")
@pytest.mark.parametrize(
    "extension,start",
    [("pdf", b"%PDF"), ("docx", b"PK"), ("odt", b"PK"), ("rtf", b"{\\rtf"), ("xlsx", b"PK"), ("ods", b"PK")],
)
def test_office_downloads(extension, start):
    content, _, _ = downloads.build(_message(), extension)
    assert content.startswith(start)


@pytest.mark.django_db
def test_only_the_asker_can_download_an_answer(django_user_model):
    from apps.chatbot.models import ChatMessage, ChatSession

    owner = django_user_model.objects.create_user("owner", password="pw-owner-123")
    other = django_user_model.objects.create_user("other", password="pw-other-123")
    session = ChatSession.objects.create(user=owner, title="Leave")
    message = ChatMessage.objects.create(session=session, question="Leave types?", answer=TABLE)
    url = f"/api/rag/messages/{message.id}/download/"
    client = APIClient()
    client.force_authenticate(owner)
    page = client.get(url + "html/?inline=1")
    assert page.status_code == 200 and page["Content-Disposition"].startswith("inline")
    assert "default-src 'none'" in page["Content-Security-Policy"]
    assert client.get(url + "csv/")["Content-Disposition"] == 'attachment; filename="leave-types.csv"'
    assert client.get(url + "exe/").status_code == 404
    client.force_authenticate(other)
    assert client.get(url + "csv/").status_code == 404


STEPS = "1. Submit the request in the HR portal [S1]\n2. Manager approves [S1]\n3. HR records it [S2]"
NESTED = "- IT policy\n  - Devices\n    - Laptops\n    - USB drives (blocked)\n  - Network\n    - VPN: required"
EVENTS = "- **2026-01** - Policy written [S1]\n- **2026-04** - Policy approved [S1]"
NUMBERS = "| Leave | Days |\n|---|---|\n| Earned [S1] | 18 |\n| Sick [S1] | 12 |\n| Casual | 6 days |"
FLOWS = "| From | To | Amount |\n|---|---|---|\n| Budget | Travel | 40 |\n| Budget | Training | 60 |"
SERIES_TABLE = "| Measure | Plan A | Plan B |\n|---|---|---|\n| Cost | 3 | 5 |\n| Speed | 4 | 2 |\n| Support | 5 | 3 |"
DRAWINGS = [
    ("steps", STEPS, ["flowchart TD", 's1["Submit the request in the HR portal"]', "s2 --> s3"]),
    ("mindmap", NESTED, ["mindmap", "root((IT policy))", "      USB drives blocked", "      VPN required"]),
    ("timeline", EVENTS, ["timeline", "  2026-01 : Policy written"]),
    ("pie", NUMBERS, ["pie showData", '"Earned" : 18', '"Casual" : 6']),
    ("bar", NUMBERS, ["xychart-beta", 'x-axis ["Earned", "Sick", "Casual"]', 'y-axis "Days"', "bar [18, 12, 6]"]),
    ("line", NUMBERS, ["line [18, 12, 6]"]),
    ("sankey", FLOWS, ["sankey-beta", "Budget,Travel,40"]),
    ("radar", SERIES_TABLE, ["radar-beta", 'curve c1["Plan A"]{3, 4, 5}']),
]


@pytest.mark.parametrize("kind,text,expected", DRAWINGS)
def test_a_diagram_is_drawn_by_rule_from_simple_markdown(kind, text, expected):
    drawn = formats.draw(text, kind, "leave days")
    code = drawn.split("```mermaid\n", 1)[1].split("\n```", 1)[0]
    for part in expected:
        assert part in code, (part, code)
    assert drawn.endswith(text)  # the steps, list or table stay below, readable without the picture


def test_without_the_structure_the_models_own_diagram_is_kept_and_fenced():
    """Real reply: a mind map without its ```mermaid fence showed as plain text."""
    assert (
        formats.draw("mindmap\n  root((IT))\n    Devices", "mindmap")
        == "```mermaid\nmindmap\n  root((IT))\n    Devices\n```"
    )
    assert formats.draw("No list here.", "steps") == "No list here."


@pytest.mark.parametrize(
    "question,title",
    [("show the vvdn leave types", "Vvdn leave types"), ("of the vvdn it policy", "Vvdn it policy"), ("", "Answer")],
)
def test_diagram_title(question, title):
    """Real titles were "Show the vvdn leave types" and "Of the vvdn it policy"."""
    assert formats.diagram_title(question) == title


def test_a_flowchart_request_asks_the_model_for_steps_not_mermaid():
    assert formats.BY_KEY["flowchart"].render == "drawn"
    assert "numbered list of short steps" in formats.BY_KEY["flowchart"].instruction
    assert formats.BY_KEY["sequence"].render == "mermaid"  # no simple Markdown maps onto a sequence diagram
