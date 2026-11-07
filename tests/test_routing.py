"""Batch 5 of the production pipeline: questions routed away from the search, and answers written again when their
figures are in no source."""

from types import SimpleNamespace

import pytest
from rest_framework.test import APIClient

from apps.chatbot.models import ChatMessage, ChatSession
from apps.chatbot.services import chat_service, generation, rag_workflow, routing
from apps.chatbot.services.llm_gateway import LLMResponse
from apps.chatbot.services.rag_workflow import RagOutcome
from apps.common.constants import Role
from apps.documents.models import Document, DocumentChunk
from apps.retrieval.models import RagConfig
from apps.retrieval.services.hybrid_search import RetrievalResult, RetrievedChunk

HISTORY = [{"question": "How many earned leave days?", "answer": "18 days [S1]."}]


@pytest.mark.parametrize(
    "question,earlier,history,expected",
    [
        # The real questions that were declined.
        ("how many vvdn documents in databse pls ?", None, None, "library"),
        ("pls summery this thred history", "How many earned leave days?", HISTORY, "conversation"),
        ("pls properly summery this pls", None, None, "clarify"),
        # More ways of asking the same.
        ("list my documents", None, None, "library"),
        ("list documents", None, None, "library"),
        ("what documents do you have?", None, None, "library"),
        ("how many files are uploaded", None, None, "library"),
        ("summarise our conversation so far", "q", HISTORY, "conversation"),
        ("summarise this thread", None, [], "clarify"),  # nothing to summarise yet
        ("summary", None, None, "clarify"),
        ("tell me more", None, None, "clarify"),
        # Questions about the documents' contents still search.
        ("list the documents required for joining", None, None, "search"),
        ("which documents should I submit for reimbursement?", None, None, "search"),
        ("explain the document retention policy", None, None, "search"),
        ("how many days of annual leave", None, None, "search"),
        ("summarize the chat usage policy", None, None, "search"),
        ("summarize the employment history of balamurugan", None, None, "search"),
        ("explain leave policy", None, None, "search"),
        ("pls properly summery this pls", "Summarise the HR policy", HISTORY, "search"),  # a follow-up
    ],
)
def test_where_a_chat_question_goes(question, earlier, history, expected):
    assert routing.route(question, earlier, history) == expected


@pytest.fixture
def library(db, django_user_model):
    owner = django_user_model.objects.create_user("owner", password="pw-o-12345", role=Role.ANALYST)
    made = {
        name: Document.objects.create(filename=name, file_path=name, sha256=name, status="ready", page_count=pages)
        for name, pages in [("VVDN_HR_Policy.docx", 14), ("VVDN_IT_Policy.pdf", 6), ("Tenarai_Brochure.pdf", 2)]
    }
    made["secret"] = Document.objects.create(
        filename="VVDN_Salaries.xlsx", file_path="s", sha256="s", status="ready", visibility="restricted", owner=owner
    )
    return made


def test_library_questions_are_answered_from_the_document_list(library):
    readable = [library[name].id for name in ("VVDN_HR_Policy.docx", "VVDN_IT_Policy.pdf", "Tenarai_Brochure.pdf")]
    answer = routing.library_answer("how many vvdn documents in databse pls ?", readable)
    assert answer.startswith('You can read 3 documents. 2 have "vvdn" in the name:')
    assert "| VVDN_HR_Policy.docx | Word | 14 |" in answer and "Tenarai" not in answer
    assert "Salaries" not in answer  # only documents the asker may read
    none = routing.library_answer("how many payroll files are in the system", readable)
    assert 'none has "payroll" in its name' in none and "Tenarai_Brochure.pdf" in none
    assert routing.library_answer("list documents", []).startswith("You have no documents")


def _config():
    return SimpleNamespace(
        name="c", index_key="k", llm_model="fake", prompt_version="v3", temperature=0.0, max_tokens=64
    )


@pytest.fixture
def fake_llm(monkeypatch):
    calls = SimpleNamespace(messages=[], replies=[])

    def complete(messages, **kwargs):
        calls.messages.append(messages)
        return LLMResponse(calls.replies.pop(0), "fake", 100, 20, 1000.0)

    monkeypatch.setattr(generation.llm_gateway, "complete", complete)
    return calls


@pytest.mark.django_db
def test_routes_are_chat_only_and_skip_the_search(library, fake_llm, monkeypatch):
    searched = []
    monkeypatch.setattr(rag_workflow, "retrieve", lambda *args, **kwargs: searched.append(1) or RetrievalResult())
    ids = [document.id for document in library.values() if document.visibility != "restricted"]
    chat = rag_workflow.run_rag("how many vvdn documents in databse pls ?", _config(), ids, interactive=True)
    assert chat.route == "library" and chat.model == "" and searched == [] and "| # | Document |" in chat.answer
    # Evaluation runs always search, so their scores stay comparable.
    run = rag_workflow.run_rag("how many vvdn documents in databse pls ?", _config(), ids)
    assert run.route == "search" and searched == [1]
    vague = rag_workflow.run_rag("pls properly summery this pls", _config(), ids, interactive=True)
    assert vague.route == "clarify" and "Tenarai Brochure" in vague.answer and not vague.abstained


@pytest.mark.django_db
def test_a_conversation_is_summarised_from_its_own_answers(fake_llm, django_user_model):
    user = django_user_model.objects.create_user("asker", password="pw-x-12345", role=Role.VIEWER)
    config = RagConfig.objects.create(name="Route config", is_default=True)
    session = ChatSession.objects.create(user=user, title="t", rag_config=config)
    ChatMessage.objects.create(session=session, question="earned leave?", answer="18 days [S1].")
    ChatMessage.objects.create(session=session, question="secret", answer="", blocked=True)
    ChatMessage.objects.create(
        session=session, question="and sick leave?", standalone_question="How many sick leave days?", answer="12 days."
    )
    fake_llm.replies = ["- Earned leave: 18 days.\n- Sick leave: 12 days. Mail priya.raman@example.com."]
    message, outcome = chat_service.ask(user, "pls summery this thred history", config, session)
    prompt = fake_llm.messages[0][-1]["content"]
    assert "Question 1: earned leave?\nAnswer 1: 18 days." in prompt and "[S1]" not in prompt
    assert "Question 2: How many sick leave days?" in prompt and "secret" not in prompt
    assert outcome.route == "conversation" and message.route == "conversation"
    assert "priya.raman@example.com" not in message.answer  # the summary passes the output guard too


def _search(monkeypatch, text):
    chunk = RetrievedChunk(1, 1, "hr.docx", 1, text, 2.0)
    monkeypatch.setattr(
        rag_workflow,
        "retrieve",
        lambda *args, **kwargs: RetrievalResult(chunks=[chunk], stages={"rerank": [{"chunk_id": 1, "score": 2.0}]}),
    )


@pytest.fixture
def leave_document(db):
    document = Document.objects.create(filename="hr.docx", file_path="h", sha256="h", status="ready")
    DocumentChunk.objects.create(
        document=document, index_key="k", chunk_index=0, chunk_text="Sick leave is 12 days a year.", token_count=7
    )
    return document


def test_an_answer_with_a_figure_in_no_source_is_written_again(leave_document, fake_llm, monkeypatch):
    _search(monkeypatch, "Sick leave is 12 days a year.")
    fake_llm.replies = ["Sick leave is 10 days a year [S1].", "Sick leave is 12 days a year [S1]."]
    outcome = rag_workflow.run_rag("how many sick leave days?", _config(), [leave_document.id], interactive=True)
    assert outcome.answer == "Sick leave is 12 days a year [S1]."
    assert outcome.checks == {"regenerated": {"figures": ["10"], "fixed": True}}
    assert "10" in fake_llm.messages[1][-1]["content"]  # the model was told which figure was wrong
    assert (outcome.prompt_tokens, outcome.completion_tokens) == (200, 40)  # both calls are counted


def test_the_better_draft_is_kept(leave_document, fake_llm, monkeypatch):
    _search(monkeypatch, "Sick leave is 12 days a year.")
    fake_llm.replies = ["Sick leave is 10 days [S1].", "Sick leave is 10 or 11 days [S1]."]
    outcome = rag_workflow.run_rag("how many sick leave days?", _config(), [leave_document.id], interactive=True)
    assert outcome.answer == "Sick leave is 10 days [S1]."
    assert outcome.checks == {"numbers": ["10"], "regenerated": {"figures": ["10"], "fixed": False}}


def test_evaluation_runs_are_not_written_again(leave_document, fake_llm, monkeypatch):
    _search(monkeypatch, "Sick leave is 12 days a year.")
    fake_llm.replies = ["Sick leave is 10 days [S1]."]
    outcome = rag_workflow.run_rag("how many sick leave days?", _config(), [leave_document.id])
    assert outcome.checks == {"numbers": ["10"]} and len(fake_llm.messages) == 1


@pytest.mark.django_db
def test_a_rewrite_clears_the_streamed_draft(monkeypatch, django_user_model):
    user = django_user_model.objects.create_user("streamer", password="pw-s-12345", role=Role.VIEWER)
    RagConfig.objects.create(name="Stream config", is_default=True)

    def fake_stream_rag(question, *args, **kwargs):
        yield "token", "Sick leave is 10 days. "
        yield "reset", ""
        yield "token", "Sick leave is 12 days. "
        yield "outcome", RagOutcome(question=question, answer="Sick leave is 12 days.")

    monkeypatch.setattr(chat_service, "stream_rag", fake_stream_rag)
    client = APIClient()
    client.force_authenticate(user)
    response = client.post(
        "/api/rag/query/stream/", {"question": "sick leave days?"}, format="json", HTTP_ACCEPT="text/event-stream"
    )
    names = [
        block.split("\n", 1)[0].removeprefix("event: ")
        for block in b"".join(response.streaming_content).decode().strip().split("\n\n")
    ]
    assert names == ["text", "reset", "text", "done"]
