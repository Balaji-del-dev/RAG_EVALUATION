"""API behaviour: auth, roles, document permissions, validation, and the reports."""

import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone
from rest_framework.test import APIClient

from apps.common.constants import JobStatus, Role
from apps.common.exceptions import ServiceUnavailable
from apps.documents.models import Document
from apps.evaluation.models import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationResult,
    EvaluationRun,
    MetricDefinition,
    MetricScore,
)
from apps.evaluation.services.run_executor import summarize, sync_metric_definitions
from apps.retrieval.models import RagConfig

pytestmark = pytest.mark.django_db


def test_a_new_frontend_build_is_served_without_a_restart(settings, tmp_path, client):
    """A cached page would keep pointing at bundle files the build has deleted, which shows a blank page."""
    settings.TEMPLATES = [{**settings.TEMPLATES[0], "DIRS": [tmp_path]}]
    page = tmp_path / "index.html"
    page.write_text("first build")
    assert b"first build" in client.get("/chat").content

    page.write_text("second build")
    response = client.get("/chat")
    assert b"second build" in response.content
    assert "no-cache" in response["Cache-Control"]


@pytest.fixture
def people(django_user_model):
    make = django_user_model.objects.create_user
    return SimpleNamespace(
        admin=make("a", password="pw-a-12345", role=Role.ADMIN),
        analyst=make("b", password="pw-b-12345", role=Role.ANALYST),
        viewer=make("c", password="pw-c-12345", role=Role.VIEWER),
        other=make("d", password="pw-d-12345", role=Role.VIEWER),
    )


def client_for(user=None):
    client = APIClient()
    if user:
        client.force_authenticate(user)
    return client


@pytest.fixture
def config():
    return RagConfig.objects.create(name="Test config", is_default=True)


@pytest.fixture
def dataset():
    dataset = EvaluationDataset.objects.create(name="Set", version="1")
    EvaluationCase.objects.create(dataset=dataset, question="Q1?", reference_answer="A1.", evidence=["A1"])
    return dataset


@pytest.fixture
def restricted_doc(people):
    return Document.objects.create(
        filename="secret.txt", file_path="k", sha256="s1", visibility="restricted", status="ready", owner=people.analyst
    )


@pytest.mark.parametrize(
    "path",
    [
        "documents/",
        "rag/configs/",
        "evaluation/runs/",
        "evaluation/datasets/",
        "evaluation/metrics/",
        "security/results/",
        "security/tests/",
        "monitoring/metrics/",
        "monitoring/health/",
        "reports/summary/",
    ],
)
def test_every_endpoint_requires_sign_in(path):
    assert client_for().get(f"/api/{path}").status_code == 403


def test_login_and_me(people):
    client = APIClient()
    assert client.post("/api/accounts/login/", {"username": "c", "password": "wrong"}).status_code == 400
    response = client.post("/api/accounts/login/", {"username": "c", "password": "pw-c-12345"})
    assert response.status_code == 200
    assert response.data["role"] == "viewer"
    assert "password" not in response.data
    assert client.get("/api/accounts/me/").data["username"] == "c"


def test_viewer_cannot_upload(people):
    upload = SimpleUploadedFile("a.txt", b"hello world")
    assert client_for(people.viewer).post("/api/documents/upload/", {"file": upload}).status_code == 403


@patch("apps.documents.views.enqueue")
@patch("apps.documents.services.storage.put_object")
def test_analyst_upload_stores_and_queues(put_object, enqueue, people, config):
    upload = SimpleUploadedFile("policy.txt", b"Employees get 24 days.", content_type="text/plain")
    response = client_for(people.analyst).post("/api/documents/upload/", {"file": upload, "visibility": "restricted"})
    assert response.status_code == 201
    document = Document.objects.get(pk=response.data["id"])
    assert document.owner == people.analyst
    assert document.visibility == "restricted"
    assert "file_path" not in response.data
    put_object.assert_called_once()
    enqueue.assert_called_once()


@patch("apps.documents.views.enqueue")
@patch("apps.documents.services.storage.put_object")
def test_duplicate_and_bad_uploads_are_rejected(put_object, enqueue, people, config):
    client = client_for(people.analyst)
    first = client.post("/api/documents/upload/", {"file": SimpleUploadedFile("a.txt", b"same bytes")})
    assert first.status_code == 201
    duplicate = client.post("/api/documents/upload/", {"file": SimpleUploadedFile("b.txt", b"same bytes")})
    assert duplicate.status_code == 400
    assert "already been uploaded" in duplicate.data["error"]
    wrong_type = client.post("/api/documents/upload/", {"file": SimpleUploadedFile("a.exe", b"MZ")})
    assert wrong_type.status_code == 400
    assert "Unsupported" in wrong_type.data["error"]


def test_restricted_document_is_hidden_as_404(people, restricted_doc, config):
    outsider = client_for(people.viewer)
    assert outsider.get("/api/documents/").data["count"] == 0
    for path in ("", "status/", "chunks/"):
        assert outsider.get(f"/api/documents/{restricted_doc.id}/{path}").status_code == 404


def test_owner_and_admin_see_restricted_document(people, restricted_doc, config):
    assert client_for(people.analyst).get(f"/api/documents/{restricted_doc.id}/").status_code == 200
    assert client_for(people.admin).get(f"/api/documents/{restricted_doc.id}/").status_code == 200


def test_viewer_cannot_delete_document(people, config):
    public = Document.objects.create(filename="p.txt", file_path="k2", sha256="s2", visibility="public", status="ready")
    assert client_for(people.viewer).delete(f"/api/documents/{public.id}/").status_code == 403
    assert Document.objects.filter(pk=public.id).exists()


@patch("apps.documents.views.enqueue")
def test_reprocess_shows_queued_and_blocks_a_second_click(enqueue, people, config):
    document = Document.objects.create(filename="p.txt", file_path="k3", sha256="s3", status="ready")
    client = client_for(people.analyst)
    assert client.post(f"/api/documents/{document.id}/process/").status_code == 202
    document.refresh_from_db()
    assert document.status == "uploaded"
    assert client.post(f"/api/documents/{document.id}/process/").status_code == 409
    enqueue.assert_called_once()


@patch("apps.documents.views.enqueue")
def test_reprocess_allowed_again_when_the_job_looks_lost(enqueue, people, config):
    document = Document.objects.create(filename="p.txt", file_path="k4", sha256="s4", status="processing")
    Document.objects.filter(pk=document.id).update(updated_at=timezone.now() - timedelta(minutes=16))
    assert client_for(people.analyst).post(f"/api/documents/{document.id}/process/").status_code == 202


def test_health_reports_whether_the_worker_is_alive(people):
    from apps.evaluation.tasks import WORKER_HEARTBEAT_KEY

    client = client_for(people.viewer)
    cache.delete(WORKER_HEARTBEAT_KEY)
    assert client.get("/api/monitoring/health/").data["worker"] is False
    cache.set(WORKER_HEARTBEAT_KEY, "w1", 30)
    assert client.get("/api/monitoring/health/").data["worker"] is True
    cache.delete(WORKER_HEARTBEAT_KEY)


def test_run_only_searches_the_documents_its_dataset_is_pinned_to(config, dataset):
    from apps.evaluation.services.run_executor import corpus_for

    sample = Document.objects.create(filename="s.txt", file_path="k5", sha256="s5", status="ready")
    Document.objects.create(filename="uploaded_later.pdf", file_path="k6", sha256="s6", status="ready")
    run = EvaluationRun.objects.create(rag_config=config, dataset=dataset)
    assert len(corpus_for(run)) == 2  # not pinned: every ready document
    dataset.documents.add(sample)
    assert corpus_for(run) == [sample.id]


def test_relevance_labels_ignore_documents_outside_the_corpus(config):
    from apps.documents.models import DocumentChunk
    from apps.evaluation.services.dataset_manager import relevant_chunk_ids

    first = Document.objects.create(filename="a.txt", file_path="k7", sha256="s7")
    other = Document.objects.create(filename="b.txt", file_path="k8", sha256="s8")
    text = "Backups are retained for 35 days."
    inside = DocumentChunk.objects.create(document=first, index_key="k", chunk_index=0, chunk_text=text, token_count=6)
    outside = DocumentChunk.objects.create(document=other, index_key="k", chunk_index=0, chunk_text=text, token_count=6)
    assert relevant_chunk_ids([text], "k", [first.id]) == {inside.id}
    assert relevant_chunk_ids([text], "k") == {inside.id, outside.id}


def test_dataset_create_validates_cases(people):
    client = client_for(people.analyst)
    body = {"name": "D", "version": "1", "cases": [{"question": "Q?"}]}
    response = client.post("/api/evaluation/datasets/", body, format="json")
    assert response.status_code == 400
    assert "reference_answer" in response.data["error"]
    body["cases"] = [{"question": "Q?", "reference_answer": "A"}, {"question": "U?", "answerable": False}]
    created = client.post("/api/evaluation/datasets/", body, format="json")
    assert created.status_code == 201
    assert len(created.data["cases"]) == 2
    assert client.post("/api/evaluation/datasets/", body, format="json").status_code == 400  # same version again


def test_dataset_csv_upload(people):
    content = b"question,reference_answer,answerable,evidence\nQ1?,A1,true,one||two\nQ2?,,false,\n"
    upload = SimpleUploadedFile("set.csv", content, content_type="text/csv")
    response = client_for(people.analyst).post(
        "/api/evaluation/datasets/upload/", {"file": upload, "name": "CSV set", "version": "1"}
    )
    assert response.status_code == 201
    assert response.data["cases"][0]["evidence"] == ["one", "two"]
    assert response.data["cases"][1]["answerable"] is False


@pytest.mark.parametrize(
    "filename,data,message",
    [
        # The real mistake: a Word document dropped on the dataset form.
        ("VVDN_HR_Policy_Manual.docx", b"PK\x03\x04\x90 binary", "is a document, not a dataset"),
        ("cases.xml", b"<cases/>", "Upload a .json or .csv file"),
        ("cases.json", b"{not json", "is not valid JSON"),
        ("cases.json", b'{"name": "x"}', 'without a "cases" list'),
    ],
)
def test_a_wrong_dataset_file_gets_a_clear_message(people, filename, data, message):
    upload = SimpleUploadedFile(filename, data)
    response = client_for(people.analyst).post("/api/evaluation/datasets/upload/", {"file": upload, "name": "Set"})
    assert response.status_code == 400
    assert message in response.json()["error"]
    assert "codec" not in response.json()["error"]


def test_a_csv_saved_by_excel_on_windows_is_read(people):
    text = "question,reference_answer,answerable,evidence\nCafé hours?,9 to 5 in the café,true,Café opens at 9\n"
    upload = SimpleUploadedFile("cases.csv", text.encode("cp1252"))
    response = client_for(people.analyst).post(
        "/api/evaluation/datasets/upload/", {"file": upload, "name": "Excel set"}
    )
    assert response.status_code == 201, response.json()


@patch("apps.evaluation.views.enqueue")
def test_start_run_queues_task(enqueue, people, config, dataset):
    body = {"rag_config": config.id, "dataset": dataset.id, "run_type": "quality", "k": 3}
    response = client_for(people.analyst).post("/api/evaluation/runs/", body, format="json")
    assert response.status_code == 201
    assert response.data["status"] == JobStatus.PENDING
    enqueue.assert_called_once()


def test_run_validation(people, config, dataset):
    client = client_for(people.analyst)
    base = {"rag_config": config.id, "dataset": dataset.id}
    assert client.post("/api/evaluation/runs/", {"rag_config": config.id}, format="json").status_code == 400
    assert client.post("/api/evaluation/runs/", {**base, "frameworks": ["nope"]}, format="json").status_code == 400
    assert (
        client.post("/api/evaluation/runs/", {**base, "selected_metrics": ["nope"]}, format="json").status_code == 400
    )
    assert client.post("/api/evaluation/runs/", {**base, "k": 0}, format="json").status_code == 400


def test_viewer_cannot_start_run(people, config, dataset):
    body = {"rag_config": config.id, "dataset": dataset.id}
    assert client_for(people.viewer).post("/api/evaluation/runs/", body, format="json").status_code == 403


@pytest.fixture
def finished_run(config, dataset):
    sync_metric_definitions()
    run = EvaluationRun.objects.create(rag_config=config, dataset=dataset, status=JobStatus.COMPLETED, name="Done")
    result = EvaluationResult.objects.create(
        run=run, case=dataset.cases.first(), answer="=cmd() secret context", contexts=["restricted text"]
    )
    faithfulness = MetricDefinition.objects.get(name="faithfulness")
    MetricScore.objects.create(run=run, result=result, metric=faithfulness, score=0.75)
    MetricScore.objects.create(run=run, result=result, metric=faithfulness, score=0.25, framework="ragas")
    MetricScore.objects.create(run=run, metric=MetricDefinition.objects.get(name="throughput"), score=0.5)
    run.summary = summarize(run)
    run.save()
    return run


def test_summary_averages_with_sql_and_keeps_frameworks_apart(finished_run):
    metrics = finished_run.summary["metrics"]
    assert metrics["faithfulness"]["mean"] == 0.75
    assert metrics["faithfulness:ragas"]["mean"] == 0.25
    assert metrics["throughput"]["mean"] == 0.5
    assert finished_run.summary["metrics_scored"] == 2  # frameworks do not count as extra metrics


@pytest.fixture
def indexed_corpus():
    from apps.documents.models import DocumentChunk

    public = Document.objects.create(filename="pub.txt", file_path="u1", sha256="u1", status="ready")
    secret = Document.objects.create(
        filename="sec.txt", file_path="u2", sha256="u2", status="ready", visibility="restricted"
    )
    for document, text in (
        (public, "The learning budget is INR 40,000. See the Security Policy for Priya Raman."),
        (secret, "Project Bluebird has a budget of INR 90 crore."),
    ):
        DocumentChunk.objects.create(document=document, index_key="k", chunk_index=0, chunk_text=text, token_count=9)
    return public, secret


@pytest.mark.parametrize(
    "question,unknown",
    [
        ("What is the budget for Project Bluebird?", True),  # regression: the model quoted the wrong budget
        ("What is the budget?", False),
        ("Explain the Security Policy.", False),  # title case, but the word is in a readable document
        ("How do I contact Priya Raman?", False),
        ("Bluebird budget", True),  # a keyword search has no sentence-start exemption
        ("priya raman", False),
    ],
)
def test_question_naming_something_unknown_to_this_user(indexed_corpus, question, unknown):
    from apps.chatbot.services.generation import unknown_terms

    public, _ = indexed_corpus
    assert bool(unknown_terms(question, "k", [public.id])) is unknown


def test_the_same_name_is_known_to_a_user_who_may_read_it(indexed_corpus):
    from apps.chatbot.services.generation import unknown_terms

    public, secret = indexed_corpus
    question = "What is the budget for Project Bluebird?"
    assert unknown_terms(question, "k", [public.id]) == ["Project", "Bluebird"]
    assert unknown_terms(question, "k", [public.id, secret.id]) == []


@pytest.fixture
def brochure():
    from apps.documents.models import DocumentChunk

    document = Document.objects.create(filename="Tenarai Mini Brochure.pdf", file_path="b", sha256="b", status="ready")
    DocumentChunk.objects.create(
        document=document, index_key="k", chunk_index=0, chunk_text="Tenarai is an AI company.", token_count=5
    )
    return document


def test_words_in_a_file_name_are_not_unknown(brochure):
    """A real question: "Mini" and "Brochure" appear only in the name, and it was declined without asking the model."""
    from apps.chatbot.services.generation import unknown_terms

    assert unknown_terms("Tenarai Mini Brochure", "k", [brochure.id]) == []


def test_a_keyword_search_needs_one_known_term_but_a_question_needs_all(brochure):
    from apps.chatbot.services.generation import unknown_terms

    assert unknown_terms("Tenarai Company", "k", [brochure.id]) == []
    assert unknown_terms("Zephyrine Bluebird", "k", [brochure.id]) == ["Zephyrine", "Bluebird"]
    assert unknown_terms("What is the budget for Tenarai Bluebird?", "k", [brochure.id]) == ["Bluebird"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("tennari", "tenarai"),
        ("Tenari company", "Tenarai company"),
        ("summerizae the brochure", "summarize the brochure"),
        ("tenarai company", "tenarai company"),  # nothing to correct
        ("cricket world cup", "cricket world cup"),  # nothing close, so left alone
        ("companies", "companies"),  # a real word in another form, not a typo
    ],
)
def test_correct_spelling(brochure, question, expected):
    from apps.chatbot.services.generation import correct_spelling

    assert correct_spelling(question, "k", [brochure.id]) == expected


def test_a_real_word_is_not_turned_into_another_word():
    """Real failure: "resigned" became "designed", and the answer was about designing software."""
    from apps.chatbot.services.generation import correct_spelling
    from apps.documents.models import DocumentChunk

    document = Document.objects.create(filename="r.txt", file_path="r", sha256="r", status="ready")
    text = "His resignation was designed and written up. A falling contact."
    DocumentChunk.objects.create(document=document, index_key="k", chunk_index=0, chunk_text=text, token_count=9)
    for question in ("why he resigned", "write it", "calling him"):
        assert correct_spelling(question, "k", [document.id]) == question
    assert correct_spelling("resgnation", "k", [document.id]) == "resignation"


def test_spelling_cannot_reveal_a_term_from_an_unreadable_document(brochure):
    from apps.chatbot.services.generation import correct_spelling
    from apps.documents.models import DocumentChunk

    secret = Document.objects.create(
        filename="s.txt", file_path="s", sha256="s", status="ready", visibility="restricted"
    )
    DocumentChunk.objects.create(
        document=secret, index_key="k", chunk_index=0, chunk_text="Bluebird budget.", token_count=2
    )
    assert correct_spelling("Blubird", "k", [brochure.id]) == "Blubird"
    assert correct_spelling("Blubird", "k", [brochure.id, secret.id]) == "bluebird".capitalize()


@pytest.mark.parametrize(
    "question,named",
    [
        ("Tenarai Mini Brochure", True),
        ("summary of the brochure", True),
        ("What is Tenarai?", True),
        ("tena", True),
        ("explain the brochure pls", True),
        ("brochure pricing", False),  # another topic, so search everything
        ("summary", False),  # names nothing
    ],
)
def test_a_message_can_pick_a_document_by_file_name(brochure, question, named):
    from apps.chatbot.services.generation import documents_named

    assert documents_named(question, [brochure.id]) == ([brochure.id] if named else [])


def test_i_want_this_policy_picks_the_policy_document():
    from apps.chatbot.services.generation import documents_named

    policy = Document.objects.create(
        filename="VVDN_Technologies_IT_Policy_Manual.docx", file_path="v", sha256="v", status="ready"
    )
    assert documents_named("i want this it policy pls?", [policy.id]) == [policy.id]


def test_naming_a_document_cannot_reach_an_unreadable_one(brochure):
    from apps.chatbot.services.generation import documents_named

    secret = Document.objects.create(
        filename="salary_bands.md", file_path="s", sha256="s", status="ready", visibility="restricted"
    )
    assert documents_named("salary bands", [brochure.id]) == []
    assert documents_named("salary bands", [brochure.id, secret.id]) == [secret.id]


def test_a_real_policy_question_is_not_narrowed_to_a_file_with_policy_in_its_name():
    from apps.chatbot.services.generation import documents_named

    policy = Document.objects.create(filename="it_security_policy.md", file_path="p", sha256="p", status="ready")
    assert documents_named("explain leave policy", [policy.id]) == []
    assert documents_named("security policy", [policy.id]) == [policy.id]


@pytest.fixture
def payslips():
    from apps.documents.models import DocumentChunk

    made = {}
    for month, filename in (("MAY", "Pay Slip_66567_May-26.pdf"), ("FEBRUARY", "Pay Slip_Feb_2026.pdf")):
        document = Document.objects.create(filename=filename, file_path=filename, sha256=filename, status="ready")
        for index, text in enumerate((f"PAYSLIP FOR THE MONTH OF {month} 2026 Emp Name Balamurugan", "LOP Days 0.0")):
            DocumentChunk.objects.create(
                document=document, index_key="k", chunk_index=index, chunk_text=text, token_count=5
            )
        made[month] = document
    return made


def _ask_payslips(monkeypatch, payslips, question, reply, earlier=None, earlier_answer=""):
    from apps.chatbot.services import generation, rag_workflow
    from apps.documents.models import DocumentChunk
    from apps.retrieval.services.hybrid_search import RetrievalResult, RetrievedChunk

    searched = []

    def fake_retrieve(query, config, document_ids, **kwargs):
        searched.append(sorted(document_ids))
        rows = DocumentChunk.objects.filter(document_id__in=document_ids).select_related("document").order_by("id")
        return RetrievalResult(
            chunks=[RetrievedChunk(r.id, r.document_id, r.document.filename, 1, r.chunk_text, 1.0) for r in rows]
        )

    class Fake:
        messages = None

        def complete(self, messages, **kwargs):
            from apps.chatbot.services.llm_gateway import LLMResponse

            Fake.messages = messages
            return LLMResponse(reply, "fake", 10, 5, 1.0)

    monkeypatch.setattr(generation.llm_gateway, "complete", Fake().complete)
    monkeypatch.setattr(rag_workflow, "retrieve", fake_retrieve)
    config = SimpleNamespace(
        name="c", index_key="k", llm_model="fake", prompt_version="v3", temperature=0.0, max_tokens=10
    )
    ids = [document.id for document in payslips.values()]
    return rag_workflow.run_rag(question, config, ids, earlier, earlier_answer), searched, Fake


def test_a_month_question_searches_only_documents_about_that_month(monkeypatch, payslips):
    outcome, searched, _ = _ask_payslips(
        monkeypatch, payslips, "How many LOP days in May 2026?", "In May 2026 LOP was 0 days [S1]."
    )
    assert searched == [[payslips["MAY"].id]]
    assert not outcome.abstained


def test_an_answer_about_another_month_is_withheld(monkeypatch, payslips):
    """The real failure: asked about May, the model gave February's figure."""
    outcome, _, _ = _ask_payslips(
        monkeypatch, payslips, "How many LOP days in May 2026?", "February 2026 LOP is 0 days [S1]."
    )
    assert outcome.abstained
    assert "February" in outcome.decline_reason and "May" in outcome.decline_reason


def test_a_month_no_document_covers_is_declined_without_the_model(monkeypatch, payslips):
    outcome, searched, fake = _ask_payslips(monkeypatch, payslips, "How many LOP days in June 2026?", "x")
    assert outcome.abstained and "June" in outcome.decline_reason
    assert searched == [] and fake.messages is None


def test_this_month_follows_the_conversation(monkeypatch, payslips):
    rewrite = "How many LOP days did Balamurugan have in May 2026?"
    outcome, searched, _ = _ask_payslips(
        monkeypatch,
        payslips,
        "how many lop on this month pls ?",
        rewrite,
        earlier="balamurugan thangadurai may month salary pls ?",
        earlier_answer="The salary for the month of May 2026 is Rs. 57,559.00.",
    )
    assert outcome.standalone_question == rewrite
    assert searched == [[payslips["MAY"].id]]


def test_every_response_has_a_request_id():
    client = APIClient()
    given = client.get("/api/accounts/csrf/", HTTP_X_REQUEST_ID="client-abc-123")
    assert given["X-Request-ID"] == "client-abc-123"
    assert given["X-Correlation-ID"] == "client-abc-123"
    # Not plainly an ID, so it is replaced rather than written into the logs.
    made = client.get("/api/accounts/csrf/", HTTP_X_REQUEST_ID="bad id\nINFO fake log line")
    assert len(made["X-Request-ID"]) == 32 and "\n" not in made["X-Request-ID"]


def test_chat_requests_are_rate_limited(people, monkeypatch):
    from rest_framework.throttling import ScopedRateThrottle

    monkeypatch.setattr(ScopedRateThrottle, "THROTTLE_RATES", {**ScopedRateThrottle.THROTTLE_RATES, "query": "2/min"})
    client = client_for(people.viewer)
    codes = [client.post("/api/rag/query/", {}, format="json").status_code for _ in range(3)]
    assert codes == [400, 400, 429]


def test_repeated_wrong_passwords_lock_the_account_for_a_while(people):
    client = APIClient()
    codes = [
        client.post("/api/accounts/login/", {"username": "c", "password": "wrong"}, format="json").status_code
        for _ in range(5)
    ]
    assert codes == [400, 400, 400, 400, 429]  # the fifth failure locks the account
    locked = client.post("/api/accounts/login/", {"username": "c", "password": "pw-c-12345"}, format="json")
    assert locked.status_code == 429
    assert "Too many failed sign-ins" in locked.json()["error"]
    # Another user from the same address is not locked out.
    assert (
        client.post("/api/accounts/login/", {"username": "d", "password": "pw-d-12345"}, format="json").status_code
        == 200
    )


def test_api_clients_can_sign_in_with_a_jwt(people):
    from apps.security.models import AuditLog

    client = APIClient()
    assert client.post("/api/accounts/token/", {"username": "c", "password": "wrong"}, format="json").status_code == 401
    assert AuditLog.objects.filter(action="login_failed", target="c").exists()
    tokens = client.post("/api/accounts/token/", {"username": "c", "password": "pw-c-12345"}, format="json").json()
    client.credentials(HTTP_AUTHORIZATION=f"Bearer {tokens['access']}")
    assert client.get("/api/accounts/me/").json()["username"] == "c"
    client.credentials(HTTP_AUTHORIZATION="Bearer not-a-token")
    # 403 rather than 401: session sign-in comes first and DRF takes the challenge header from the first method.
    assert client.get("/api/accounts/me/").status_code == 403


def test_feedback_on_your_own_answer(people, conversation):
    message = conversation.session.messages.get()
    url = f"/api/rag/messages/{message.id}/feedback/"
    assert client_for(people.other).post(url, {"rating": "up"}, format="json").status_code == 404
    assert client_for(people.viewer).post(url, {"rating": "great"}, format="json").status_code == 400
    assert client_for(people.viewer).post(url, {"rating": "down", "comment": "Wrong month"}, format="json").json() == {
        "rating": "down",
        "comment": "Wrong month",
    }
    detail = client_for(people.viewer).get(f"/api/rag/sessions/{conversation.session.id}/").json()
    assert detail["messages"][0]["feedback"] == "down"
    assert client_for(people.viewer).delete(url).status_code == 204
    assert (
        client_for(people.viewer).get(f"/api/rag/sessions/{conversation.session.id}/").json()["messages"][0]["feedback"]
        is None
    )


def _stream(client, monkeypatch, pieces, answer="Final answer."):
    from apps.chatbot.services import chat_service
    from apps.chatbot.services.rag_workflow import RagOutcome

    def fake_stream_rag(question, *args, **kwargs):
        for piece in pieces:
            yield "token", piece
        yield "outcome", RagOutcome(question=question, answer=answer)

    monkeypatch.setattr(chat_service, "stream_rag", fake_stream_rag)
    # As browsers and SSE clients send it. Without a renderer for it, DRF answered 406 before the view ran.
    response = client.post(
        "/api/rag/query/stream/", {"question": "Who do I write to?"}, format="json", HTTP_ACCEPT="text/event-stream"
    )
    assert response["Content-Type"] == "text/event-stream"
    events = []
    for block in b"".join(response.streaming_content).decode().strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_metrics_are_served_to_this_machine_only(people, config, monkeypatch):
    _stream(client_for(people.viewer), monkeypatch, ["Leave is 24 days. "])
    local = APIClient().get("/metrics")
    assert local.status_code == 200
    body = local.content.decode()
    assert 'rag_answers_total{outcome="answered"}' in body
    assert "django_http_requests_total_by_method_total" in body
    assert APIClient().get("/metrics", REMOTE_ADDR="10.0.0.5").status_code == 403


def test_streaming_sends_sentences_then_the_final_answer(people, config, monkeypatch):
    events = _stream(
        client_for(people.viewer), monkeypatch, ["Leave is ", "24 days. Sick", " leave is 12 days. Unfinished"]
    )
    assert events[:-1] == [("text", {"text": "Leave is 24 days."}), ("text", {"text": " Sick leave is 12 days."})]
    name, done = events[-1]
    assert name == "done" and done["answer"] == "Final answer." and done["message_id"]
    assert done["request_id"]


def test_a_stream_request_that_fails_validation_gets_a_readable_error(people):
    response = client_for(people.viewer).post(
        "/api/rag/query/stream/", {}, format="json", HTTP_ACCEPT="text/event-stream"
    )
    assert response.status_code == 400
    assert "question" in json.loads(response.content)


def test_streaming_masks_pii_before_it_reaches_the_screen(people, config, monkeypatch):
    events = _stream(client_for(people.viewer), monkeypatch, ["Write to priya.raman@example.com today. "])
    streamed = " ".join(data["text"] for name, data in events if name == "text")
    assert "priya.raman@example.com" not in streamed and "[EMAIL_ADDRESS]" in streamed


def test_streaming_stops_at_a_blocked_sentence(people, config, monkeypatch):
    events = _stream(
        client_for(people.viewer), monkeypatch, ["Fine sentence. ", "You are stupid idiots. ", "More text. "]
    )
    assert [data["text"] for name, data in events if name == "text"] == ["Fine sentence."]


def test_definitions_come_back_in_roadmap_order(people):
    sync_metric_definitions()
    rows = client_for(people.viewer).get("/api/evaluation/metrics/").data
    assert [row["order"] for row in rows] == list(range(1, 38))
    assert rows[0]["display_name"] == "Precision@K"
    assert rows[1]["priority"] == "Highly Critical"
    assert rows[20]["display_name"] == "Contradiction Detection"


def test_compare_lists_metrics_in_roadmap_order(people, config, dataset, finished_run):
    other = EvaluationRun.objects.create(rag_config=config, dataset=dataset, status=JobStatus.COMPLETED)
    other.summary = finished_run.summary
    other.save()
    rows = client_for(people.viewer).get(f"/api/reports/compare/?runs={finished_run.id},{other.id}").data["metrics"]
    assert [row["order"] for row in rows] == sorted(row["order"] for row in rows)
    assert rows[0]["key"] == "faithfulness"
    assert rows[1]["key"] == "faithfulness:ragas"


def test_viewer_cannot_read_answers_or_contexts(people, finished_run):
    """Evaluation runs read restricted documents, so their text is not for viewers."""
    viewer = client_for(people.viewer)
    result_id = finished_run.results.first().id
    assert viewer.get(f"/api/evaluation/runs/{finished_run.id}/results/").status_code == 403
    assert viewer.get(f"/api/evaluation/results/{result_id}/").status_code == 403
    assert viewer.get(f"/api/reports/runs/{finished_run.id}/export/").status_code == 403
    assert viewer.get("/api/security/results/").status_code == 403
    # The scores themselves are fine to see.
    assert viewer.get(f"/api/evaluation/runs/{finished_run.id}/").status_code == 200


def test_analyst_reads_results(people, finished_run):
    response = client_for(people.analyst).get(f"/api/evaluation/runs/{finished_run.id}/results/")
    assert response.status_code == 200
    assert response.data["results"][0]["contexts"] == ["restricted text"]


def test_csv_export_defuses_formula_injection(people, finished_run):
    response = client_for(people.analyst).get(f"/api/reports/runs/{finished_run.id}/export/?type=csv")
    assert response.status_code == 200
    assert response["Content-Type"] == "text/csv"
    body = response.content.decode()
    assert "'=cmd()" in body
    assert "faithfulness" in body


def test_json_export_records_who_exported(people, finished_run):
    assert client_for(people.analyst).get(f"/api/reports/runs/{finished_run.id}/export/").status_code == 200
    report = finished_run.reports.get()
    assert report.created_by == people.analyst


def test_bad_export_type(people, finished_run):
    assert client_for(people.analyst).get(f"/api/reports/runs/{finished_run.id}/export/?type=exe").status_code == 400


def test_compare_needs_two_runs(people, finished_run):
    client = client_for(people.viewer)
    assert client.get(f"/api/reports/compare/?runs={finished_run.id}").status_code == 400
    assert client.get("/api/reports/compare/?runs=x,y").status_code == 400


def test_compare_picks_best_respecting_direction(people, config, dataset, finished_run):
    sync_metric_definitions()
    other_config = RagConfig.objects.create(name="Other", top_k=9)
    other = EvaluationRun.objects.create(rag_config=other_config, dataset=dataset, status=JobStatus.COMPLETED)
    other.summary = {
        "metrics": {
            "faithfulness": {**finished_run.summary["metrics"]["faithfulness"], "mean": 0.9},
            "throughput": {**finished_run.summary["metrics"]["throughput"], "mean": 0.1},
        }
    }
    other.snapshot = {"rag_config": {"top_k": "9"}}
    other.save()
    finished_run.snapshot = {"rag_config": {"top_k": "5"}}
    finished_run.save()

    response = client_for(people.viewer).get(f"/api/reports/compare/?runs={finished_run.id},{other.id}")
    assert response.status_code == 200
    rows = {row["key"]: row for row in response.data["metrics"]}
    assert rows["faithfulness"]["best_run"] == other.id
    assert rows["throughput"]["best_run"] == finished_run.id  # higher throughput is better
    assert response.data["differing_settings"][0]["setting"] == "top_k"


def test_lower_is_better_metric_marks_smallest_as_best():
    from apps.reports.services.comparison import _better

    assert _better({1: 0.2, 2: 0.1}, higher_is_better=False) == 2
    assert _better({1: 0.2, 2: 0.2}, higher_is_better=False) is None


def test_values_that_look_the_same_are_not_called_best():
    from apps.reports.services.comparison import _better

    assert _better({1: 0.968, 2: 0.965, 3: 0.967}, higher_is_better=True) is None
    assert _better({1: 0.90, 2: 0.85}, higher_is_better=True) == 1


def test_summary_finds_security_scores_when_the_latest_run_has_none(people, config, dataset):
    sync_metric_definitions()
    security = EvaluationRun.objects.create(
        rag_config=config, run_type="full", dataset=dataset, status=JobStatus.COMPLETED
    )
    EvaluationRun.objects.filter(pk=security.id).update(finished_at=timezone.now() - timedelta(hours=1))
    quality = EvaluationRun.objects.create(
        rag_config=config, dataset=dataset, status=JobStatus.COMPLETED, finished_at=timezone.now()
    )
    data = client_for(people.viewer).get("/api/reports/summary/").data
    assert data["latest_run"]["id"] == quality.id
    assert data["latest_security_run"]["id"] == security.id


@patch("apps.retrieval.views.retrieve")
def test_viewer_cannot_read_pii_from_the_sources_beside_a_masked_answer(retrieve, people, config):
    """Regression: the answer was masked, but the raw chunks returned next to it were not."""
    from apps.retrieval.services.hybrid_search import RetrievalResult, RetrievedChunk

    chunk = RetrievedChunk(1, 1, "c.csv", 1, "Contact priya.raman@example.com for help.", 1.0)
    retrieve.return_value = RetrievalResult(chunks=[chunk])
    body = {"question": "contact"}
    assert "priya.raman" not in str(client_for(people.viewer).post("/api/rag/retrieve/", body, format="json").data)
    assert "priya.raman" in str(client_for(people.analyst).post("/api/rag/retrieve/", body, format="json").data)


@pytest.mark.parametrize("role", ["viewer", "analyst"])
def test_document_chunks_are_masked_for_viewers_only(people, config, role):
    from apps.documents.models import DocumentChunk

    document = Document.objects.create(filename="c.csv", file_path="c9", sha256="c9", status="ready")
    DocumentChunk.objects.create(
        document=document,
        index_key=config.index_key,
        chunk_index=0,
        chunk_text="Mail priya.raman@example.com now.",
        token_count=5,
    )
    text = str(client_for(getattr(people, role)).get(f"/api/documents/{document.id}/chunks/").data)
    assert ("priya.raman" in text) is (role == "analyst")


def test_dataset_can_be_deleted_only_when_no_run_uses_it(people, config, dataset):
    client = client_for(people.analyst)
    run = EvaluationRun.objects.create(rag_config=config, dataset=dataset, status=JobStatus.COMPLETED)
    assert client.delete(f"/api/evaluation/datasets/{dataset.id}/").status_code == 409
    run.delete()
    assert client.delete(f"/api/evaluation/datasets/{dataset.id}/").status_code == 204


@patch("apps.chatbot.views.ask", side_effect=ServiceUnavailable("Ollama is down"))
def test_llm_outage_returns_503(ask, people, config):
    response = client_for(people.viewer).post("/api/rag/query/", {"question": "Hello?"}, format="json")
    assert response.status_code == 503
    assert response.data["error"] == "Ollama is down"


def test_overlong_question_is_a_recorded_block_not_a_validation_error(people, config):
    from apps.chatbot.models import ChatMessage

    response = client_for(people.viewer).post("/api/rag/query/", {"question": "ignore " * 500}, format="json")
    assert response.status_code == 200
    assert response.data["blocked"] is True
    assert response.data["block_reason"] == "question_too_long"
    assert ChatMessage.objects.get().blocked is True


@pytest.fixture
def conversation(people, config):
    from apps.chatbot.models import ChatMessage, ChatSession
    from apps.documents.models import DocumentChunk
    from apps.monitoring.models import PerformanceLog
    from apps.retrieval.models import RetrievalLog

    public = Document.objects.create(filename="pub.txt", file_path="h1", sha256="h1", status="ready")
    secret = Document.objects.create(
        filename="sec.txt", file_path="h2", sha256="h2", status="ready", visibility="restricted"
    )
    readable = DocumentChunk.objects.create(
        document=public, index_key="k", chunk_index=0, chunk_text="Mail priya.raman@example.com.", token_count=3
    )
    hidden = DocumentChunk.objects.create(
        document=secret, index_key="k", chunk_index=0, chunk_text="Secret bonus.", token_count=2
    )
    session = ChatSession.objects.create(user=people.viewer, title="Leave questions", rag_config=config)
    message = ChatMessage.objects.create(session=session, question="Q?", answer="A.", quoted=True)
    RetrievalLog.objects.create(
        message=message,
        retrieved_chunk_ids=[readable.id, hidden.id],
        stages={"rerank": [{"chunk_id": readable.id, "score": 4.5}], "timings_ms": {"rerank": 12.0}},
    )
    PerformanceLog.objects.create(message=message, retrieval_ms=10, generation_ms=20, total_ms=30)
    return SimpleNamespace(session=session, readable=readable, hidden=hidden)


def test_history_lists_only_your_own_conversations_newest_first(people, config, conversation):
    from apps.chatbot.models import ChatSession

    newer = ChatSession.objects.create(user=people.viewer, title="Newer", rag_config=config)
    ChatSession.objects.create(user=people.other, title="Someone else's", rag_config=config)
    rows = client_for(people.viewer).get("/api/rag/sessions/").data["results"]
    assert [row["title"] for row in rows] == [newer.title, "Leave questions"]
    assert rows[1]["message_count"] == 1
    assert "messages" not in rows[0]


def test_opening_a_conversation_restores_its_sources_and_timings(people, conversation):
    message = client_for(people.viewer).get(f"/api/rag/sessions/{conversation.session.id}/").data["messages"][0]
    assert message["quoted"] is True
    assert message["stages"]["rerank"][0]["score"] == 4.5
    assert "timings_ms" not in message["stages"]
    assert message["timings_ms"]["total"] == 30
    assert message["timings_ms"]["rerank"] == 12.0


def test_history_rechecks_permissions_and_masks_for_viewers(people, conversation):
    """A saved conversation must not be a way around permissions removed later, or around masking."""
    sources = (
        client_for(people.viewer).get(f"/api/rag/sessions/{conversation.session.id}/").data["messages"][0]["sources"]
    )
    assert [source["chunk_id"] for source in sources] == [conversation.readable.id]
    assert "priya.raman" not in sources[0]["text"]
    assert sources[0]["score"] == 4.5


def test_you_can_delete_your_own_conversation_but_not_anothers(people, conversation):
    assert client_for(people.other).delete(f"/api/rag/sessions/{conversation.session.id}/").status_code == 404
    assert client_for(people.other).get(f"/api/rag/sessions/{conversation.session.id}/").status_code == 404
    assert client_for(people.viewer).delete(f"/api/rag/sessions/{conversation.session.id}/").status_code == 204
    assert client_for(people.viewer).get(f"/api/rag/sessions/{conversation.session.id}/").status_code == 404


def test_cannot_continue_another_users_session(people, config):
    from apps.chatbot.models import ChatSession

    session = ChatSession.objects.create(user=people.other, rag_config=config)
    response = client_for(people.viewer).post(
        "/api/rag/query/", {"question": "Hi?", "session": session.id}, format="json"
    )
    assert response.status_code == 404


def test_audit_log_is_admin_only(people):
    assert client_for(people.analyst).get("/api/security/audit/").status_code == 403
    assert client_for(people.admin).get("/api/security/audit/").status_code == 200


def test_config_validation(people):
    client = client_for(people.analyst)
    bad = {"name": "X", "chunk_size": 10, "chunk_overlap": 20}
    assert client.post("/api/rag/configs/", bad, format="json").status_code == 400
    assert client.post("/api/rag/configs/", {"name": "Y", "prompt_version": "v99"}, format="json").status_code == 400
    assert client.post("/api/rag/configs/", {"name": "Z", "mmr_lambda": 2}, format="json").status_code == 400


def test_index_key_changes_only_with_index_settings():
    base = RagConfig(name="a")
    assert base.index_key == RagConfig(name="b", top_k=9, llm_model="other").index_key
    assert base.index_key != RagConfig(name="c", chunk_size=100).index_key
    assert base.index_key != RagConfig(name="d", embedding_model="other").index_key
