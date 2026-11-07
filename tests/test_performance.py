"""Telemetry SQL against real Postgres (percentile_cont), and integration tests of the real pipeline.

The integration tests need Qdrant, Ollama and the ML models. Run them with: pytest -m integration
"""

from decimal import Decimal

import pytest
from django.contrib.auth import get_user_model

from apps.chatbot.models import ChatMessage, ChatSession
from apps.common.constants import Role
from apps.common.exceptions import ServiceUnavailable
from apps.monitoring.models import CostRecord, PerformanceLog, TokenUsage
from apps.monitoring.services.cost_tracker import compute_cost
from apps.monitoring.services.latency import live_metrics
from apps.retrieval.models import RagConfig


@pytest.fixture
def traffic(db):
    user = get_user_model().objects.create_user("u", password="x", role=Role.VIEWER)
    session = ChatSession.objects.create(user=user)
    for total in (100.0, 200.0, 300.0, 400.0):
        message = ChatMessage.objects.create(session=session, question="q")
        PerformanceLog.objects.create(message=message, retrieval_ms=total / 10, total_ms=total)
        TokenUsage.objects.create(message=message, model_name="m", input_tokens=10, output_tokens=5)
        CostRecord.objects.create(message=message, input_cost=Decimal("1.0"), output_cost=Decimal("0.5"))
    failed = ChatMessage.objects.create(session=session, question="q")
    PerformanceLog.objects.create(message=failed, total_ms=0, succeeded=False)


def test_live_metrics_percentiles_ignore_failed_requests(traffic):
    metrics = live_metrics(7)
    assert metrics["requests"] == 5
    assert metrics["errors"] == 1
    assert metrics["error_rate"] == pytest.approx(0.2)
    assert metrics["p50_ms"] == pytest.approx(250.0)
    assert metrics["p95_ms"] == pytest.approx(385.0)


def test_live_metrics_totals(traffic):
    metrics = live_metrics(7)
    assert metrics["input_tokens"] == 40
    assert metrics["output_tokens"] == 20
    assert metrics["total_cost"] == pytest.approx(6.0)
    assert metrics["cost_per_query"] == pytest.approx(6.0 / 5)


def test_live_metrics_empty(db):
    metrics = live_metrics(7)
    assert metrics["requests"] == 0
    assert metrics["error_rate"] == 0.0
    assert metrics["p95_ms"] == 0.0


def test_compute_cost():
    config = RagConfig(input_cost_per_1k=Decimal("0.5"), output_cost_per_1k=Decimal("2"))
    assert compute_cost(config, 2000, 500) == (Decimal("1.0"), Decimal("1.0"))


def test_local_model_costs_nothing_by_default():
    assert compute_cost(RagConfig(), 2000, 500) == (Decimal("0"), Decimal("0"))


@pytest.mark.django_db
def test_llm_failure_is_recorded_for_the_error_rate(monkeypatch):
    """A failed question must leave a row behind, or the error rate would always read zero."""
    from apps.chatbot.services import chat_service

    def fail(*args, **kwargs):
        raise ServiceUnavailable("down")

    monkeypatch.setattr(chat_service, "run_rag", fail)
    user = get_user_model().objects.create_user("v", password="x")
    with pytest.raises(ServiceUnavailable):
        chat_service.ask(user, "Hello?", RagConfig.objects.create(name="C"))
    assert PerformanceLog.objects.get().succeeded is False


@pytest.mark.django_db
def test_run_records_failures_without_losing_other_cases(monkeypatch):
    """One failed question must not lose the rest of the run."""
    from apps.chatbot.services.rag_workflow import RagOutcome
    from apps.evaluation.models import EvaluationCase, EvaluationDataset, EvaluationRun
    from apps.evaluation.services import run_executor

    def flaky(question, config, allowed):
        if question == "boom":
            raise ServiceUnavailable("LLM down")
        return RagOutcome(question=question, answer="I don't know based on the provided documents.", abstained=True)

    monkeypatch.setattr(run_executor, "run_rag", flaky)
    dataset = EvaluationDataset.objects.create(name="D")
    EvaluationCase.objects.create(dataset=dataset, question="fine", answerable=False)
    EvaluationCase.objects.create(dataset=dataset, question="boom", answerable=False)
    run = EvaluationRun.objects.create(rag_config=RagConfig.objects.create(name="C"), dataset=dataset)

    run_executor.answer_cases(run)

    run.refresh_from_db()
    assert run.completed_cases == 1
    assert run.failed_cases == 1
    assert run.results.count() == 1
    assert run.errors.get().stage == "answering"


@pytest.mark.django_db
def test_resumed_run_keeps_answers_and_retries_failures(monkeypatch):
    """After a crash, finished questions must not be asked again, and failed ones get another try."""
    from apps.chatbot.services.rag_workflow import RagOutcome
    from apps.evaluation.models import (
        EvaluationCase,
        EvaluationDataset,
        EvaluationError,
        EvaluationResult,
        EvaluationRun,
    )
    from apps.evaluation.services import run_executor

    asked = []

    def record(question, config, allowed):
        asked.append(question)
        return RagOutcome(question=question, answer="I don't know.", abstained=True)

    monkeypatch.setattr(run_executor, "run_rag", record)
    monkeypatch.setattr(run_executor, "_warm_up", lambda config, allowed: None)
    dataset = EvaluationDataset.objects.create(name="D")
    answered = EvaluationCase.objects.create(dataset=dataset, question="answered before the crash", answerable=False)
    EvaluationCase.objects.create(dataset=dataset, question="failed before the crash", answerable=False)
    EvaluationCase.objects.create(dataset=dataset, question="never reached", answerable=False)
    run = EvaluationRun.objects.create(rag_config=RagConfig.objects.create(name="C"), dataset=dataset)
    EvaluationResult.objects.create(run=run, case=answered, answer="kept")
    EvaluationError.objects.create(run=run, stage="answering", error_message="LLM down")

    run_executor.answer_cases(run)

    assert asked == ["failed before the crash", "never reached"]
    assert run.results.get(case=answered).answer == "kept"
    assert run.results.count() == 3
    assert run.completed_cases == 3
    assert run.failed_cases == 0
    assert not run.errors.exists()


@pytest.mark.django_db
def test_queued_task_for_a_deleted_run_does_nothing():
    from apps.evaluation.services.run_executor import execute_run

    assert execute_run(999999) is None


@pytest.mark.django_db
def test_finished_run_is_not_executed_again():
    """Several restarts can queue several resumes. Only the first may do any work."""
    from apps.common.constants import JobStatus
    from apps.evaluation.models import EvaluationRun
    from apps.evaluation.services.run_executor import execute_run

    run = EvaluationRun.objects.create(
        rag_config=RagConfig.objects.create(name="C"),
        run_type="security",
        status=JobStatus.COMPLETED,
        summary={"kept": 1},
    )
    execute_run(run.id)
    run.refresh_from_db()
    assert run.status == JobStatus.COMPLETED
    assert run.summary == {"kept": 1}


@pytest.mark.django_db
def test_worker_restart_resumes_only_its_own_runs(monkeypatch):
    from types import SimpleNamespace

    from apps.common.constants import JobStatus
    from apps.evaluation import tasks
    from apps.evaluation.models import EvaluationRun

    config = RagConfig.objects.create(name="C")
    mine = EvaluationRun.objects.create(rag_config=config, run_type="security", status=JobStatus.RUNNING, worker="w1")
    EvaluationRun.objects.create(rag_config=config, run_type="security", status=JobStatus.RUNNING, worker="w2")
    EvaluationRun.objects.create(rag_config=config, run_type="security", status=JobStatus.COMPLETED, worker="w1")
    queued = []
    monkeypatch.setattr(tasks.run_evaluation_task, "delay", queued.append)

    tasks.resume_interrupted_runs(sender=SimpleNamespace(hostname="w1"))

    assert queued == [mine.id]


class TestRealPipeline:
    """Uses the real embedding, reranker and NLI models, Qdrant, and Postgres. No LLM, so these stay quick."""

    pytestmark = [pytest.mark.integration, pytest.mark.django_db]

    @pytest.fixture
    def indexed(self):
        from apps.documents.models import Document, DocumentPage
        from apps.documents.services.ingestion import index_document
        from apps.retrieval.services.vector_search import drop_collection

        config = RagConfig.objects.create(name="Integration", chunk_size=40, chunk_overlap=5, top_k=2, candidate_k=5)
        public = Document.objects.create(
            filename="pub.txt", file_path="p", sha256="p", status="ready", visibility="public"
        )
        secret = Document.objects.create(
            filename="sec.txt", file_path="s", sha256="s", status="ready", visibility="restricted"
        )
        DocumentPage.objects.create(
            document=public, page_number=1, content="Annual leave is 24 days for every employee."
        )
        DocumentPage.objects.create(document=secret, page_number=1, content="The secret annual leave bonus is 99 days.")
        index_document(public, config)
        index_document(secret, config)
        yield config, public, secret
        drop_collection(config.index_name)

    def test_hybrid_retrieval_finds_the_passage(self, indexed):
        from apps.retrieval.services.hybrid_search import retrieve

        config, public, secret = indexed
        result = retrieve("How long is annual leave?", config, [public.id, secret.id])
        assert result.chunks
        assert set(result.stages) >= {"vector", "bm25", "rrf", "rerank"}

    def test_permissions_hold_in_both_retrievers(self, indexed):
        """The core security promise: a restricted document can never come back, however well it matches."""
        from apps.retrieval.services.hybrid_search import retrieve

        config, public, secret = indexed
        result = retrieve("What is the secret annual leave bonus?", config, [public.id])
        assert result.chunks
        assert {chunk.document_id for chunk in result.chunks} == {public.id}
        for stage in ("vector", "bm25", "rrf", "rerank"):
            ids = {hit["chunk_id"] for hit in result.stages[stage]}
            assert not ids & set(secret.chunks.values_list("id", flat=True)), stage

    def test_no_permissions_means_no_results(self, indexed):
        from apps.retrieval.services.hybrid_search import retrieve

        config, _, _ = indexed
        assert retrieve("annual leave", config, []).chunks == []

    def test_stale_index_entry_cannot_reach_the_llm(self, indexed):
        """Even if an index still holds a deleted chunk, the SQL re-check drops it."""
        from apps.retrieval.services.hybrid_search import retrieve

        config, public, _ = indexed
        public.chunks.all().delete()  # the row goes, but the vector stays in Qdrant
        assert retrieve("annual leave", config, [public.id]).chunks == []

    def test_nli_separates_supported_from_invented_claims(self):
        from apps.evaluation.services import nli

        source = ["Full-time employees receive 24 days of paid annual leave per calendar year."]
        supported, invented, contradicted = nli.judge_claims(
            [
                "Employees get 24 days of annual leave.",
                "Employees receive a free company car.",
                "Employees receive 5 days of paid annual leave.",
            ],
            source,
        )
        assert supported.supported
        assert not invented.supported
        assert not contradicted.supported

    EXPENSES = (
        "# Expenses\n\nExpense claims must be submitted within 30 days of the purchase. The meal allowance during "
        "business travel is INR 1,500 per day. Any single expense above INR 25,000 requires pre-approval."
    )

    def test_claim_copied_from_a_passage_with_other_numbers_is_supported(self):
        """Regression: this scored 0.03, because the passage's other numbers looked like contradictions."""
        from apps.evaluation.services import nli

        claim = "The meal allowance during business travel is INR 1,500 per day."
        assert nli.judge_claims([claim], [self.EXPENSES])[0].supported

    @pytest.mark.parametrize(
        "claim",
        [
            "The meal allowance during business travel is INR 5,000 per day.",
            "Expense claims must be submitted within 90 days of the purchase.",
            "Employees receive a free company car.",
        ],
    )
    def test_changed_numbers_and_inventions_are_still_rejected(self, claim):
        """The other half of that fix: judging single sentences must not make everything pass."""
        from apps.evaluation.services import nli

        assert not nli.judge_claims([claim], [self.EXPENSES])[0].supported

    def test_semantic_chunking_splits_on_topic_with_the_real_model(self):
        from functools import partial

        from apps.documents.services.chunking import chunk_semantic
        from apps.documents.services.embeddings import embed_texts
        from apps.documents.services.parser import ParsedPage

        text = (
            "Employees receive 24 days of annual leave. Sick leave is 12 days per year. Parental leave lasts 16 weeks. "
            "Production databases are backed up every night. Backups are retained for 35 days. Restore tests run quarterly."
        )
        embed = partial(embed_texts, model_name="sentence-transformers/all-MiniLM-L6-v2")
        drafts = chunk_semantic([ParsedPage(1, text)], size=200, embed_fn=embed)
        assert len(drafts) >= 2
        assert not any("leave" in draft.text and "backed up" in draft.text for draft in drafts)

    def test_scanned_page_with_no_text_layer_is_read_by_ocr(self):
        import io
        import shutil

        import pymupdf
        from PIL import Image, ImageDraw

        from apps.documents.services.parser import parse_document

        if not shutil.which("tesseract"):
            pytest.skip("tesseract is not installed")
        image = Image.new("RGB", (1000, 200), "white")
        ImageDraw.Draw(image).text((20, 60), "Scanned invoice number 48213 is approved", fill="black", font_size=40)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=700, height=200)
            page.insert_image(page.rect, stream=buffer.getvalue())
            data = pdf.tobytes()
        assert "48213" in parse_document(data, "scan.pdf")[0].content

    def _correctness(self, answer, reference, contexts):
        from apps.evaluation.services.evaluator_registry import EvalContext, load_all_metrics

        ctx = EvalContext(question="q", answer=answer, reference=reference, contexts=contexts)
        return load_all_metrics().evaluate("answer_correctness", ctx)

    def test_correct_answer_with_extra_true_detail_scores_high(self):
        """Regression: this scored 0.19 because the answer says more than the reference."""
        value = self._correctness(
            "The production databases are backed up every night and are retained for 35 days.",
            "Backups are retained for 35 days.",
            ["Production databases are backed up every night. Backups are retained for 35 days."],
        )
        assert value.score > 0.8
        assert value.details["fp"] == 0

    def test_wrong_answer_still_scores_low(self):
        value = self._correctness(
            "Backups are retained for 90 days.",
            "Backups are retained for 35 days.",
            ["Production databases are backed up every night. Backups are retained for 35 days."],
        )
        assert value.score < 0.5

    def test_presidio_finds_and_masks_pii(self):
        from apps.security.services.guardrails import check_output

        verdict = check_output("Contact priya.raman@example.com or call +1 212 555 0147 about the invoice.")
        assert "priya.raman" not in verdict.text
        assert "0147" not in verdict.text
        assert "EMAIL_ADDRESS" in verdict.findings["pii"]
        assert "invoice" in verdict.text

    def test_output_guard_masks_secret_in_changed_case(self):
        """Regression: the LLM repeated a key as SK-... and a case-sensitive pattern let it through."""
        from apps.security.services.guardrails import check_output

        verdict = check_output("[SK-meridian-FAKE7Hq2Lm9Xw4Tz8Vb3]")
        assert "FAKE7Hq2" not in verdict.text
        assert verdict.findings["secrets"] == ["API_KEY"]


@pytest.mark.integration
class TestOpenSearch:
    """Runs only when BM25_BACKEND=opensearch points at a live server."""

    @pytest.fixture
    def backend(self):
        from config.env import env

        if env.BM25_BACKEND != "opensearch":
            pytest.skip("BM25_BACKEND is not opensearch")
        from apps.retrieval.services.bm25_search import OpenSearchBM25

        backend = OpenSearchBM25()
        if not backend.ping():
            pytest.skip("OpenSearch is not reachable")
        return backend

    def test_index_search_filter_delete(self, backend):
        from types import SimpleNamespace

        config = SimpleNamespace(index_name="rageval_pytest")
        backend.client.indices.delete(index=config.index_name, ignore_unavailable=True)
        chunks = [
            SimpleNamespace(id=1, document_id=10, chunk_text="Annual leave is 24 days."),
            SimpleNamespace(id=2, document_id=20, chunk_text="Annual leave bonus is secret."),
            SimpleNamespace(id=3, document_id=10, chunk_text="Backups run nightly."),
        ]
        try:
            backend.index(config, chunks)
            assert {h.chunk_id for h in backend.search(config, "annual leave", 5, [10, 20])} == {1, 2}
            # The permission filter applies inside the search engine.
            assert {h.chunk_id for h in backend.search(config, "annual leave", 5, [10])} == {1}
            assert backend.search(config, "annual leave", 5, []) == []
            backend.delete_document(config, 10)
            assert {h.chunk_id for h in backend.search(config, "annual leave", 5, [10, 20])} == {2}
        finally:
            backend.client.indices.delete(index=config.index_name, ignore_unavailable=True)
