"""Batch 3 of the production pipeline: the answer cache, parallel search, first-token time, speed, and SLOs."""

import json
import threading
from types import SimpleNamespace

import pytest
from rest_framework.test import APIClient

from apps.chatbot.models import ChatMessage, ChatSession, LLMRun
from apps.chatbot.services import answer_cache, chat_service
from apps.chatbot.services.llm_gateway import LLMResponse
from apps.chatbot.services.rag_workflow import RagOutcome, _speed
from apps.common.constants import Role
from apps.documents.models import Document, DocumentChunk
from apps.monitoring.models import PerformanceLog
from apps.retrieval.models import RagConfig
from apps.retrieval.services import hybrid_search
from apps.retrieval.services.hybrid_search import RetrievedChunk
from apps.retrieval.services.vector_search import Hit

pytestmark = pytest.mark.django_db


@pytest.fixture
def setup(django_user_model):
    make = django_user_model.objects.create_user
    viewer = make("viewer", password="pw-v-12345", role=Role.VIEWER)
    analyst = make("analyst", password="pw-a-12345", role=Role.ANALYST)
    config = RagConfig.objects.create(name="Cache config", is_default=True)
    document = Document.objects.create(filename="hr.docx", file_path="h", sha256="h1", status="ready")
    return SimpleNamespace(viewer=viewer, analyst=analyst, config=config, document=document)


def _answer(question, **changes):
    chunk = RetrievedChunk(1, 1, "hr.docx", 1, "Earned leave is 18 days.", 2.0)
    fields = dict(
        question=question, answer="Earned leave is 18 days [S1].", stages={"rerank": [{"chunk_id": 1, "score": 2.0}]}
    )
    fields.update(
        model="ollama_chat/test", prompt_tokens=50, completion_tokens=10, tokens_per_second=6.0, chunks=[chunk]
    )
    fields.update(changes)
    return RagOutcome(**fields)


@pytest.fixture
def pipeline(monkeypatch):
    """chat_service.run_rag replaced; `calls` lists the questions that reached it, `next` shapes the next answer."""
    state = SimpleNamespace(calls=[], next={})

    def fake_run_rag(question, config, allowed, earlier_question=None, earlier_answer="", interactive=False):
        state.calls.append((question, earlier_question))
        return _answer(question, **state.next)

    monkeypatch.setattr(chat_service, "run_rag", fake_run_rag)
    return state


def test_a_repeated_question_is_answered_from_the_cache(setup, pipeline):
    first, _ = chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    second, outcome = chat_service.ask(setup.viewer, "How many  earned leave days", setup.config)  # spacing, no "?"
    assert len(pipeline.calls) == 1
    assert second.cached and not first.cached and second.answer == first.answer
    # No model ran, so nothing is charged or counted as model usage.
    assert outcome.model == "" and outcome.tokens_per_second == 0 and not LLMRun.objects.filter(message=second).exists()
    assert PerformanceLog.objects.get(message=second).total_ms < 1000


def test_the_cache_never_crosses_permissions(setup, pipeline):
    Document.objects.create(
        filename="secret.txt", file_path="s", sha256="s1", visibility="restricted", status="ready", owner=setup.analyst
    )
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    _, outcome = chat_service.ask(setup.analyst, "How many earned leave days?", setup.config)
    assert len(pipeline.calls) == 2 and not outcome.cached  # the analyst can read more, so may get another answer


def test_a_document_or_config_change_empties_the_cache(setup, pipeline):
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    setup.document.save()
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    setup.config.save()
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    assert len(pipeline.calls) == 3


def test_a_different_model_is_a_different_answer(setup, pipeline):
    import copy

    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    other = copy.copy(setup.config)
    other.llm_model = "ollama_chat/qwen2.5:3b"
    chat_service.ask(setup.viewer, "How many earned leave days?", other)
    assert len(pipeline.calls) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"abstained": True, "answer": "I don't know."},
        {"checks": {"numbers": ["5"]}},
        {"findings": {"output": {"EMAIL_ADDRESS": 1}}},
        {"blocked": True, "block_reason": "injection"},
        {"stages": {}},  # a greeting: no search
    ],
)
def test_only_clean_answers_are_kept(setup, pipeline, changes):
    pipeline.next = changes
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    assert len(pipeline.calls) == 2


def test_answer_again_skips_the_cache(setup, pipeline):
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    message, _ = chat_service.ask(setup.viewer, "How many earned leave days?", setup.config, fresh=True)
    assert len(pipeline.calls) == 2 and not message.cached


def test_follow_ups_are_never_cached(setup, pipeline):
    session = ChatSession.objects.create(user=setup.viewer, title="t", rag_config=setup.config)
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config, session)
    chat_service.ask(setup.viewer, "and sick leave?", setup.config, session)
    # The same words in a new conversation mean something else, so they are not answered from the first one.
    _, outcome = chat_service.ask(setup.viewer, "and sick leave?", setup.config)
    assert [call[0] for call in pipeline.calls] == ["How many earned leave days?", "and sick leave?", "and sick leave?"]
    assert pipeline.calls[1][1] == "How many earned leave days?" and not outcome.cached


def test_the_cache_can_be_turned_off(setup, pipeline, monkeypatch):
    monkeypatch.setattr(answer_cache.env, "ANSWER_CACHE_SECONDS", 0)
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    chat_service.ask(setup.viewer, "How many earned leave days?", setup.config)
    assert len(pipeline.calls) == 2


def _stream(user, monkeypatch, question="How many earned leave days?"):
    calls = []

    def fake_stream_rag(question, *args, **kwargs):
        calls.append(question)
        yield "token", "Earned leave is 18 days [S1]. "
        yield "outcome", _answer(question)

    monkeypatch.setattr(chat_service, "stream_rag", fake_stream_rag)
    client = APIClient()
    client.force_authenticate(user)
    response = client.post(
        "/api/rag/query/stream/", {"question": question}, format="json", HTTP_ACCEPT="text/event-stream"
    )
    events = []
    for block in b"".join(response.streaming_content).decode().strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events, calls


def test_a_streamed_repeat_arrives_whole_and_says_so(setup, monkeypatch):
    first, calls = _stream(setup.viewer, monkeypatch)
    assert first[-1][1]["cached"] is False and first[-1][1]["timings_ms"]["first_token"] > 0
    second, calls = _stream(setup.viewer, monkeypatch)
    assert calls == []
    assert second[0] == ("text", {"text": "Earned leave is 18 days [S1]."})
    done = second[-1][1]
    assert done["cached"] is True and done["usage"]["model"] == ""
    saved = APIClient()
    saved.force_authenticate(setup.viewer)
    history = saved.get(f"/api/rag/sessions/{done['session']}/").json()
    assert history["messages"][0]["cached"] is True
    metrics = APIClient().get("/metrics").content.decode()
    assert 'rag_answer_cache_total{result="hit"}' in metrics


def test_writing_speed_leaves_out_the_time_spent_reading_the_prompt():
    llm = LLMResponse(
        text="x", model="m", prompt_tokens=900, completion_tokens=100, latency_ms=12000, first_token_ms=2000
    )
    assert _speed(llm) == 10.0
    assert _speed(None) == 0.0
    llm.first_token_ms = 0.0  # not streamed: the prompt's reading time cannot be separated out
    assert _speed(llm) == 0.0


def _config():
    return SimpleNamespace(
        retrieval_mode="hybrid",
        embedding_model="e",
        candidate_k=5,
        rrf_k=60,
        index_key="k",
        reranker="",
        use_mmr=False,
        top_k=5,
        rerank_top_n=5,
        rerank_min_score=None,
    )


def test_vector_and_keyword_search_run_at_the_same_time(setup, monkeypatch):
    rows = [
        DocumentChunk.objects.create(
            document=setup.document, index_key="k", chunk_index=i, chunk_text=f"Leave rule {i}.", token_count=3
        )
        for i in range(2)
    ]
    threads, both_started = {}, threading.Barrier(2, timeout=5)

    def vector(*args):
        threads["vector"] = threading.current_thread().name
        both_started.wait()  # passes only if the keyword search is running at the same moment
        return [Hit(rows[0].id, 0.9)]

    def keywords(*args):
        threads["bm25"] = threading.current_thread().name
        both_started.wait()
        return [Hit(rows[1].id, 3.0)]

    monkeypatch.setattr(hybrid_search.embeddings, "embed_query", lambda *args: [0.1])
    monkeypatch.setattr(hybrid_search.vector_search, "search", vector)
    monkeypatch.setattr(hybrid_search.bm25_search, "backend", lambda: SimpleNamespace(search=keywords))
    result = hybrid_search.retrieve("leave", _config(), [setup.document.id])
    assert threads["vector"].startswith("vector-search") and threads["bm25"] == threading.current_thread().name
    assert set(result.chunk_ids) == {rows[0].id, rows[1].id}
    assert result.stages["vector"][0]["chunk_id"] == rows[0].id and {"embed", "vector", "bm25"} <= set(
        result.timings_ms
    )


def _log(config, user, **fields):
    session = ChatSession.objects.create(user=user, title="t", rag_config=config)
    message = ChatMessage.objects.create(session=session, question="q", cached=fields.pop("cached", False))
    return PerformanceLog.objects.create(message=message, **fields)


def test_the_slo_report_judges_each_objective(setup, settings):
    settings.SLO = {"p95_ms": 20000, "first_token_p95_ms": 5000, "error_rate": 0.3, "tokens_per_second": 8.0}
    for _ in range(3):
        _log(setup.config, setup.viewer, total_ms=10000, first_token_ms=8000, tokens_per_second=12.0)
    _log(setup.config, setup.viewer, total_ms=0, cached=True, first_token_ms=50)
    client = APIClient()
    client.force_authenticate(setup.viewer)
    report = client.get("/api/monitoring/slo/?days=7").json()
    verdicts = {row["key"]: row["met"] for row in report["objectives"]}
    assert verdicts == {"p95_ms": True, "first_token_p95_ms": False, "error_rate": True, "tokens_per_second": True}
    assert (report["met"], report["judged"], report["cache_hit_rate"]) == (3, 4, 0.25)
    assert client.get("/api/monitoring/metrics/?days=7").json()["slo"]["judged"] == 4


def test_with_no_answers_nothing_is_judged(setup):
    client = APIClient()
    client.force_authenticate(setup.viewer)
    report = client.get("/api/monitoring/slo/").json()
    assert report["judged"] == 0 and all(row["met"] is None for row in report["objectives"])
