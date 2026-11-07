"""Batch 6 of the production pipeline: query expansion (multi-query, HyDE), the request deadline, trace IDs, and the
metrics that go with them."""

import re
from types import SimpleNamespace

import pytest
from rest_framework.test import APIClient

from apps.chatbot.services import generation, llm_gateway, rag_workflow
from apps.chatbot.services.llm_gateway import LLMResponse
from apps.common import deadline
from apps.common.exceptions import ServiceUnavailable
from apps.documents.models import Document, DocumentChunk
from apps.retrieval.services import hybrid_search, query_expansion
from apps.retrieval.services.hybrid_search import RetrievalResult, RetrievedChunk
from apps.retrieval.services.vector_search import Hit


def test_other_wordings_are_cleaned_up():
    text = "1. What are the time off rules?\n- How many holidays do I get?\nHow many leave days?\n3) A third one"
    assert query_expansion._variants(text, "How many leave days?") == [
        "What are the time off rules?",
        "How many holidays do I get?",
    ]


def _config(expansion):
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
        llm_model="fake",
        query_expansion=expansion,
    )


@pytest.fixture
def corpus(db, monkeypatch):
    document = Document.objects.create(filename="hr.docx", file_path="h", sha256="h", status="ready")
    rows = {
        name: DocumentChunk.objects.create(
            document=document, index_key="k", chunk_index=i, chunk_text=f"{name} text.", token_count=2
        )
        for i, name in enumerate(["leave", "holidays", "time off"])
    }
    searched = SimpleNamespace(vector=[], bm25=[])

    def embed(text, model):
        searched.vector.append(text)
        return [0.1]

    def vector(config, query_vector, k, allowed):
        return [Hit(rows["leave"].id, 0.9)]

    def keywords(config, text, k, allowed):
        searched.bm25.append(text)
        return [Hit(rows[name].id, 2.0) for name in rows if name in text.lower()]

    monkeypatch.setattr(hybrid_search.embeddings, "embed_query", embed)
    monkeypatch.setattr(hybrid_search.vector_search, "search", vector)
    monkeypatch.setattr(hybrid_search.bm25_search, "backend", lambda: SimpleNamespace(search=keywords))
    return SimpleNamespace(document=document, rows=rows, searched=searched)


def _model_writes(monkeypatch, text):
    calls = []

    def complete(messages, **kwargs):
        calls.append(messages)
        return LLMResponse(text, "fake", 30, 12, 800.0)

    monkeypatch.setattr(llm_gateway, "complete", complete)
    return calls


def test_multi_query_searches_each_wording_and_fuses_them(corpus, monkeypatch):
    _model_writes(monkeypatch, "What are the holidays?\nHow does time off work?")
    result = hybrid_search.retrieve("How many leave days?", _config("multi_query"), [corpus.document.id])
    assert corpus.searched.bm25 == ["How many leave days?", "What are the holidays?", "How does time off work?"]
    # "time off" was found only by a wording the model wrote.
    assert corpus.rows["time off"].id in result.chunk_ids and corpus.rows["holidays"].id in result.chunk_ids
    assert result.expansions == ["What are the holidays?", "How does time off work?"]
    assert result.expansion_llm.completion_tokens == 12 and "expand" in result.timings_ms


def test_hyde_searches_the_passage_by_vector_and_the_question_by_keyword(corpus, monkeypatch):
    _model_writes(monkeypatch, "Employees get 18 days of earned leave a year.")
    hybrid_search.retrieve("How many leave days?", _config("hyde"), [corpus.document.id])
    assert corpus.searched.vector == ["Employees get 18 days of earned leave a year."]
    assert corpus.searched.bm25 == ["How many leave days?"]


def test_without_expansion_nothing_extra_is_asked(corpus, monkeypatch):
    calls = _model_writes(monkeypatch, "unused")
    hybrid_search.retrieve("How many leave days?", _config(""), [corpus.document.id])
    assert calls == [] and corpus.searched.bm25 == ["How many leave days?"]


def test_a_failed_expansion_searches_the_question_alone(corpus, monkeypatch):
    def down(*args, **kwargs):
        raise ServiceUnavailable("Ollama is down")

    monkeypatch.setattr(llm_gateway, "complete", down)
    result = hybrid_search.retrieve("How many leave days?", _config("multi_query"), [corpus.document.id])
    assert result.chunk_ids and corpus.searched.bm25 == ["How many leave days?"]
    assert "without other wordings" in result.notes[0]


def test_the_expansion_call_is_counted_in_the_answer(db, monkeypatch):
    chunk = RetrievedChunk(1, 1, "hr.docx", 1, "Earned leave is 18 days.", 2.0)
    expansion = LLMResponse("other wordings", "fake", 30, 12, 800.0)
    monkeypatch.setattr(
        rag_workflow,
        "retrieve",
        lambda *args, **kwargs: RetrievalResult(chunks=[chunk], expansion_llm=expansion),
    )
    monkeypatch.setattr(
        generation.llm_gateway, "complete", lambda messages, **kwargs: LLMResponse("18 days [S1].", "fake", 100, 5, 1.0)
    )
    config = SimpleNamespace(
        name="c", index_key="k", llm_model="fake", prompt_version="v3", temperature=0.0, max_tokens=64
    )
    Document.objects.create(filename="hr.docx", file_path="h", sha256="h", status="ready")
    outcome = rag_workflow.run_rag("how many earned leave days?", config, [1])
    assert (outcome.prompt_tokens, outcome.completion_tokens) == (130, 17)


@pytest.fixture
def litellm_kwargs(monkeypatch):
    import litellm

    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        message = SimpleNamespace(content="Hello")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        )

    monkeypatch.setattr(litellm, "completion", completion)
    return seen


def test_the_model_call_gets_only_the_time_left(litellm_kwargs):
    with deadline.within(40):
        llm_gateway.complete([{"role": "user", "content": "x"}], model="ollama_chat/test")
    assert 39 < litellm_kwargs[0]["timeout"] <= 40
    with deadline.within(0.2), pytest.raises(ServiceUnavailable, match="ran out of time"):
        llm_gateway.complete([{"role": "user", "content": "x"}], model="ollama_chat/test")
    assert len(litellm_kwargs) == 1  # the second call was not even tried
    llm_gateway.complete([{"role": "user", "content": "x"}], model="ollama_chat/test")
    assert litellm_kwargs[1]["timeout"] == llm_gateway.env.LLM_TIMEOUT_SECONDS  # no deadline: evaluation runs


def test_a_second_attempt_is_skipped_when_it_could_not_finish(db, monkeypatch):
    document = Document.objects.create(filename="hr.docx", file_path="h", sha256="h", status="ready")
    DocumentChunk.objects.create(
        document=document, index_key="k", chunk_index=0, chunk_text="Sick leave is 12 days.", token_count=5
    )
    chunk = RetrievedChunk(1, document.id, "hr.docx", 1, "Sick leave is 12 days.", 2.0)
    monkeypatch.setattr(rag_workflow, "retrieve", lambda *args, **kwargs: RetrievalResult(chunks=[chunk]))
    calls = []
    # The first attempt took 20 s; only 5 s are left, so a second one would not finish.
    monkeypatch.setattr(
        generation.llm_gateway,
        "complete",
        lambda messages, **kwargs: (
            calls.append(1) or LLMResponse("Sick leave is 10 days [S1].", "fake", 10, 5, 20000.0)
        ),
    )
    config = SimpleNamespace(
        name="c", index_key="k", llm_model="fake", prompt_version="v3", temperature=0.0, max_tokens=64
    )
    with deadline.within(5):
        outcome = rag_workflow.run_rag("sick leave days?", config, [document.id], interactive=True)
    assert len(calls) == 1 and outcome.checks == {"numbers": ["10"]}


@pytest.mark.django_db
def test_every_response_carries_a_trace_id(django_user_model):
    client = APIClient()
    own = client.get("/api/accounts/csrf/")
    assert re.fullmatch(r"[0-9a-f]{32}", own["X-Trace-ID"])
    # A caller's W3C trace is continued, so its spans and ours join up.
    parent = "4bf92f3577b34da6a3ce929d0e0e4736"
    joined = client.get("/api/accounts/csrf/", HTTP_TRACEPARENT=f"00-{parent}-00f067aa0ba902b7-01")
    assert joined["X-Trace-ID"] == parent


def test_only_question_traffic_is_sent_to_langfuse():
    from apps.monitoring.services import tracing

    passed = []
    inner = SimpleNamespace(on_end=passed.append, on_start=lambda *a, **k: None)
    keep = tracing.QuestionsOnly(inner)
    poll = SimpleNamespace(name="http.request", attributes={"http.target": "/api/rag/sessions/"})
    chat = SimpleNamespace(name="http.request", attributes={"http.target": "/api/rag/query/stream/"})
    stage = SimpleNamespace(name="retrieval.bm25", attributes={})
    for item in (poll, chat, stage):
        keep.on_end(item)
    assert passed == [chat, stage]  # the dashboard's polling is dropped; questions and their stages are kept


def test_a_model_call_is_marked_as_a_generation_without_its_prompt():
    import json
    from datetime import UTC, datetime

    from apps.monitoring.services import tracing

    attributes = {}
    current = SimpleNamespace(set_attribute=attributes.__setitem__)
    first = datetime(2026, 11, 7, 16, 0, 5, tzinfo=UTC)
    tracing.mark_generation(
        current, model="ollama_chat/qwen2.5:1.5b", prompt_tokens=586, completion_tokens=32, completion_start=first
    )
    assert attributes["langfuse.observation.type"] == "generation"
    assert attributes["langfuse.observation.model.name"] == "ollama_chat/qwen2.5:1.5b"
    assert json.loads(attributes["langfuse.observation.usage_details"]) == {"input": 586, "output": 32}
    assert attributes["langfuse.observation.completion_start_time"] == "2026-11-07T16:00:05+00:00"
    assert not any(key.rsplit(".", 1)[-1] == "input" for key in attributes)  # no prompt, which holds source text


def test_tests_never_trace_into_the_real_langfuse():
    from apps.monitoring.services import tracing

    assert not tracing.langfuse_enabled()
