"""Batch 2 of the production pipeline: checks on each answer, LLM retries and circuit breaker, search fallbacks."""

from types import SimpleNamespace

import pytest

from apps.chatbot.services import llm_gateway, verification
from apps.common.exceptions import ServiceUnavailable
from apps.retrieval.services import hybrid_search
from apps.retrieval.services.hybrid_search import RetrievalResult, RetrievedChunk

SOURCES = ["Earned / Privileged Leave (EL) | 18 Days | Max 30 Days.", "Maternity Leave | 26 Weeks."]


def test_a_figure_in_no_source_is_reported():
    """The real reply: a comparison table gave annual leave as 5 days; the sources say 18."""
    answer = (
        "| # | Leave | Days | Source |\n|---|---|---|---|\n| 1 | Annual | 5 Days | [S1] |\n| 2 | Earned | 18 | [S1] |"
    )
    assert verification.unverified_numbers(answer, SOURCES, "leave types") == ["5"]


def test_layout_numbers_code_and_the_questions_own_figures_are_not_reported():
    answer = "1. Earned leave is 18 days [S1].\n2. Carry up to 30 days [S1].\n```python\nx = 7\n```"
    assert verification.unverified_numbers(answer, SOURCES, "what about 2026?") == []
    assert verification.unverified_numbers("In 2026 it is 18 days.", SOURCES, "leave in 2026") == []
    assert verification.unverified_numbers("It is 18.00 days or 19 weeks.", SOURCES, "") == ["19"]  # 18.00 is 18


def _chunk(text, number=1):
    return RetrievedChunk(number, 1, "hr.docx", 1, text, 1.0)


def test_citation_to_a_source_it_was_not_given():
    citations = [{"marker": "S1", "chunk_id": 1}, {"marker": "S7", "chunk_id": None}]
    assert verification.check("Earned leave is 18 days [S1] [S7].", [_chunk(SOURCES[0])], "", citations) == {
        "missing_sources": ["S7"]
    }


@pytest.mark.parametrize(
    "layout,answer,problem",
    [
        ("table", "Earned leave is 18 days [S1].", "A table was asked for, but the answer has no table."),
        ("bullets", "Earned leave is 18 days [S1].", "A bullet list was asked for, but the answer has no list."),
        ("bullets", "- Earned leave is 18 days [S1].", ""),
        ("", "Earned leave is 18 days [S1].", ""),
    ],
)
def test_a_layout_that_was_asked_for_but_not_given(layout, answer, problem):
    assert verification.layout_problem(answer, layout) == problem


def test_the_sentence_check_runs_only_when_turned_on(monkeypatch):
    from apps.evaluation.services import nli

    judged = []

    def judge(claims, sources):
        judged.append((claims, sources))
        return [SimpleNamespace(claim=claims[0], supported=False, contradicted="5 days" in claims[0])]

    monkeypatch.setattr(nli, "judge_claims", judge)
    answer = "Annual leave is 5 days for everyone [S1]. Maternity leave lasts 26 weeks in total."
    monkeypatch.setattr(verification.env, "ONLINE_NLI_CHECK", False)
    assert "unsupported" not in verification.check(answer, [_chunk(SOURCES[0]), _chunk(SOURCES[1], 2)], "", [])
    monkeypatch.setattr(verification.env, "ONLINE_NLI_CHECK", True)
    found = verification.check(answer, [_chunk(SOURCES[0]), _chunk(SOURCES[1], 2)], "", [])
    assert found["contradicted"] == ["Annual leave is 5 days for everyone."]
    assert found["unsupported"] == ["Maternity leave lasts 26 weeks in total."]
    assert judged[0][1] == [SOURCES[0]]  # a cited sentence is judged against the source it cites
    assert judged[1][1] == SOURCES  # an uncited one against all of them


@pytest.fixture
def litellm_calls(monkeypatch):
    """litellm.completion replaced: each item in `script` is either a reply or an exception to raise."""
    import litellm

    state = {"script": [], "calls": 0, "slept": []}

    def completion(**kwargs):
        state["calls"] += 1
        step = state["script"].pop(0)
        if isinstance(step, Exception):
            raise step
        message = SimpleNamespace(content=step)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2)
        )

    monkeypatch.setattr(litellm, "completion", completion)
    monkeypatch.setattr(llm_gateway.time, "sleep", lambda seconds: state["slept"].append(seconds))
    return state


def _ask():
    return llm_gateway.complete([{"role": "user", "content": "x"}], model="ollama_chat/test")


def test_a_failed_call_is_retried_with_backoff(litellm_calls):
    litellm_calls["script"] = [ConnectionError("refused"), "Hello"]
    assert _ask().text == "Hello"
    assert litellm_calls["slept"] == [1.0]


def test_a_timeout_is_not_retried(litellm_calls):
    class Timeout(Exception):
        pass

    litellm_calls["script"] = [Timeout("Request timed out")]
    with pytest.raises(ServiceUnavailable):
        _ask()
    assert litellm_calls["calls"] == 1


def test_three_failures_open_the_circuit_and_calls_then_fail_fast(litellm_calls):
    litellm_calls["script"] = [ConnectionError("refused")] * 9
    for _ in range(3):
        with pytest.raises(ServiceUnavailable):
            _ask()
    calls = litellm_calls["calls"]
    with pytest.raises(ServiceUnavailable, match="pause for 30 seconds"):
        _ask()
    assert litellm_calls["calls"] == calls  # the model was not even tried


def _config(mode="hybrid"):
    return SimpleNamespace(
        retrieval_mode=mode,
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


@pytest.mark.django_db
def test_a_failed_vector_index_falls_back_to_keywords(monkeypatch):
    from apps.documents.models import Document, DocumentChunk
    from apps.retrieval.services.vector_search import Hit

    document = Document.objects.create(filename="hr.docx", file_path="h", sha256="h", status="ready")
    row = DocumentChunk.objects.create(
        document=document, index_key="k", chunk_index=0, chunk_text="Leave is 18 days.", token_count=4
    )
    monkeypatch.setattr(hybrid_search.embeddings, "embed_query", lambda *args: [0.1])
    monkeypatch.setattr(hybrid_search.time, "sleep", lambda seconds: None)

    def down(*args):
        raise ServiceUnavailable("Qdrant search failed")

    monkeypatch.setattr(hybrid_search.vector_search, "search", down)
    monkeypatch.setattr(
        hybrid_search.bm25_search, "backend", lambda: SimpleNamespace(search=lambda *a: [Hit(row.id, 2.0)])
    )
    result = hybrid_search.retrieve("leave", _config(), [document.id])
    assert [chunk.chunk_id for chunk in result.chunks] == [row.id]
    assert result.notes == ["The vector index did not answer, so this used keyword search only."]
    with pytest.raises(ServiceUnavailable):
        hybrid_search.retrieve("leave", _config("vector"), [document.id])  # nothing to fall back to


def _result(best):
    return RetrievalResult(stages={"rerank": [{"chunk_id": 1, "score": best}]})


def test_a_weak_search_is_widened_in_the_chat_only(monkeypatch):
    calls = []

    def once(query, config, ids):
        calls.append(config.candidate_k)
        return _result(-2.0 if len(calls) == 1 else 1.5)

    monkeypatch.setattr(hybrid_search, "_retrieve_once", once)
    config = SimpleNamespace(candidate_k=12, rerank_top_n=8)
    assert hybrid_search.retrieve("q", config, [1]).stages["rerank"][0]["score"] == -2.0  # evaluation: as it was
    calls.clear()
    widened = hybrid_search.retrieve("q", config, [1], widen_if_weak=True)
    assert calls == [12, 24] and widened.stages["rerank"][0]["score"] == 1.5
    assert "widened to 24 candidates" in widened.notes[0]


def test_a_good_search_is_not_widened(monkeypatch):
    calls = []
    monkeypatch.setattr(hybrid_search, "_retrieve_once", lambda *args: calls.append(1) or _result(4.0))
    hybrid_search.retrieve("q", SimpleNamespace(candidate_k=12, rerank_top_n=8), [1], widen_if_weak=True)
    assert len(calls) == 1
