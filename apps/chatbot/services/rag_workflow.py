"""The RAG workflow as a LangGraph state machine, shared by live chat and evaluation runs.

input_guard ──blocked──> END
     ├──small talk──> small_talk -> END
     │
  understand -> retrieve -> generate -> output_guard -> END

"understand" corrects spelling against the readable documents and completes a follow-up in a conversation. Evaluation
runs send single questions, so the follow-up part never runs for them.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from apps.chatbot.services import generation
from apps.chatbot.services.llm_gateway import LLMResponse
from apps.monitoring.services.tracing import record_generation, span
from apps.retrieval.models import RagConfig
from apps.retrieval.services.hybrid_search import RetrievalResult, RetrievedChunk, retrieve
from apps.security.services import guardrails


class RagState(TypedDict, total=False):
    question: str
    config: RagConfig
    allowed_document_ids: list[int]
    earlier_question: str | None
    earlier_answer: str
    stream: bool
    layout: str
    missing_months: list[str]
    standalone_question: str
    whole_document: bool
    rewrite_llm: LLMResponse
    rewrite_ms: float
    blocked: bool
    block_reason: str
    findings: dict
    retrieval: RetrievalResult
    retrieval_ms: float
    answer: str
    raw_answer: str
    abstained: bool
    decline_reason: str
    quoted: bool
    citations: list[dict]
    llm: LLMResponse | None
    generation_ms: float


def input_guard_node(state: RagState) -> RagState:
    with span("guard.input"):
        verdict = guardrails.check_input(state["question"])
    if verdict.blocked:
        return {
            "blocked": True,
            "block_reason": verdict.reason,
            "findings": {"input": verdict.findings},
            "answer": guardrails.INPUT_BLOCK_MESSAGE,
            "abstained": False,
            "citations": [],
        }
    return {"blocked": False, "findings": {}}


def small_talk_node(state: RagState) -> RagState:
    return {"answer": generation.small_talk_reply(state["question"]), "abstained": False, "citations": []}


def _after_input_guard(state: RagState) -> str:
    if state["blocked"]:
        return END
    return "small_talk" if generation.small_talk_reply(state["question"]) else "understand"


def understand_node(state: RagState) -> RagState:
    question, config, earlier = state["question"], state["config"], state.get("earlier_question")
    layout, without_format = generation.answer_format(question)
    # "download this as pdf" leaves nothing to search for: with a conversation, that means the previous answer.
    understood = generation.correct_spelling(
        without_format or question, config.index_key, state["allowed_document_ids"]
    )
    update: RagState = {}
    if earlier and layout and generation.only_asks_for_a_format(without_format):
        # "here i need this data html format pls": the previous question again, answered in the new layout.
        understood = generation.correct_spelling(earlier, config.index_key, state["allowed_document_ids"])
        return {"standalone_question": understood, "layout": layout}
    if earlier:
        understood = generation.resolve_time_reference(understood, earlier, state.get("earlier_answer", ""))
        started = time.perf_counter()
        with span("rag.understand"):
            understood, llm = generation.standalone_question(understood, earlier, config)
        update = {"rewrite_llm": llm, "rewrite_ms": (time.perf_counter() - started) * 1000}
    # This wording goes on to the answer step, so it has to pass the check the user's own words did.
    if understood != question and guardrails.check_input(understood).blocked:
        understood = question
    return {**update, "standalone_question": understood, "layout": layout}


def retrieve_node(state: RagState) -> RagState:
    started = time.perf_counter()
    with span("rag.retrieve"):
        question, allowed = state["standalone_question"], state["allowed_document_ids"]
        named = generation.documents_named(question, allowed)
        # Always a subset of what the user may read, so this can only narrow the search.
        scope = named or allowed
        months = generation.months_in(question)
        if months:
            # "LOP in May" must not be answered from the February payslip, which matches every other word.
            scope = generation.documents_mentioning(months, state["config"].index_key, scope)
        result = retrieve(question, state["config"], scope) if scope else RetrievalResult()
        whole_document = generation.asks_for_whole_document(question, named)
        if whole_document:
            # Search scores mean little for "summary of the brochure", so give the model the text in reading order.
            result.chunks.sort(key=lambda chunk: chunk.chunk_id)
    return {
        "retrieval": result,
        "whole_document": whole_document,
        "missing_months": generation.month_names(months) if months and not scope else "",
        "retrieval_ms": (time.perf_counter() - started) * 1000,
    }


def generate_node(state: RagState) -> RagState:
    result = state["retrieval"]
    # With no usable evidence, abstain without an LLM call. It is faster, and nothing can be made up.
    question, config = state["standalone_question"], state["config"]
    unknown = generation.unknown_terms(question, config.index_key, state["allowed_document_ids"])
    chunks, reason = result.chunks, ""
    if result.below_relevance_floor:
        chunks, reason = [], "Nothing in your documents was close enough to this question, so the model was not asked."
    elif unknown:
        names = ", ".join(f'"{term}"' for term in unknown)
        chunks, reason = [], f"No document you can read mentions {names}, so the model was not asked."
    elif state.get("missing_months"):
        chunks, reason = [], f"No document you can read covers {state['missing_months']}, so the model was not asked."
    elif not chunks:
        reason = "Nothing in the documents you can read matched this."
    on_token = None
    if state.get("stream"):
        write = get_stream_writer()
        on_token = lambda text: write({"token": text})  # noqa: E731
    started = time.perf_counter()
    with span("rag.generate"):
        generated = generation.generate(
            question, chunks, config, state.get("whole_document", False), on_token, state.get("layout", "")
        )
    # Never another month's figures in place of the one asked about.
    if not generated.abstained and (mismatch := generation.wrong_month(question, generated.text)):
        generated = generation.GeneratedAnswer(generation.ABSTENTION_TEXT, True, [], generated.llm)
        reason = mismatch
    return {
        "raw_answer": generated.text,
        "answer": generated.text,
        "abstained": generated.abstained,
        "decline_reason": (
            "The model kept repeating itself, so its answer was withheld."
            if generated.degenerate
            else reason or "The model read the sources and did not find an answer."
        )
        if generated.abstained
        else "",
        "quoted": generated.quoted,
        "citations": generated.citations,
        "llm": generated.llm,
        "generation_ms": (time.perf_counter() - started) * 1000,
    }


def output_guard_node(state: RagState) -> RagState:
    with span("guard.output"):
        verdict = guardrails.check_output(state["answer"])
    findings = dict(state.get("findings") or {})
    if verdict.findings:
        findings["output"] = verdict.findings
    update: RagState = {"answer": verdict.text, "findings": findings}
    if verdict.blocked:
        update.update(blocked=True, block_reason=verdict.reason, citations=[])
    return update


@lru_cache(maxsize=1)
def _graph():
    graph = StateGraph(RagState)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("small_talk", small_talk_node)
    graph.add_node("understand", understand_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("output_guard", output_guard_node)
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", _after_input_guard, {END: END, "small_talk": "small_talk", "understand": "understand"}
    )
    graph.add_edge("small_talk", END)
    graph.add_edge("understand", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "output_guard")
    graph.add_edge("output_guard", END)
    return graph.compile()


@dataclass
class RagOutcome:
    question: str
    answer: str
    # The complete question that was searched for, when a follow-up had to be rewritten. Otherwise empty.
    standalone_question: str = ""
    # The format the question asked for, a key of formats.BY_KEY, or "".
    answer_format: str = ""
    raw_answer: str = ""
    abstained: bool = False
    decline_reason: str = ""
    quoted: bool = False
    blocked: bool = False
    block_reason: str = ""
    findings: dict = field(default_factory=dict)
    citations: list[dict] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    stage_timings_ms: dict = field(default_factory=dict)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _inputs(question, config, allowed_document_ids, earlier_question, earlier_answer) -> RagState:
    return {
        "question": question,
        "config": config,
        "allowed_document_ids": allowed_document_ids,
        "earlier_question": earlier_question,
        "earlier_answer": earlier_answer,
    }


def run_rag(
    question: str,
    config: RagConfig,
    allowed_document_ids: list[int],
    earlier_question: str | None = None,
    earlier_answer: str = "",
) -> RagOutcome:
    started = time.perf_counter()
    with span("rag.pipeline", config=config.name):
        state = _graph().invoke(_inputs(question, config, allowed_document_ids, earlier_question, earlier_answer))
    return _outcome(question, config, state, started)


def stream_rag(
    question: str,
    config: RagConfig,
    allowed_document_ids: list[int],
    earlier_question: str | None = None,
    earlier_answer: str = "",
) -> Iterator[tuple[str, str | RagOutcome]]:
    """run_rag, yielding ("token", text) while the answer is written and then ("outcome", RagOutcome). The tokens are
    a preview: the checks after generation can still change the answer, and the outcome holds the final one."""
    started = time.perf_counter()
    state: RagState = {}
    inputs = {**_inputs(question, config, allowed_document_ids, earlier_question, earlier_answer), "stream": True}
    with span("rag.pipeline", config=config.name, streamed=True):
        for mode, data in _graph().stream(inputs, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield "token", data["token"]
            else:
                state = data
    yield "outcome", _outcome(question, config, state, started)


def _outcome(question: str, config: RagConfig, state: RagState, started: float) -> RagOutcome:
    retrieval: RetrievalResult = state.get("retrieval") or RetrievalResult()
    llm: LLMResponse | None = state.get("llm")
    rewrite: LLMResponse | None = state.get("rewrite_llm")
    # Both calls take time and cost tokens, and the records should say so.
    calls = [call for call in (rewrite, llm) if call]
    standalone = state.get("standalone_question", question)
    outcome = RagOutcome(
        question=question,
        answer=state["answer"],
        standalone_question=standalone if standalone != question else "",
        answer_format=state.get("layout", ""),
        raw_answer=state.get("raw_answer", ""),
        abstained=state.get("abstained", False),
        decline_reason=state.get("decline_reason", ""),
        quoted=state.get("quoted", False),
        blocked=state.get("blocked", False),
        block_reason=state.get("block_reason", ""),
        findings=state.get("findings", {}),
        citations=state.get("citations", []),
        chunks=retrieval.chunks,
        stages=retrieval.stages,
        stage_timings_ms=retrieval.timings_ms,
        retrieval_ms=state.get("retrieval_ms", 0.0),
        generation_ms=state.get("generation_ms", 0.0) + state.get("rewrite_ms", 0.0),
        total_ms=(time.perf_counter() - started) * 1000,
        model=calls[-1].model if calls else "",
        prompt_tokens=sum(call.prompt_tokens for call in calls),
        completion_tokens=sum(call.completion_tokens for call in calls),
    )
    if rewrite:
        record_generation(
            name="rag-rewrite",
            model=rewrite.model,
            question=question,
            answer=standalone,
            prompt_tokens=rewrite.prompt_tokens,
            completion_tokens=rewrite.completion_tokens,
            metadata={"config": config.name},
        )
    if llm:
        record_generation(
            name="rag-answer",
            model=llm.model,
            question=question,
            answer=outcome.answer,
            prompt_tokens=llm.prompt_tokens,
            completion_tokens=llm.completion_tokens,
            metadata={"config": config.name, "prompt_version": config.prompt_version, "chunk_ids": retrieval.chunk_ids},
        )
    return outcome
