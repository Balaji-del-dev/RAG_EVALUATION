"""Other wordings of a question, written by the model and searched alongside it. Off unless a configuration turns it
on (RagConfig.query_expansion): each costs a model call, several seconds on a CPU, and has to earn that in an
evaluation run against the same configuration without it.

- multi_query: the model writes two other phrasings. Each is searched by vector and keyword, and all the lists are
  fused. It helps when the asker's words differ from the document's ("time off" against "leave").
- hyde: the model writes a short passage that could answer the question, and that passage's embedding is searched
  instead of the question's. Keyword search still uses the question, so names and figures are matched as typed.

The reranker always scores chunks against the original question, so a wording the model got wrong can bring in
candidates but cannot decide the order. If the model call fails, the search runs without expansion and says so.
"""

import logging
import re
from dataclasses import dataclass, field

from apps.common.exceptions import ServiceUnavailable

logger = logging.getLogger(__name__)

MULTI_QUERY = "multi_query"
HYDE = "hyde"
VARIANTS = 2
MAX_VARIANT_WORDS = 30
MAX_PASSAGE_WORDS = 120

MULTI_QUERY_PROMPT = (
    "Rewrite the question below in {n} different ways, for searching company documents. Keep its meaning, names and "
    "figures; change the other words. Write one question per line, with no numbering and nothing else."
)
HYDE_PROMPT = (
    "Write two or three sentences that could appear in a company document and answer the question below. Write them "
    "as the document would, with no introduction. If you do not know the facts, write a plausible passage anyway: "
    "it is used only to search, and nobody reads it."
)
LEADING_MARK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|question\s*\d*:)\s*", re.IGNORECASE)


@dataclass
class Expansion:
    queries: list[str] = field(default_factory=list)  # multi_query: the other phrasings
    passage: str = ""  # hyde: the text whose embedding is searched
    llm: object = None  # the LLMResponse, so its time and tokens are counted
    note: str = ""  # why expansion was skipped, when it was


def _variants(text: str, query: str) -> list[str]:
    seen, found = {query.strip().lower()}, []
    for line in text.splitlines():
        line = LEADING_MARK_RE.sub("", line).strip().strip('"')
        if not line or len(line.split()) > MAX_VARIANT_WORDS or line.lower() in seen:
            continue
        seen.add(line.lower())
        found.append(line)
    return found[:VARIANTS]


def expand(query: str, config, mode: str) -> Expansion:
    from apps.chatbot.services import llm_gateway  # the chat app's gateway: retries, circuit breaker, fallbacks

    prompt = MULTI_QUERY_PROMPT.format(n=VARIANTS) if mode == MULTI_QUERY else HYDE_PROMPT
    try:
        llm = llm_gateway.complete(
            [{"role": "system", "content": prompt}, {"role": "user", "content": f"Question: {query}"}],
            model=config.llm_model,
            temperature=0.3 if mode == MULTI_QUERY else 0.0,
            max_tokens=120 if mode == MULTI_QUERY else 160,
        )
    except ServiceUnavailable as exc:
        logger.warning("Query expansion failed, searching without it: %s", exc)
        return Expansion(note="The model did not answer, so the question was searched without other wordings.")
    if mode == MULTI_QUERY:
        return Expansion(queries=_variants(llm.text, query), llm=llm)
    passage = " ".join(llm.text.split()[:MAX_PASSAGE_WORDS])
    return Expansion(passage=passage, llm=llm)
