"""Checks on a finished answer against the sources it was given, shown under the answer so the reader knows what to
double-check.

Asked for a comparison table of leave types, the small model wrote "Annual leave: 5 days" when the sources say 18. A
figure that appears in no source is the cheapest hallucination to catch, so every answer gets that check, and a
citation to a source it was not given. Checking every sentence with the NLI model catches more, but loads about 400 MB
into the web server, so it is off unless ONLINE_NLI_CHECK is set.
"""

import re

from apps.chatbot.services import formats
from config.env import env

CITATION_RE = re.compile(r"\[S(\d+)\]")
NUMBER_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")
LIST_NUMBER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
ROW_NUMBER_RE = re.compile(r"^\s*\|\s*\d+\s*\|", re.MULTILINE)
CODE_RE = re.compile(r"```.*?```", re.DOTALL)
TABLE_FORMATS = {"html_table", "searchable_table"}
LIST_FORMATS = {"bullets", "numbered", "nested", "steps", "checklist"}
MAX_REPORTED = 6
MAX_SENTENCES = 8
# A sentence ends at . ! or ?, perhaps after its citation: "18 days [S1]." or "18 days. [S1]"
SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s*\[S\d+\])*\s+|\n+")


def _normal(number: str) -> str:
    number = number.replace(",", "")
    return number.rstrip("0").rstrip(".") if "." in number else number


def numbers_in(text: str) -> set[str]:
    return {_normal(found) for found in NUMBER_RE.findall(CITATION_RE.sub(" ", text))}


def unverified_numbers(answer: str, sources: list[str], question: str) -> list[str]:
    """Figures in the answer that appear in none of its sources, nor in the question. List and row numbers that the
    layout adds ("1.", "| 1 |") are not figures."""
    text = CITATION_RE.sub(" ", CODE_RE.sub(" ", answer))
    text = ROW_NUMBER_RE.sub("|", LIST_NUMBER_RE.sub(" ", text))
    known = numbers_in(" ".join(sources)) | numbers_in(question)
    unknown = [found for found in NUMBER_RE.findall(text) if _normal(found) not in known]
    return list(dict.fromkeys(unknown))[:MAX_REPORTED]


def layout_problem(answer: str, layout: str) -> str:
    fmt = formats.BY_KEY.get(layout)
    if not fmt:
        return ""
    if (fmt.group == "table" or fmt.key in TABLE_FORMATS) and not formats.markdown_tables(answer):
        return f"A {fmt.label.lower()} was asked for, but the answer has no table."
    if fmt.key in LIST_FORMATS and not any(formats.ITEM_RE.match(line) for line in answer.splitlines()):
        return f"A {fmt.label.lower()} was asked for, but the answer has no list."
    return ""


def _sentence_checks(answer: str, chunks) -> dict:
    """Each sentence judged by the NLI model against the sources it cites, or all of them if it cites none."""
    from apps.evaluation.services.nli import judge_claims

    prose = "\n".join(line for line in CODE_RE.sub(" ", answer).splitlines() if not line.strip().startswith("|"))
    # Split here rather than with the evaluation's claim splitter, which drops the [S1] markers this needs.
    sentences = [part.strip() for part in SENTENCE_RE.split(prose) if part.strip()]
    unsupported, contradicted = [], []
    for claim in sentences[:MAX_SENTENCES]:
        if len(claim.split()) < 4:
            continue
        cited = [
            chunks[int(number) - 1].text for number in CITATION_RE.findall(claim) if 0 < int(number) <= len(chunks)
        ]
        plain = re.sub(r"\s+([.!?,;:])", r"\1", CITATION_RE.sub("", claim)).strip()
        verdict = judge_claims([plain], cited or [chunk.text for chunk in chunks])[0]
        if verdict.contradicted:
            contradicted.append(verdict.claim)
        elif not verdict.supported:
            unsupported.append(verdict.claim)
    return {
        **({"unsupported": unsupported[:MAX_REPORTED]} if unsupported else {}),
        **({"contradicted": contradicted[:MAX_REPORTED]} if contradicted else {}),
    }


def check(answer: str, chunks, question: str, citations: list[dict], layout: str = "") -> dict:
    """What a reader should double-check in this answer. Empty when nothing was found."""
    found = {}
    if numbers := unverified_numbers(answer, [chunk.text for chunk in chunks], question):
        found["numbers"] = numbers
    if missing := [citation["marker"] for citation in citations if not citation.get("chunk_id")]:
        found["missing_sources"] = missing
    if problem := layout_problem(answer, layout):
        found["layout"] = problem
    if env.ONLINE_NLI_CHECK and chunks:
        found.update(_sentence_checks(answer, chunks))
    return found
