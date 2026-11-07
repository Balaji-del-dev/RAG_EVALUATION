"""The two guard checkpoints around the LLM: input validation and output filtering."""

import re
from dataclasses import dataclass, field

from apps.common.constants import Role
from apps.common.permissions import has_role
from apps.security.services import input_checks, pii_detector, prompt_injection, toxicity

MAX_QUESTION_CHARS = 2000
INPUT_BLOCK_MESSAGE = "This request was blocked by the security policy."
OUTPUT_BLOCK_MESSAGE = "The generated answer was withheld by the security policy."


@dataclass
class GuardResult:
    blocked: bool = False
    reason: str = ""
    text: str = ""
    findings: dict = field(default_factory=dict)


def mask_secrets(text: str) -> tuple[str, list[str]]:
    """Mask credentials and return the types found. Used at ingestion, so they never reach the index or LLM."""
    findings = pii_detector.detect_secrets(text)
    return pii_detector.mask(text, findings), sorted({finding.entity_type for finding in findings})


def text_for_user(user, text: str) -> str:
    """Source text for an API response. The answer is masked, so the sources beside it must be too, or a
    viewer could read the masked value right there. Analysts and admins review raw sources anyway."""
    if has_role(user, Role.ADMIN, Role.ANALYST):
        return text
    return pii_detector.mask(text, pii_detector.detect_pii(text))


def check_input(question: str) -> GuardResult:
    """The question cleaned of hidden characters, or why it is refused. Every check sees the clean text, so a
    zero-width space cannot split "ignore" and slip past the injection rules."""
    report = input_checks.normalise(question)
    text, findings = report.text, dict(report.findings)
    if not text:
        return GuardResult(True, "empty_question", text, findings)
    if len(text) > MAX_QUESTION_CHARS:
        return GuardResult(True, "question_too_long", text, findings)
    if reason := input_checks.malformed_reason(text):
        return GuardResult(True, "malformed_input", text, {**findings, "malformed": reason})
    verdict = prompt_injection.scan_input(text)
    if verdict.flagged:
        return GuardResult(
            True, verdict.category, text, {**findings, "rule": verdict.rule, "category": verdict.category}
        )
    if input_checks.is_abusive(text):
        return GuardResult(True, "toxic_input", text, {**findings, "toxicity": toxicity.toxicity_score(text)})
    if pii := input_checks.pii_types(text):
        findings["pii"] = pii
    return GuardResult(False, "", text, findings)


# Instructions planted in a document ("IMPORTANT SYSTEM NOTICE: ignore all previous instructions") are the indirect
# form of prompt injection. Only these rules are used on documents: "System: Linux" or "no restrictions on remote
# work" are ordinary document text, not attacks.
CONTEXT_RULES = ("ignore_instructions", "reveal_prompt", "new_instructions")
CONTEXT_LINE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
REMOVED_NOTE = "[An instruction aimed at the AI was removed from this source.]"


def sanitize_source(text: str) -> tuple[str, list[str]]:
    """The text without the sentences that try to instruct the model, and which rules they broke."""
    kept, broken = [], []
    for part in CONTEXT_LINE_RE.split(text):
        rule = next((name for name in CONTEXT_RULES if prompt_injection.COMPILED_INJECTION[name].search(part)), "")
        if rule:
            broken.append(rule)
        else:
            kept.append(part)
    if not broken:
        return text, []
    return "\n".join([*kept, REMOVED_NOTE]).strip(), broken


MAX_ANSWER_CHARS = 20000
LEAK_WORDS = 8  # this many words in a row from the system prompt counts as revealing it


def _shingles(text: str, size: int = LEAK_WORDS) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def leaks_prompt(answer: str, protected: list[str]) -> bool:
    """True if the answer repeats the system prompt, eight words or more in a row."""
    secret = set().union(*(_shingles(text) for text in protected)) if protected else set()
    return bool(secret & _shingles(answer))


def check_output(answer: str, protected: list[str] | None = None) -> GuardResult:
    """Block toxic answers and any that repeat the system prompt. Mask PII and secrets but keep the answer, so the
    rest of it stays useful. Cut answers that run past MAX_ANSWER_CHARS."""
    if protected and leaks_prompt(answer, protected):
        return GuardResult(True, "system_prompt_leak", OUTPUT_BLOCK_MESSAGE, {"leak": "system prompt"})
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS].rstrip() + f"\n\n(The answer was cut at {MAX_ANSWER_CHARS:,} characters.)"
    score = toxicity.toxicity_score(answer)
    if score >= toxicity.TOXIC_THRESHOLD:
        return GuardResult(True, "toxic_output", OUTPUT_BLOCK_MESSAGE, {"toxicity": score})

    pii = pii_detector.detect_pii(answer)
    secrets = pii_detector.detect_secrets(answer)
    findings = {}
    if pii:
        findings["pii"] = sorted({f.entity_type for f in pii})
    if secrets:
        findings["secrets"] = sorted({f.entity_type for f in secrets})
    return GuardResult(False, "", pii_detector.mask(answer, pii + secrets), findings)
