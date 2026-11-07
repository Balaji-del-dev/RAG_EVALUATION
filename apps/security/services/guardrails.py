"""The two guard checkpoints around the LLM: input validation and output filtering."""

from dataclasses import dataclass, field

from apps.common.constants import Role
from apps.common.permissions import has_role
from apps.security.services import pii_detector, prompt_injection, toxicity

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
    if not question.strip():
        return GuardResult(True, "empty_question", question)
    if len(question) > MAX_QUESTION_CHARS:
        return GuardResult(True, "question_too_long", question)
    verdict = prompt_injection.scan_input(question)
    if verdict.flagged:
        return GuardResult(True, verdict.category, question, {"rule": verdict.rule, "category": verdict.category})
    return GuardResult(False, "", question)


def check_output(answer: str) -> GuardResult:
    """Block toxic answers. Mask PII and secrets but keep the answer, so the rest of it stays useful."""
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
