"""PII detection and masking with Microsoft Presidio, plus rules for secrets and credentials."""

import logging
import re
import threading
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)
_lock = threading.Lock()

# PERSON is left out on purpose. Company documents name people all the time ("contact Priya in HR"),
# and masking every name would destroy answers. The identifiers below are the actual leakage risk.
PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
    "IP_ADDRESS",
    "IN_PAN",
    "IN_AADHAAR",
    "IN_BANK_ACCOUNT",
    "IN_UAN",
]
# A bare run of digits could be anything, so these score low on their own and only pass MIN_SCORE next to their label.
# A payslip showed a viewer the bank account and UAN in full, with only the PAN masked.
WEAK_NUMBER_SCORE = 0.1
MIN_SCORE = 0.4

SECRET_PATTERNS = {
    # Case-insensitive on purpose: LLMs change casing when they repeat text. A model turned "sk-..." into
    # "SK-..." in testing, and a case-sensitive pattern let the key straight through.
    "API_KEY": re.compile(r"(?i)\b(?:sk|pk|rk)[-_][a-z0-9][a-z0-9_-]{15,}\b"),
    "AWS_ACCESS_KEY": re.compile(r"(?i)\bAKIA[0-9A-Z]{16}\b"),
    "BEARER_TOKEN": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}\b"),
    # The value must look like a secret, so that "the password is required" stays readable. The optional
    # colon matters: without it, "password is: abc" matched "password is:" and left the real value showing.
    "PASSWORD_ASSIGNMENT": re.compile(
        r"(?i)\b(?:password|passwd|pwd)\s*(?:is\s*:?|[:=])\s*(?=\S*[0-9!@#$%^&*_-])\S{6,}"
    ),
    "PRIVATE_KEY": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


@dataclass
class Finding:
    entity_type: str
    start: int
    end: int
    score: float


@lru_cache(maxsize=1)
def _analyzer():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import InAadhaarRecognizer, InPanRecognizer

    # The small spaCy model: the default en_core_web_lg costs about 500 MB of RAM more.
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    )
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    # Presidio ships these Indian identifiers but does not load them by default.
    analyzer.registry.add_recognizer(InPanRecognizer())
    analyzer.registry.add_recognizer(InAadhaarRecognizer())
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="IN_BANK_ACCOUNT",
            patterns=[Pattern("account number", r"\b\d{9,18}\b", WEAK_NUMBER_SCORE)],
            context=["account", "a/c", "acct", "bank"],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="IN_UAN",
            patterns=[Pattern("universal account number", r"\b\d{12}\b", WEAK_NUMBER_SCORE)],
            context=["uan", "universal"],
        )
    )
    return analyzer


def detect_pii(text: str) -> list[Finding]:
    if not text.strip():
        return []
    with _lock:
        analyzer = _analyzer()
    results = analyzer.analyze(text=text, language="en", entities=PII_ENTITIES, score_threshold=MIN_SCORE)
    return [Finding(r.entity_type, r.start, r.end, r.score) for r in results]


def detect_secrets(text: str) -> list[Finding]:
    return [
        Finding(name, match.start(), match.end(), 1.0)
        for name, pattern in SECRET_PATTERNS.items()
        for match in pattern.finditer(text)
    ]


def mask(text: str, findings: list[Finding]) -> str:
    """Replace findings with [TYPE]. Overlapping findings are merged first: masking each one separately
    would leave part of the sensitive value visible. Replacing from the end keeps earlier offsets valid."""
    merged: list[Finding] = []
    for finding in sorted(findings, key=lambda f: (f.start, -f.end)):
        if merged and finding.start < merged[-1].end:
            merged[-1].end = max(merged[-1].end, finding.end)
        else:
            merged.append(Finding(finding.entity_type, finding.start, finding.end, finding.score))
    for finding in reversed(merged):
        text = f"{text[: finding.start]}[{finding.entity_type}]{text[finding.end :]}"
    return text
