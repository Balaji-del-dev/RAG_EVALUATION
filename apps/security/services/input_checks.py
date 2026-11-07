"""Checks on a question before anything else reads it: hidden characters, malformed input, language and abuse.

Hidden characters matter because they hide attacks: "ig​nore all previous instructions", with a zero-width space,
slips past a pattern that looks for "ignore". So text is normalised first and every later check sees the clean form.
"""

import re
import unicodedata
from dataclasses import dataclass, field

from apps.security.services import pii_detector, toxicity

# Zero-width characters, word joiners, the byte order mark and bidirectional overrides, which can reorder what a
# reader sees without changing what a program reads.
HIDDEN_RE = re.compile("[​-‏‪-‮⁠-⁯﻿]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
WHITESPACE_RE = re.compile(r"[ \t]{2,}")
MAX_WORD_CHARS = 200
# The documents and the model work in English. Text in another script would be searched against nothing.
SUPPORTED_SCRIPTS = {"LATIN"}
# Input is blocked for severe abuse or two insults. One rude word in a real question ("is this policy dumb?") is not.
INPUT_TOXIC_THRESHOLD = 0.75
UNSUPPORTED_LANGUAGE_MESSAGE = (
    "Please ask in English. Your documents and the model work in English, so a question in another language "
    "would not find anything."
)


@dataclass
class InputReport:
    text: str
    findings: dict = field(default_factory=dict)


def normalise(text: str) -> InputReport:
    """NFKC form (so full-width and styled letters become plain ones), without hidden or control characters."""
    findings = {}
    clean = unicodedata.normalize("NFKC", text)
    if HIDDEN_RE.search(clean):
        findings["hidden_characters"] = len(HIDDEN_RE.findall(clean))
        clean = HIDDEN_RE.sub("", clean)
    if CONTROL_RE.search(clean):
        findings["control_characters"] = len(CONTROL_RE.findall(clean))
        clean = CONTROL_RE.sub("", clean)
    clean = WHITESPACE_RE.sub(" ", clean).strip()
    return InputReport(clean, findings)


def malformed_reason(text: str) -> str:
    """Why the text cannot be a question, or ""."""
    if not any(character.isalnum() for character in text):
        return "no_words"
    if any(len(word) > MAX_WORD_CHARS for word in text.split()):
        return "overlong_word"
    return ""


def script(text: str) -> str:
    """The writing system of most of the letters: LATIN, TAMIL, DEVANAGARI, CJK, ARABIC and so on."""
    counts: dict[str, int] = {}
    for character in text:
        if character.isalpha():
            name = unicodedata.name(character, "")
            key = "CJK" if name.startswith(("CJK", "HIRAGANA", "KATAKANA", "HANGUL")) else name.split(" ", 1)[0]
            counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get) if counts else "LATIN"


def is_supported_language(text: str) -> bool:
    return script(text) in SUPPORTED_SCRIPTS


def is_abusive(text: str) -> bool:
    return toxicity.toxicity_score(text) >= INPUT_TOXIC_THRESHOLD


def pii_types(text: str) -> list[str]:
    """Personal data in the question itself, recorded so that it can be masked in traces and logs."""
    return sorted({finding.entity_type for finding in pii_detector.detect_pii(text)})
