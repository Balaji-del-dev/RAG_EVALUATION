"""Break an answer into checkable claims. Rule-based, so the same answer always gives the same claims."""

import re
from dataclasses import dataclass, field

from apps.chatbot.services.generation import CITATION_RE

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
MARKDOWN_RE = re.compile(r"[*_`#]+")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,!?;:])")
MIN_CLAIM_WORDS = 3


@dataclass
class Claim:
    text: str
    markers: list[int] = field(default_factory=list)


def _remove_markers(text: str) -> str:
    """Drop [Sn] markers, and the space each one leaves before punctuation."""
    return SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", " ".join(CITATION_RE.sub("", text or "").split()))


def split_claims(text: str) -> list[Claim]:
    claims: list[Claim] = []
    for piece in SENTENCE_RE.split(text or ""):
        # "The policy includes the following:" introduces claims but states none, so it cannot be judged.
        if piece.rstrip().endswith(":"):
            continue
        markers = [int(m) for m in CITATION_RE.findall(piece)]
        clean = _remove_markers(MARKDOWN_RE.sub("", BULLET_RE.sub("", piece))).strip(" ,;:")
        if len(clean.split()) >= MIN_CLAIM_WORDS:
            claims.append(Claim(clean, markers))
        elif markers and claims:
            # A citation stranded after the full stop ("...24 days. [S1]") belongs to the previous claim.
            claims[-1].markers.extend(markers)
    return claims


def strip_citations(text: str) -> str:
    return _remove_markers(text)


def split_sentences(text: str) -> list[str]:
    return [" ".join(s.split()) for s in SENTENCE_RE.split(text or "") if len(s.split()) >= MIN_CLAIM_WORDS]
