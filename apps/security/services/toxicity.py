"""Lexicon-based toxicity check on model output.

A word list is a baseline, not a classifier: it misses implied insults. A transformer model such as
Detoxify is more accurate but adds several hundred MB of RAM, too much for the current machine. To
swap one in, keep the `toxicity_score` signature.
"""

import re

SEVERE = r"\b(kill yourself|kys|subhuman|vermin|nazi scum)\b"
INSULTS = (
    r"\b(idiot|idiots|stupid|moron|morons|imbecile|dumb|retard|retarded|loser|losers|worthless|pathetic|"
    r"trash|scum|shut up|hate you|fuck\w*|shit\w*|bitch\w*|asshole\w*|bastard\w*)\b"
)
SEVERE_RE = re.compile(SEVERE, re.IGNORECASE)
INSULTS_RE = re.compile(INSULTS, re.IGNORECASE)
TOXIC_THRESHOLD = 0.5


def toxicity_score(text: str) -> float:
    """1.0 for severe abuse. Otherwise 0.5 for one insulting term and 0.25 more per further term, capped at 1."""
    if SEVERE_RE.search(text):
        return 1.0
    hits = len(INSULTS_RE.findall(text))
    return min(1.0, 0.25 + 0.25 * hits) if hits else 0.0


def is_toxic(text: str) -> bool:
    return toxicity_score(text) >= TOXIC_THRESHOLD
