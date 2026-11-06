"""Natural Language Inference: does a premise entail, contradict, or say nothing about a hypothesis?

This judges faithfulness, contradiction and citation support without an LLM. It is deterministic, costs
nothing, and takes milliseconds rather than the seconds an LLM judge needs per claim on CPU.
"""

import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from django.conf import settings

from apps.evaluation.services.claims import split_claims
from config.env import env

_lock = threading.Lock()
BATCH_SIZE = 16
WINDOW_SENTENCES = 3


@dataclass
class ClaimVerdict:
    claim: str
    entailment: float
    contradiction: float

    @property
    def supported(self) -> bool:
        return self.entailment >= settings.EVALUATION["ENTAILMENT_THRESHOLD"]

    @property
    def contradicted(self) -> bool:
        return not self.supported and self.contradiction >= settings.EVALUATION["CONTRADICTION_THRESHOLD"]


@lru_cache(maxsize=1)
def _load():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(env.NLI_MODEL, device="cpu")
    labels = {label.lower(): index for index, label in model.config.id2label.items()}
    return model, labels["entailment"], labels["contradiction"]


def _classify(pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    with _lock:
        model, entail_index, contra_index = _load()
    logits = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    return probabilities[:, entail_index], probabilities[:, contra_index]


def premise_windows(text: str) -> list[str]:
    """The texts a claim is judged against: each sentence alone, overlapping windows, and the whole source.

    Small NLI models get confused by long premises. A claim copied word for word scored 0.03 against a
    passage whose other sentences held different numbers, because those looked like contradictions, so
    each sentence is judged alone. The windows and whole text catch facts spread across sentences.
    """
    sentences = [claim.text for claim in split_claims(text)]
    windows = [" ".join(sentences[i : i + WINDOW_SENTENCES]) for i in range(len(sentences) - WINDOW_SENTENCES + 1)]
    return list(dict.fromkeys([*sentences, *windows, text]))


def judge_claims(claims: list[str], sources: list[str]) -> list[ClaimVerdict]:
    """For each claim, take the strongest entailment and strongest contradiction across all sources."""
    if not claims:
        return []
    premises = [window for source in sources for window in premise_windows(source)]
    if not premises:
        return [ClaimVerdict(claim, 0.0, 0.0) for claim in claims]
    pairs = [(premise, claim) for claim in claims for premise in premises]
    entailment, contradiction = _classify(pairs)
    size = len(premises)
    return [
        ClaimVerdict(
            claim,
            float(entailment[position * size : (position + 1) * size].max()),
            float(contradiction[position * size : (position + 1) * size].max()),
        )
        for position, claim in enumerate(claims)
    ]
