"""How long one chat request may take in all. The model call gets whatever time is left, rather than its own full
timeout after a slow search, and a second attempt at an answer is skipped when it could not finish in time.

Evaluation runs set no deadline: a run measures quality, and each question there may take as long as it needs.
"""

import time
from contextlib import contextmanager
from contextvars import ContextVar

_deadline: ContextVar[float | None] = ContextVar("deadline", default=None)


@contextmanager
def within(seconds: float):
    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining() -> float | None:
    """Seconds left, or None when there is no deadline."""
    end = _deadline.get()
    return None if end is None else end - time.monotonic()
