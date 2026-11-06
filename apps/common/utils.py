import math
import re
import string

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)
ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = str.maketrans("", "", string.punctuation)


def count_tokens(text: str) -> int:
    """Approximate token count: words plus punctuation. Fast, and needs no tokenizer download."""
    return len(TOKEN_RE.findall(text))


def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation and articles, collapse whitespace."""
    text = text.lower().translate(PUNCTUATION)
    return " ".join(ARTICLES_RE.sub(" ", text).split())


def word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def percentile(values: list[float], pct: float) -> float:
    """Percentile with linear interpolation between closest ranks."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default
