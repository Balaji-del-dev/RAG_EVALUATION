"""Chat answers worth a second look, and new test cases made from them.

Real questions that went wrong are the best test cases there are: they are what people actually ask. An analyst looks
through the answers rated thumbs-down, flagged by the checks, declined, or failed, writes the correct answer for the
ones that matter, and saves them as a new version of a dataset. The next evaluation run then measures whether a change
fixed them.

Who asked is left out on purpose: judging an answer needs the question and the answer, not the person.
"""

import operator
from datetime import timedelta
from functools import reduce

from django.db.models import Exists, OuterRef, Q
from django.utils import timezone

from apps.chatbot.models import AnswerFeedback, ChatMessage
from apps.evaluation.models import EvaluationDataset
from apps.evaluation.services import dataset_manager

# Check results that point at a problem. "retrieval" only notes how the search worked around something.
PROBLEM_CHECKS = ["numbers", "missing_sources", "layout", "unsupported", "contradicted"]

KINDS = {
    "thumbs_down": Q(thumbs_down=True),
    "flagged": Q(checks__has_any_keys=PROBLEM_CHECKS),
    "declined": Q(abstained=True, blocked=False),
    "failed": Q(performance__succeeded=False),
}
LABELS = {
    "thumbs_down": "Rated not helpful",
    "flagged": "Flagged by the checks",
    "declined": "Declined",
    "failed": "Failed",
}


def _messages(days: int):
    since = timezone.now() - timedelta(days=days)
    down = AnswerFeedback.objects.filter(message=OuterRef("pk"), rating=AnswerFeedback.Rating.DOWN)
    return (
        ChatMessage.objects.filter(created_at__gte=since)
        .annotate(thumbs_down=Exists(down))
        .select_related("performance")
        .prefetch_related("feedback", "llm_runs")
    )


def counts(days: int) -> dict:
    messages = _messages(days)
    return {kind: messages.filter(rule).count() for kind, rule in KINDS.items()}


def queue(days: int, kind: str = ""):
    """Messages that need review, newest first: of one kind, or of any."""
    rule = KINDS.get(kind) or reduce(operator.or_, KINDS.values())
    return _messages(days).filter(rule).order_by("-created_at", "-id")


def _tags(message) -> list[str]:
    performance = getattr(message, "performance", None)
    found = {
        "thumbs_down": message.thumbs_down,
        "flagged": any(key in message.checks for key in PROBLEM_CHECKS),
        "declined": message.abstained and not message.blocked,
        "failed": performance is not None and not performance.succeeded,
    }
    return [kind for kind, present in found.items() if present]


def as_item(message) -> dict:
    feedback = next(iter(message.feedback.all()), None)
    run = next(iter(message.llm_runs.all()), None)
    return {
        "id": message.id,
        "created_at": message.created_at,
        "question": message.question,
        "standalone_question": message.standalone_question,
        "answer": message.answer,
        "tags": _tags(message),
        "feedback": {"rating": feedback.rating, "comment": feedback.comment} if feedback else None,
        "checks": {key: value for key, value in message.checks.items() if key in PROBLEM_CHECKS},
        "decline_reason": message.decline_reason,
        "model": run.model_name if run else "",
        "sources": sorted({citation["filename"] for citation in message.citations if citation.get("filename")}),
    }


def next_version(name: str) -> str:
    versions = EvaluationDataset.objects.filter(name=name).values_list("version", flat=True)
    numbers = [int(version) for version in versions if str(version).isdigit()]
    return str(max(numbers, default=0) + 1)


def create_dataset(name: str, items: list[dict], user) -> EvaluationDataset:
    """A new version of dataset `name`, one case per reviewed answer. Each case needs the correct answer, or to be
    marked as one the system should decline, as any dataset does. The full question is used, so a follow-up such as
    "and sick leave?" becomes a question that stands on its own."""
    messages = ChatMessage.objects.in_bulk([item["message_id"] for item in items])
    missing = [item["message_id"] for item in items if item["message_id"] not in messages]
    if missing:
        raise dataset_manager.InvalidDataset(f"No chat answer with id {', '.join(map(str, missing))}")
    cases = []
    for item in items:
        message = messages[item["message_id"]]
        cases.append(
            {
                "question": message.standalone_question or message.question,
                "reference_answer": item.get("reference_answer", ""),
                "answerable": item.get("answerable", True),
            }
        )
    count = len(cases)
    description = (
        f"From {count} reviewed chat {'answer' if count == 1 else 'answers'}, {timezone.localdate():%d %b %Y}."
    )
    return dataset_manager.create_dataset(
        name=name, version=next_version(name), description=description, cases=cases, user=user
    )
