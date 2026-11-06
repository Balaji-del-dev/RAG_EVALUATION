"""Document authorization. A prompt saying "do not reveal confidential data" is not access control:
what the user may not read must never reach the LLM."""

from apps.documents.models import Document


def allowed_document_ids(user) -> list[int]:
    return list(
        Document.objects.accessible_to(user).filter(status=Document.Status.READY).values_list("id", flat=True)
    )


def all_ready_document_ids() -> list[int]:
    """For evaluation runs only, which measure quality over the whole corpus. Never use for user traffic."""
    return list(Document.objects.filter(status=Document.Status.READY).values_list("id", flat=True))
