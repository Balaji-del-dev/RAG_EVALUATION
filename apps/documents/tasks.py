from celery import shared_task

from apps.common.exceptions import InvalidDocument, ServiceUnavailable
from apps.documents.services import ingestion


@shared_task(
    bind=True,
    name="documents.process_document",
    autoretry_for=(ServiceUnavailable,),
    retry_backoff=True,
    max_retries=3,
)
def process_document_task(self, document_id: int) -> None:
    try:
        ingestion.process_document(document_id)
    except InvalidDocument:
        # A corrupt file will not get better on retry; the failure is recorded on the document.
        return
