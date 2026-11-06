import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler

logger = logging.getLogger(__name__)


class ServiceUnavailable(Exception):
    """A backing service (Qdrant, OpenSearch, Ollama, MinIO) could not be reached."""


class InvalidDocument(Exception):
    """An uploaded file failed validation or could not be parsed."""


class InvalidDataset(Exception):
    """An uploaded evaluation dataset is malformed."""


def api_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is not None:
        return response
    if isinstance(exc, (InvalidDocument, InvalidDataset)):
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    if isinstance(exc, ServiceUnavailable):
        logger.warning("Service unavailable: %s", exc)
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    logger.exception("Unhandled API error", exc_info=exc)
    return Response({"error": "Internal server error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
