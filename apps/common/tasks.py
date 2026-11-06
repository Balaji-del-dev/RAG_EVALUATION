from kombu.exceptions import OperationalError

from apps.common.exceptions import ServiceUnavailable


def enqueue(task, *args) -> None:
    """Queue a Celery task, turning a broker outage into a clean 503 instead of a long hang and a 500."""
    try:
        task.apply_async(args=args, retry=False)
    except OperationalError as exc:
        raise ServiceUnavailable(f"The task queue is unavailable: {exc}") from exc
