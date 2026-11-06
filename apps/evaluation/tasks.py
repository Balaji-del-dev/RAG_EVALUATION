import threading
import time

from celery import shared_task
from celery.signals import worker_ready
from django.core.cache import cache

from apps.common.constants import JobStatus
from apps.evaluation.models import EvaluationRun
from apps.evaluation.services.run_executor import execute_run

WORKER_HEARTBEAT_KEY = "worker_heartbeat"
HEARTBEAT_SECONDS = 10


@shared_task(bind=True, name="evaluation.run_evaluation", time_limit=4 * 3600)
def run_evaluation_task(self, run_id: int) -> None:
    execute_run(run_id, worker=self.request.hostname or "")


@worker_ready.connect
def resume_interrupted_runs(sender, **kwargs) -> None:
    """Pick up runs this worker was executing when it died, so they do not stay "running" forever.

    A worker that has just started cannot be executing anything, so a run still assigned to its name was
    interrupted. Runs on other workers are left alone, which keeps this safe with several workers.
    """
    for run in EvaluationRun.objects.filter(status=JobStatus.RUNNING, worker=sender.hostname):
        run_evaluation_task.delay(run.id)


@worker_ready.connect
def start_heartbeat(sender, **kwargs) -> None:
    """Let the dashboard tell whether a worker is alive.

    Celery's own ping cannot do this: a solo-pool worker does not answer it while a task is running,
    so a busy worker would look dead. This thread keeps beating during a task.
    """

    def beat():
        while True:
            cache.set(WORKER_HEARTBEAT_KEY, sender.hostname, HEARTBEAT_SECONDS * 3)
            time.sleep(HEARTBEAT_SECONDS)

    threading.Thread(target=beat, name="worker-heartbeat", daemon=True).start()
