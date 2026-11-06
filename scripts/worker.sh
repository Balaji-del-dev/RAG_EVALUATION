#!/usr/bin/env bash
# Runs the Celery worker under a memory guard, so that a tight machine pauses the work instead of
# having the kernel kill other applications.
#
# When free memory falls below WORKER_MIN_FREE_MB, the worker is stopped at once. It starts again when
# memory recovers to WORKER_RESUME_FREE_MB, and interrupted runs continue from their last saved answer.
set -uo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

MIN_FREE_MB="${WORKER_MIN_FREE_MB:-200}"
RESUME_FREE_MB="${WORKER_RESUME_FREE_MB:-1000}"
WORKER=""

# If the machine still runs out of memory, the kernel should kill this process, not the editor or browser.
echo 1000 > /proc/self/oom_score_adj

free_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

stop_worker() {
  [ -n "$WORKER" ] || return 0
  # SIGQUIT, not SIGTERM: a Celery warm shutdown waits for the current task and frees nothing.
  kill -QUIT "$WORKER" 2>/dev/null
  for _ in 1 2 3 4 5; do
    kill -0 "$WORKER" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$WORKER" 2>/dev/null
  wait "$WORKER" 2>/dev/null
  WORKER=""
}

trap 'stop_worker; exit 0' INT TERM

while true; do
  # The solo pool keeps a single copy of the ML models in RAM.
  celery -A config worker -n rageval@%h -Q rageval --pool=solo --loglevel=INFO --without-mingle --without-gossip &
  WORKER=$!

  while kill -0 "$WORKER" 2>/dev/null; do
    if [ "$(free_mb)" -lt "$MIN_FREE_MB" ]; then
      echo "[worker.sh] Only $(free_mb) MB free. Stopping the worker until memory recovers."
      stop_worker
      break
    fi
    sleep 2
  done

  if [ -n "$WORKER" ]; then
    wait "$WORKER"
    CODE=$?
    WORKER=""
    # 137 means the kernel killed it, which is worth retrying. Anything else is a real failure.
    if [ "$CODE" -ne 137 ]; then
      echo "[worker.sh] The worker exited with code $CODE."
      exit "$CODE"
    fi
    echo "[worker.sh] The kernel killed the worker."
  fi

  until [ "$(free_mb)" -ge "$RESUME_FREE_MB" ]; do
    sleep 5
  done
  echo "[worker.sh] $(free_mb) MB free. Starting the worker again."
done
