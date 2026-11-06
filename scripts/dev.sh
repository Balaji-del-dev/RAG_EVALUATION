#!/usr/bin/env bash
# Starts the containers, then the API server and the guarded Celery worker. Ctrl+C stops both.
# Open http://127.0.0.1:8010 once it is up. For live UI editing, also run: cd frontend && npm run dev
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
docker compose --env-file .env up -d postgres minio
python manage.py migrate --no-input
scripts/worker.sh &
WORKER=$!
trap 'kill "$WORKER"' EXIT
# If the machine runs out of memory, this should be killed before the editor or browser.
echo 1000 > /proc/self/oom_score_adj
python manage.py runserver 127.0.0.1:8010
