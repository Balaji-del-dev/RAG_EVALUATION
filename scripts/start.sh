#!/usr/bin/env bash
# Starts the containers, the API server and the guarded worker in the background. They keep running after
# this terminal closes, until scripts/stop.sh or a reboot. Logs are in logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
docker compose up -d postgres minio
venv/bin/python manage.py migrate --no-input >/dev/null

start() {
  local name=$1
  shift
  if [ -f "logs/$name.pid" ] && kill -0 "$(cat "logs/$name.pid")" 2>/dev/null; then
    echo "$name is already running (pid $(cat "logs/$name.pid"))"
    return
  fi
  setsid nohup "$@" >>"logs/$name.log" 2>&1 &
  echo $! >"logs/$name.pid"
  echo "Started $name (pid $!)"
}

start worker scripts/worker.sh
start server bash -c 'echo 1000 > /proc/self/oom_score_adj; exec venv/bin/python manage.py runserver 127.0.0.1:8010 --noreload'
echo "Open http://127.0.0.1:8010"
