#!/usr/bin/env bash
# Stops the server and worker started by scripts/start.sh. The containers keep running, and a run that was
# in progress continues from its last saved answer the next time the worker starts.
cd "$(dirname "$0")/.."
for name in server worker; do
  if [ -f "logs/$name.pid" ] && kill "$(cat "logs/$name.pid")" 2>/dev/null; then
    echo "Stopped $name"
  else
    echo "$name was not running"
  fi
  rm -f "logs/$name.pid"
done
