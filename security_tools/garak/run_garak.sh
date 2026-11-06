#!/usr/bin/env bash
# Runs Garak probes against the running API. Garak is large, so it gets its own environment:
#   uv venv venv-garak --python 3.12
#   uv pip install --python venv-garak/bin/python garak
#
# Usage: security_tools/garak/run_garak.sh [probes]     (default: three small probes, under 20 prompts)
# Full probe families hold hundreds of prompts. At several seconds each on a CPU, expect hours.
set -euo pipefail
cd "$(dirname "$0")/../.."
PROBES="${1:-dan.Dan_11_0,goodside.WhoIsRiley,lmrc.QuackMedicine}"
eval "$(venv/bin/python scripts/api_session.py viewer)"

# Garak reads the secrets from the generator file, so write a private copy with them filled in.
CONFIG="$(mktemp)"
trap 'rm -f "$CONFIG"' EXIT
chmod 600 "$CONFIG"
sed -e "s|\$RAG_SESSION|$RAG_SESSION|g" -e "s|\$RAG_CSRF|$RAG_CSRF|g" security_tools/garak/rest_generator.json > "$CONFIG"

venv-garak/bin/python -m garak \
  --model_type rest.RestGenerator \
  --generator_option_file "$CONFIG" \
  --probes "$PROBES" \
  --generations 1
