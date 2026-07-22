#!/usr/bin/env bash
# Smoke-check that the container answers /health
set -euo pipefail
BASE="${1:-http://127.0.0.1:8000}"
curl -fsS "${BASE}/health" | tee /dev/stderr | grep -q '"status":"ok"'
echo "OK: ${BASE}/health"
