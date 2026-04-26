#!/usr/bin/env bash
# Launch LAION-Aesthetics v2 score server on the vast.ai box.
# vast_setup.sh creates /workspace/launch_score_server.sh as a wrapper that
# activates the venv first; this file is the in-repo source of truth.
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn server:app --host 0.0.0.0 --port 8189 --workers 1
