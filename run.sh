#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -B src/experiment_runner.py "$@"
