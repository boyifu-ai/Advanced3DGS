#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/outputs/install_runtime_requirements_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${PROJECT_ROOT}/outputs"

PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
PIP_RETRIES="${PIP_RETRIES:-5}"

{
echo "Installing runtime requirements"
echo "Generated at: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Python: $(which python)"
python --version
echo "PIP_INDEX_URL=${PIP_INDEX_URL:-<default>}"
echo "PIP_TIMEOUT=${PIP_TIMEOUT}"
echo "PIP_RETRIES=${PIP_RETRIES}"

python -m pip install \
  --timeout "${PIP_TIMEOUT}" \
  --retries "${PIP_RETRIES}" \
  -r "${PROJECT_ROOT}/requirements.txt"

python "${PROJECT_ROOT}/scripts/check_runtime_dependencies.py"
} 2>&1 | tee "${LOG_FILE}"

echo
echo "Saved runtime install log to: ${LOG_FILE}"
