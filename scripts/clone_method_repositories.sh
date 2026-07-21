#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${RUN_REAL:-0}" != "1" ]]; then
  echo "This script clones the confirmed upstream repositories from configs/method_catalog.json."
  echo "It does not install packages, compile extensions, patch third-party code, or start training."
  echo "Existing repository directories are skipped."
  echo
  echo "Review commands:"
  echo "  python scripts/manage_method_repositories.py commands --all"
  echo
  echo "Clone confirmed repositories:"
  echo "  RUN_REAL=1 bash scripts/clone_method_repositories.sh"
  exit 2
fi

python scripts/manage_method_repositories.py clone --all
