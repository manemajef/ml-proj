#!/usr/bin/env bash
set -euo pipefail

PY_FILE="explore.py"
IPYNB_FILE="explore.ipynb"

case "${1:-}" in
  --marimo)
    echo "Syncing notebook -> marimo"
    marimo convert "$IPYNB_FILE" -o "$PY_FILE"
    ;;
  "")
    echo "Syncing marimo -> notebook"
    marimo export ipynb "$PY_FILE" -o "$IPYNB_FILE" --overwrite
    ;;
  *)
    echo "Usage: ./sync.sh [--marimo]"
    exit 1
    ;;
esac