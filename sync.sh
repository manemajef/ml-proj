#!/usr/bin/env bash

set -euo pipefail

VENV_DIR=".venv"
MARIMO="$VENV_DIR/bin/marimo"

if [[ ! -x "$MARIMO" ]]; then
  echo "Error: marimo not found in $VENV_DIR"
  exit 1
fi

PY_FILE="explore.py"
IPYNB_FILE="explore.ipynb"

case "${1:-}" in
  --marimo)
    echo "Syncing notebook -> marimo"
    "$MARIMO" convert "$IPYNB_FILE" -o "$PY_FILE"
    ;;
  "")
    echo "Syncing marimo -> notebook"
    "$MARIMO" export ipynb "$PY_FILE" -o "$IPYNB_FILE" -f
    ;;
  *)
    echo "Usage: ./sync.sh [--marimo]"
    exit 1
    ;;
esac