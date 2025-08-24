#!/usr/bin/env zsh
# Small helper: create and activate a venv, install requirements, and run a Python script
# Usage: ./run_in_venv.sh [--] [python-args...]

set -euo pipefail

VENV_DIR="${VENV_DIR:-./venv}"
REQ_FILE="requirements.txt"

# If a venv is already active, prefer it
if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "Detected active virtualenv at $VIRTUAL_ENV; using it."
  # shellcheck source=/dev/null
  source "$VIRTUAL_ENV/bin/activate"
else
  echo "No active virtualenv detected. Using $VENV_DIR"
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
fi

echo "Upgrading pip"
python3 -m pip install --upgrade pip >/dev/null

# Optionally skip installing requirements by setting SKIP_INSTALL=1
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  if [ -f "$REQ_FILE" ]; then
    echo "Installing requirements from $REQ_FILE"
    python3 -m pip install -r "$REQ_FILE"
  else
    echo "No $REQ_FILE found, skipping pip install"
  fi
else
  echo "SKIP_INSTALL=1 set; skipping pip install"
fi

echo "Running analyze_posts.py inside venv"
exec python3 analyze_posts.py "$@"
