#!/usr/bin/env bash
# Lightweight runner for this project.
# Usage:
#   ./run_in_venv.sh install       # create venv and install requirements + playwright browsers
#   ./run_in_venv.sh fetch         # run fetch_tweets.py (open browser, requires X creds in .env)
#   ./run_in_venv.sh analyze-db    # run analysis querying posts from DB
#   ./run_in_venv.sh analyze-default # run analysis on embedded default posts (fast)
#   ./run_in_venv.sh full          # run fetch then analyze-db

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
PY="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

ensure_venv(){
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
}

install(){
  ensure_venv
  "$PIP" install --upgrade pip
  if [ -f "$ROOT_DIR/requirements.txt" ]; then
    "$PIP" install -r "$ROOT_DIR/requirements.txt"
  else
    "$PIP" install playwright transformers torch requests beautifulsoup4 lxml
  fi
  # Install Playwright browsers
  "$PY" -m playwright install
  echo "Installed requirements and Playwright browsers."
}

fetch(){
  ensure_venv
  echo "Starting fetch_tweets.py (will open Chromium). Make sure your .env has X_USERNAME/X_PASSWORD)."
  "$PY" "$ROOT_DIR/fetch_tweets.py"
}

analyze_db(){
  ensure_venv
  echo "Running analysis on posts stored in DB (fast mode: skip retrieval)."
  "$PY" -c "import analyze_posts, sys; analyze_posts.main([], skip_retrieval=True, skip_save=False)" || true
}

analyze_default(){
  ensure_venv
  echo "Running analysis on embedded default posts (fast mode)."
  "$PY" "$ROOT_DIR/default_posts.py"
}

case "${1-}" in
  install)
    install
    ;;
  fetch)
    fetch
    ;;
  analyze-db)
    analyze_db
    ;;
  analyze-default)
    analyze_default
    ;;
  full)
    install || true
    fetch
    analyze_db
    ;;
  *)
    echo "Usage: $0 {install|fetch|analyze-db|analyze-default|full}"
    exit 1
    ;;
esac
