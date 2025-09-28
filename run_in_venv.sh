#!/usr/bin/env bash
# Lightweight runner for this project.
# Usage:
#   ./run_in_venv.sh install           # create venv and install requirements + playwright browsers
#   ./run_in_venv.sh fetch             # run fetch_tweets.py (open browser, requires X creds in .env)
#   ./run_in_venv.sh analyze-db        # run analysis querying posts from DB
#   ./run_in_venv.sh analyze-default   # run analysis on embedded default posts (fast)
#   ./run_in_venv.sh web               # start web application on port 5000
#   ./run_in_venv.sh test-suite        # run comprehensive test suite
#   ./run_in_venv.sh init-db           # initialize/reset database schema
#   ./run_in_venv.sh compare-models    # run model comparison benchmarks
#   ./run_in_venv.sh benchmarks        # run performance benchmarks
#   ./run_in_venv.sh full              # run fetch then analyze-db

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
  "$PY" "$ROOT_DIR/fetch_tweets.py" "$@"
}

analyze_db(){
  ensure_venv
  echo "Starting database analysis..."
  "$PY" "$ROOT_DIR/analyze_db_tweets.py" "$@"
}

web(){
  ensure_venv
  echo "Starting web application on port 5000..."
  cd "$ROOT_DIR/web"
  "$PY" "$ROOT_DIR/web/app.py"
}

test_suite(){
  ensure_venv
  echo "Running test suite..."
  "$PY" "$ROOT_DIR/scripts/test_suite.py" "$@"
}

init_db(){
  ensure_venv
  echo "Initializing/resetting database..."
  "$PY" "$ROOT_DIR/scripts/init_database.py" "$@"
}

compare_models(){
  ensure_venv
  echo "Running model comparison..."
  "$PY" "$ROOT_DIR/scripts/compare_models.py" "$@"
}

benchmarks(){
  ensure_venv
  echo "Running performance benchmarks..."
  "$PY" "$ROOT_DIR/scripts/performance_benchmarks.py" "$@"
}

test_status(){
  ensure_venv
  echo "Running post status tests..."
  "$PY" "$ROOT_DIR/test_post_status.py" "$@"
}

case "${1-}" in
  install)
    install
    ;;
  fetch)
    fetch
    ;;
  analyze-db)
    shift
    analyze_db "$@"
    ;;
  web)
    web
    ;;
  test-suite)
    shift
    test_suite "$@"
    ;;
  init-db)
    shift
    init_db "$@"
    ;;
  compare-models)
    shift
    compare_models "$@"
    ;;
  benchmarks)
    shift
    benchmarks "$@"
    ;;
  test-status)
    shift
    test_status "$@"
    ;;
  full)
    install || true
    fetch
    analyze_db
    ;;
  *)
    echo "Usage: $0 COMMAND [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  install           Create venv and install requirements + playwright browsers"
    echo "  fetch             Run fetch_tweets.py (requires X credentials in .env)"
    echo "  analyze-db        Run analysis on posts from database"
    echo "  web               Start web application on port 5000"
    echo "  test-suite        Run comprehensive test suite"
    echo "  init-db           Initialize/reset database schema"
    echo "  compare-models    Run model comparison benchmarks"
    echo "  benchmarks        Run performance benchmarks"
    echo "  test-status       Run post status tests"
    echo "  full              Run install, fetch, then analyze-db"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 test-suite --quick"
    echo "  $0 analyze-db --username Vox_es --limit 10"
    exit 1
    ;;
esac
