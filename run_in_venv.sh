#!/usr/bin/env bash
# Lightweight runner for this project.
# Usage:
#   ./run_in_venv.sh install           # create venv and install requirements + playwright browsers
#   ./run_in_venv.sh fetch             # run fetch_tweets.py (open browser, requires X creds in .env)
#   ./run_in_venv.sh analyze-twitter        # run analysis querying posts from DB
#   ./run_in_venv.sh analyze-default   # run analysis on embedded default posts (fast)
#   ./run_in_venv.sh web               # start web application on port 5000
#   ./run_in_venv.sh test-analyzer-integration  # run analyzer integration tests
#   ./run_in_venv.sh test-fetch-integration       # run fetch integration tests
#   ./run_in_venv.sh test-retrieval-integration   # run retrieval integration tests
#   ./run_in_venv.sh test-integration             # run all integration tests
#   ./run_in_venv.sh test-unit                   # run all unit test files in project
#   ./run_in_venv.sh test-suite                  # run complete test suite (unit + integration)
#   ./run_in_venv.sh compare-models    # run model comparison benchmarks
#   ./run_in_venv.sh benchmarks        # run performance benchmarks
#   ./run_in_venv.sh full              # run fetch then analyze-twitter

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
  # Ensure logs directory exists
  LOG_DIR="$ROOT_DIR/logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/fetch_runner.log"

  echo "Starting fetch_tweets.py (will open Chromium). Make sure your .env has X_USERNAME/X_PASSWORD)." | tee -a "$LOG_FILE"

  # Write a timestamped invocation line and the exact exec command to the log
  # Use portable date formatting (macOS/BSD date doesn't support --iso-8601)
  echo "# [$(date -u +"%Y-%m-%dT%H:%M:%SZ")] RUN: $PY $ROOT_DIR/fetcher/fetch_tweets.py $*" >> "$LOG_FILE"

  # Execute Python with unbuffered output and tee stdout/stderr to the log so we have a persistent record
  # Use exec so PID is the Python process
  export PYTHONUNBUFFERED=1
  exec "$PY" -u "$ROOT_DIR/fetcher/fetch_tweets.py" "$@" 2>&1 | tee -a "$LOG_FILE"
}

analyze_twitter(){
  ensure_venv
  echo "Starting database analysis..."
  cd "$ROOT_DIR"
  "$PY" -m analyzer.analyze_twitter "$@"
}

web(){
  ensure_venv
  echo "Starting web application on port 5000..."
  cd "$ROOT_DIR"
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  "$PY" -m web.app
}

test_analyzer_integration(){
  ensure_venv
  echo "Running analyzer integration tests..."
  "$PY" "$ROOT_DIR/analyzer/tests/test_analyze_twitter_integration.py" "$@"
}

test_fetch_integration(){
  ensure_venv
  echo "Running fetch integration tests (requires Twitter/X credentials)..."
  "$PY" "$ROOT_DIR/fetcher/tests/test_fetch_integration.py" "$@"
}

test_retrieval_integration(){
  ensure_venv
  echo "Running retrieval integration tests..."
  "$PY" -m pytest "$ROOT_DIR/retrieval/tests/test_retrieval_integration.py" -v "$@"
}

test_integration(){
  ensure_venv
  echo "Running all integration tests (retrieval, analyzer, fetch)..."
  echo "Running retrieval integration tests..."
  "$PY" -m pytest "$ROOT_DIR/retrieval/tests/test_retrieval_integration.py" -v -n auto "$@"
  echo "Running analyzer integration tests..."
  "$PY" "$ROOT_DIR/analyzer/tests/test_analyze_twitter_integration.py" "$@"
  echo "Running fetch integration tests..."
  "$PY" "$ROOT_DIR/fetcher/tests/test_fetch_integration.py"
}

test_unit(){
  ensure_venv
  echo "Running all unit test files in the project..."
  "$PY" -m pytest "$ROOT_DIR" -v --tb=short -n auto -k "not integration"
}

test_suite(){
  ensure_venv
  echo "Running complete test suite (unit + integration tests)..."
  echo "Running unit tests..."
  test_unit
  echo "Running integration tests..."
  test_integration
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

backup_db(){
  ensure_venv
  echo "Running database backup script..."
  "$PY" "$ROOT_DIR/scripts/backup_db.py" "$@"
}

case "${1-}" in
  install)
    install
    ;;
  fetch)
    shift
    fetch "$@"
    ;;
  analyze-twitter)
    shift
    analyze_twitter "$@"
    ;;
  web)
    web
    ;;
  test-analyzer-integration)
    shift
    test_analyzer_integration "$@"
    ;;
  test-fetch-integration)
    test_fetch_integration
    ;;
  test-retrieval-integration)
    shift
    test_retrieval_integration "$@"
    ;;
  test-integration)
    shift
    test_integration "$@"
    ;;
  test-unit)
    test_unit
    ;;
  test-suite)
    test_suite
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
  backup-db)
    shift
    backup_db "$@"
    ;;
  full)
    install || true
    fetch
    analyze_twitter
    ;;
  *)
    echo "Usage: $0 COMMAND [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  install           Create venv and install requirements + playwright browsers"
    echo "  fetch             Run fetch_tweets.py (requires X credentials in .env)"
    echo "  analyze-twitter        Run analysis on posts from database"
    echo "  web               Start web application on port 5000"
    echo "  test-analyzer-integration  Run analyzer integration tests"
    echo "  test-fetch-integration     Run fetch integration tests"
    echo "  test-retrieval-integration Run retrieval integration tests"
    echo "  test-integration           Run all integration tests"
    echo "  test-unit                   Run all unit test files in project"
    echo "  test-suite                 Run complete test suite (unit + integration)"
    echo "  init-db           Initialize/reset database schema"
    echo "  compare-models    Run model comparison benchmarks"
    echo "  benchmarks        Run performance benchmarks"
    echo "  backup-db         Create/list/cleanup database backups"
    echo "  full              Run install, fetch, then analyze-twitter"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 test-analyzer-integration --quick"
    echo "  $0 test-fetch-integration"
    echo "  $0 test-retrieval-integration"
    echo "  $0 test-integration"
    echo "  $0 test-unit"
    echo "  $0 test-suite"
    echo "  $0 analyze-twitter --username Vox_es --limit 10"
    echo "  $0 backup-db list"
    exit 1
    ;;
esac
