# dimetuverdad AI Agent Instructions

## ⚠️ CRITICAL TESTING REQUIREMENT ⚠️

**MANDATORY**: Every completed feature/fix MUST be tested immediately after it’s finished. No exceptions. See "MANDATORY Testing Workflow" section below for enforcement details.

**NEW FEATURES REQUIRE TESTS**: When adding new features, refactoring code, or deleting functionality, corresponding unit tests MUST be updated or added IMMEDIATELY. No feature is complete without proper test coverage.

## ⚠️ CRITICAL DATABASE SAFETY REQUIREMENT ⚠️

**ABSOLUTE PROHIBITION**: NEVER EVER execute `./run_in_venv.sh init-db --force` or `scripts/init_database.py --force` under ANY circumstances. These commands DESTROY ALL DATA and are PERMANENTLY BANNED from use.

## Copilot / AI Agent Instructions (concise)

Purpose: help AI agents be productive in this repo by summarizing architecture, workflows, conventions and risky operations.

- Quick commands (use the project's runner):
  - `./run_in_venv.sh install` — install deps
  - `./run_in_venv.sh fetch` — collect tweets
  - `./run_in_venv.sh analyze-twitter` — run analysis pipeline
  - `./run_in_venv.sh web` — start web UI (localhost:5000)
  - `./run_in_venv.sh test-unit` — run all unit tests after any code change
  - `pytest path/to/test_file.py` — run specific test file

- Big picture (files to read first):
  - `fetcher/` — data collection (entry: `fetcher/fetch_tweets.py`, collectors in `fetcher/collector.py`).
  - `analyzer/flow_manager.py` — orchestrates the 3-stage pipeline (pattern → local LLM → external).
  - `analyzer/pattern_analyzer.py` and `analyzer/categories.py` — fast rule-based detection.
  - `analyzer/ollama_client.py` / `analyzer/ollama_analyzer.py` — local LLM integration.
  - `analyzer/external_analyzer.py` / `analyzer/gemini_analyzer.py` — external (Gemini) multimodal analysis.
  - `database/` — `get_db_connection()` and repository helpers; `content_analyses` table holds `analysis_stages`, `local_explanation`, `external_explanation`.
  - `web/` — Flask UI that reads `content_analyses`.

- Dataflow summary: fetcher → SQLite (`tweets`, `content_analyses`) → `flow_manager` runs pattern → optional local LLM (Ollama) → optional external (Gemini) → store explanations and stages → web UI reads results.

- Project-specific conventions (important):
  - Always run project tasks via `./run_in_venv.sh` (do not manually activate venv or run raw scripts in most cases).
  - Use `from database import get_db_connection()`; DB rows use `sqlite3.Row` and must be accessed by name (`row['column']`).
  - Do not add post-specific content into `analyzer/prompts.py` — prompts must remain generalized.
  - Test discipline: run tests immediately after completing feature/fix. Test files live under `analyzer/tests`, `fetcher/tests`, and `tests/`.  - **Testing convention**: Add or update unit tests in the existing test file for the component you modified (e.g., put tests for changes in `fetcher/thread_detector.py` into `fetcher/tests/test_thread_detector.py`). Avoid creating new test files unless you're adding tests for a brand-new module or a clearly separate test suite.
- Integration notes and external deps:
  - Local LLM: Ollama (see `analyzer/ollama_client.py`) — preloading improves latency.
  - External LLM: Gemini (multimodal) via `analyzer/gemini_analyzer.py` / `analyzer/gemini_client.py`.
  - Playwright used in `fetcher` for scraping; `playwright install chromium` required for CI/dev.

- Dangerous / gated operations (do NOT run without approval):
  - `./run_in_venv.sh init-db --force` and `scripts/init_database.py --force` — destructive, wipes DB. Always get explicit confirmation and backup before running.

- When changing prompts or LLM code:
  - Keep prompts generalized; test locally with `./run_in_venv.sh analyze-twitter --force-reanalyze --tweet-id <id>` for targeted checks.
  - Monitor LLM latency (25–60s expected depending on model) and avoid prompt bloat.

If anything is missing or you want more detail in a specific area (DB schema, `flow_manager` internals, prompt templates, test commands), tell me which section to expand.
