# dimetuverdad AI Agent Instructions

## Project Overview

**dimetuverdad** is a Spanish far-right content analysis system that combines pattern matching, machine learning, and LLMs to detect hate speech, disinformation, and extremist content in Twitter/X data. The system follows a **multi-stage pipeline**: data collection → storage → analysis → web visualization.

<!-- NOTE: Keep Copilot-generated text concise. Prefer short explanations and avoid unnecessary verbosity. -->

## Core Architecture

### Analysis Pipeline Flow
```
Tweet Collection (fetch_tweets.py) 
    ↓ [Playwright scraping]
SQLite Database (accounts.db)
    ↓ [tweets + content_analyses tables]
Enhanced Analyzer (enhanced_analyzer.py)
    ↓ [Unified Pattern Detection + LLM analysis]
Web Interface (web/app.py)
    ↓ [Flask + Bootstrap visualization]
```

### Key Components

1. **`enhanced_analyzer.py`** - Main orchestration engine
   - `EnhancedAnalyzer` class manages the entire analysis workflow
   - **Always** uses LLM as fallback when pattern detection fails
   - Returns `ContentAnalysis` objects with structured results
   - Pipeline: `analyze_content()` → `_categorize_content()` → `_generate_explanation_with_smart_llm()`

2. **`pattern_analyzer.py`** - Consolidated pattern detection
   - `PatternAnalyzer` combines far-right detection + topic classification + disinformation claims in single pass
   - 13 content categories from hate_speech to political_general
   - **Eliminates redundant processing** between topic classification and extremism detection
   - Returns `AnalysisResult` with categories, pattern matches, confidence scores

3. **`llm_models.py`** - LLM integration layer
   - `EnhancedLLMPipeline` with Ollama integration (default: `gpt-oss:20b`)
   - Model priority levels: `"fast"`, `"balanced"`, `"quality"`
   - **Critical**: Preload models with `ollama run gpt-oss:20b --keepalive 24h` to avoid 3+ minute startup delays

3. **Database Schema** (`accounts.db`)
   ```sql
   tweets: tweet_id, content, username, tweet_url, tweet_timestamp, media_count, hashtags, mentions
   content_analyses: tweet_id, category, llm_explanation, analysis_method, username, analysis_timestamp
   ```

## Essential Workflows

### Environment Setup
```bash
# Always use virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies (includes transformers, torch, flask, playwright)
pip install -r requirements.txt
playwright install chromium

# Setup Ollama for LLM analysis
brew install ollama  # macOS
ollama serve
ollama pull gpt-oss:20b
```

### Data Collection
```bash
# Collect from 11 default Spanish far-right accounts (vox_es, Santi_ABASCAL, etc.)
python fetch_tweets.py --max 15

# Collect from specific users
python fetch_tweets.py --users "username1,username2" --max 20
```

### Analysis Execution
```bash
# Analyze all unanalyzed tweets (LLM always enabled)
python analyze_db_tweets.py

# Force reanalyze when prompts change
python analyze_db_tweets.py --force-reanalyze --limit 10

# Analyze specific user
python analyze_db_tweets.py --username Santi_ABASCAL
```

### Web Interface
```bash
cd web && python app.py  # Runs on localhost:5000
```

## Critical Development Patterns

### Analyzer Initialization
```python
# Default: balanced LLM models, LLM always enabled for fallback
analyzer = EnhancedAnalyzer()

# Performance optimized for testing
analyzer = EnhancedAnalyzer(model_priority="fast")

# Quality optimized for production
analyzer = EnhancedAnalyzer(model_priority="quality")
```

### Database Operations
- **Always** use `migrate_database_schema()` before analysis
- Database locked errors: restart Python process and check for hanging connections
- Use timeout=30.0 for all SQLite connections due to concurrent access

### LLM Model Management
- Keep models concise: prefer small, fast models for short explanations. Avoid overly verbose LLM responses by using compact prompts and concise settings.
- Preload models when possible: `ollama run gpt-oss:20b --keepalive 24h` to improve latency.
- Check model status: `ollama ps`

## Category Detection Logic

The system detects 6 categories with specific priority order:
1. **`hate_speech`** - Direct attacks, slurs, dehumanization (highest priority)
2. **`disinformation`** - False medical/scientific claims, fabricated facts
3. **`conspiracy_theory`** - Hidden agenda narratives, anti-institutional content
4. **`far_right_bias`** - Extremist political rhetoric, nationalist narratives
5. **`call_to_action`** - Mobilization calls, organized activities
6. **`general`** - Neutral content (fallback when no patterns detected)

## Testing & Validation

### Comprehensive Test Suite
```bash
# Quick test (2 cases per category, ~1 minute)
python scripts/test_suite.py --quick

# Full validation (all cases, ~6 minutes)
python scripts/test_suite.py --full

# Pattern-only testing (fastest)
python scripts/test_suite.py --patterns-only
```

### Individual Content Testing
```bash
# Fast pattern analysis
python quick_test.py "Test content here"

# Full LLM analysis
python quick_test.py --llm "Complex content requiring deep analysis"
```

## Performance Considerations

### Memory & CPU
- **32GB+ RAM recommended** for large LLM models
- M1 Pro optimized: ~30-60 seconds per LLM analysis (preloaded)
- Pattern-only analysis: ~2-5 seconds per tweet

### Optimization Strategies
- Use `model_priority="fast"` for development/testing
- Keep Ollama models loaded in memory with `--keepalive`
- Monitor system resources with `htop` on macOS
- Use `--patterns-only` flag for fastest analysis during development

## Integration Points

### Cross-Component Communication
- **Enhanced Analyzer** orchestrates the unified pattern analyzer and claim detector
- **Unified pattern results** flow directly into LLM prompting system  
- **Database layer** handles concurrent read/write operations
- **Web interface** reads directly from `content_analyses` table

### Component Architecture
- **`pattern_analyzer.py`**: Single-pass analysis combining topic classification + extremism detection + disinformation claims
- **`llm_models.py`**: LLM integration for complex cases and explanations
- **`enhanced_analyzer.py`**: Orchestration layer managing the complete pipeline

### External Dependencies
- **Playwright**: Web scraping with anti-detection features
- **Ollama**: Local LLM inference (preferred over remote APIs)
- **Transformers**: HuggingFace models for classification tasks
- **SQLite**: Embedded database with custom schema migrations

## Common Issues & Solutions

### LLM Not Loading
```bash
ollama serve  # Ensure service is running
ollama pull gpt-oss:20b  # Re-download model if corrupted
```

### Database Locked
```python
from enhanced_analyzer import migrate_database_schema
migrate_database_schema()  # Reset schema and connections
```

### Memory Issues
- Close unused applications
- Use `model_priority="fast"` for lighter models
- Check thermal throttling with Activity Monitor

### Slow Performance
- **Critical**: Preload models with `ollama run gpt-oss:20b --keepalive 24h`
- Use `--patterns-only` for development iterations
- Monitor CPU/memory usage during analysis

## Project-Specific Conventions

- **Spanish language focus**: All patterns, prompts, and outputs are Spanish-optimized
- **LLM fallback pattern**: Pattern detection → LLM enhancement → LLM primary (when no patterns)
- **Force reanalyze workflow**: Essential when prompt engineering or model parameters change
- **Terminal output formatting**: Uses emoji indicators (🚫, ❌, 🕵️, ⚡, 📢, ✅) for categories
- **Analysis method tracking**: Every result tagged as either `"pattern"` or `"llm"` in database
- **No backward compatibility**: Never add legacy compatibility code or deprecated methods - always refactor existing code to use new patterns directly

When working on this codebase, prioritize understanding the multi-stage analysis pipeline and always test with both pattern-only and LLM-enhanced modes to ensure comprehensive coverage.

## Terminal Usage - ABSOLUTE RESTRICTIONS

**FORBIDDEN**: Creating new terminals is COMPLETELY PROHIBITED.

### MANDATORY Terminal Rules:
1. **ONLY USE EXISTING TERMINALS**: Never, under any circumstances, create new terminal sessions
2. **ONE TERMINAL ONLY**: The system must have exactly one active terminal throughout the entire session
3. **USE `run_in_terminal` ONLY**: All terminal commands must use the existing terminal
4. **USE `run_in_venv.sh` FOR PROJECT COMMANDS**: Always use the project's runner script for Python operations
5. **NO EXCEPTIONS**: Background processes, directory changes, and all operations must use the same terminal
6. **CHECK BEFORE RUNNING**: Always use `terminal_last_command` or `get_terminal_output` first
7. **SEQUENTIAL EXECUTION**: Run commands one at a time in the same terminal context

**CRITICAL**: The `run_in_terminal` tool automatically uses the existing terminal. Creating new terminals is a SYSTEM VIOLATION that breaks virtual environment activation and workflow continuity.

**PROJECT RUNNER USAGE**: 
- Web app: `./run_in_venv.sh web`
- Analysis: `./run_in_venv.sh analyze-db`
- Fetch tweets: `./run_in_venv.sh fetch`
- Test status: `./run_in_venv.sh test-status`
- Install: `./run_in_venv.sh install`

**WEB APPLICATION MANAGEMENT**:
- **CHECK BEFORE STARTING**: Always use `lsof -i :5000` to check if web app is already running
- **DO NOT STOP EXISTING INSTANCES**: If port 5000 is in use, assume web app is running elsewhere and DO NOT attempt to kill processes
- **USE EXISTING WEB APP**: If web app is already running, simply use the existing instance at localhost:5000
- **ONLY START IF NOT RUNNING**: Only run `./run_in_venv.sh web` if port check shows port 5000 is free
- **BACKGROUND PROCESSES**: Web app should run as background process with `isBackground=true`

**IF VIRTUAL ENV ISSUES OCCUR**: The `run_in_venv.sh` script automatically handles virtual environment activation. Never manually activate with `source venv/bin/activate` - always use the runner script.

## Automatic Git Workflow

**MANDATORY CONFIRMATION REQUIREMENT**: Before committing and pushing ANY changes to GitHub, you MUST ask the user for explicit confirmation. This includes documentation updates, code changes, configuration files, or any other modifications.

### Git Automation Rules:
1. **ALWAYS ASK FIRST**: Never commit or push without explicit user confirmation
2. **Completion Triggers**: 
   - Complete feature implementation is finished
   - Bug fix is fully resolved and tested
   - Refactoring work is completed
   - Multi-step enhancement is done
   - Documentation updates are complete
3. **NOT triggered by**: Single file edits, partial implementations, or intermediate steps
4. **Ask for Confirmation**: Prompt user before committing with suggested commit message
5. **User Response Required**: Wait for explicit "yes" or "go ahead" before proceeding
6. **Automatic Commit**: Stage all changes and create a descriptive commit message (after confirmation)
7. **Automatic Push**: Push directly to the main branch without user intervention (after confirmation)
8. **Commit Message Format**: Use format: `feat: [brief description]`, `fix: [description]`, or `refactor: [description]`
9. **Include All Related Files**: Stage and commit all files modified during the complete work session
10. **Push Immediately**: Execute `git push origin main` after successful commit

**EXECUTION STEPS**:
1. Verify the complete feature/fix/refactor is finished
2. Ask user: "This feature/fix/refactor appears complete. Would you like me to commit and push these changes with message: '[proposed commit message]'?"
3. **WAIT FOR USER RESPONSE** - Do not proceed without explicit confirmation
4. If confirmed: `git add .` to stage all changes
5. `git commit -m "descriptive message"`
6. `git push origin main`
7. Confirm successful push with brief status message

**VIOLATION CONSEQUENCES**: Never commit or push without user confirmation. This is a critical workflow requirement to prevent unwanted changes.

## Test Script Management

**MANDATORY WORKFLOW**: When creating temporary test scripts for debugging or validation, follow this exact process to maintain repository cleanliness.

### Test Script Creation Rules:
1. **TEMPORARY ONLY**: Test scripts are for debugging and validation only - never commit them to repository
2. **NAMING CONVENTION**: Use descriptive names like `test_truncation.py`, `debug_extraction.py`, `validate_content.py`
3. **LOCATION**: Create in project root directory alongside other scripts
4. **PURPOSE LIMITATION**: Only for testing specific functionality - not for production features

### Test Script Cleanup Rules:
1. **IMMEDIATE DELETION**: Delete test scripts immediately after validation is complete
2. **NO EXCEPTIONS**: Never leave test scripts in the repository, even temporarily
3. **VERIFICATION**: Confirm deletion with `ls` command to ensure clean workspace
4. **DOCUMENTATION**: If test reveals permanent code changes needed, implement them in proper source files before cleanup

### Test Script Workflow:
1. Create test script for specific validation
2. Run test and verify results
3. If test passes and no code changes needed: delete immediately
4. If test reveals bugs: implement fixes in proper source files, then delete test script
5. If test reveals new features needed: implement in proper files, then delete test script
6. **ALWAYS END WITH CLEAN REPO**: Repository must be clean of test scripts before any commit

**VIOLATION CONSEQUENCES**: Test scripts found in commits will break the workflow. Always clean up before git operations.

## Test File Organization

**MANDATORY STRUCTURE**: Maintain clean, consolidated test file organization to prevent fragmentation and ensure maintainable test suites.

### Test File Consolidation Rules:
1. **ONE TEST FILE PER MODULE**: Each source module must have exactly one corresponding test file
   - `fetcher/db.py` → `fetcher/tests/test_db.py`
   - `fetcher/parsers.py` → `fetcher/tests/test_parsers.py`
   - `fetch_tweets.py` → `fetcher/tests/test_fetch_tweets.py`
   - `enhanced_analyzer.py` → `tests/test_enhanced_analyzer.py` (project root)

2. **NO FRAGMENTED TESTS**: Never create multiple test files for the same module
   - ❌ `test_parsers.py`, `test_parsers_additional.py`, `test_parsers_more.py`
   - ✅ `test_parsers.py` (single comprehensive file)

3. **COMPREHENSIVE COVERAGE**: Each test file must include all relevant test cases
   - Unit tests for all public functions
   - Integration tests for component interactions
   - Edge cases and error conditions
   - Mock objects for external dependencies

4. **TEST FILE NAMING**: Use consistent naming: `test_[module_name].py`
   - Located in appropriate test directories (`fetcher/tests/`, `tests/`)
   - Match source module names exactly

### Test Organization Workflow:
1. **Identify Module**: Determine which source module the tests belong to
2. **Check Existing**: Verify if `test_[module].py` already exists
3. **Consolidate**: Add new tests to existing file or create if missing
4. **Remove Fragments**: Delete any fragmented test files after consolidation
5. **Verify**: Run consolidated tests to ensure functionality

### Prohibited Patterns:
- Creating `test_parsers_more.py` when `test_parsers.py` exists
- Multiple test files for same functionality (e.g., `test_fetcher.py` + `test_fetch_enhanced_mock.py`)
- Test files in wrong directories (e.g., db tests in project root instead of `fetcher/tests/`)

**VIOLATION CONSEQUENCES**: Fragmented test files will be consolidated and removed. Always use the single test file per module pattern.