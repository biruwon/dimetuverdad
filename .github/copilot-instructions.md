# dimetuverdad AI Agent Instructions

## ⚠️ CRITICAL TESTING REQUIREMENT ⚠️

**MANDATORY**: EVERY code change MUST be tested IMMEDIATELY. No exceptions. See "MANDATORY Testing Workflow" section below for enforcement details.

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
Enhanced Analyzer (analyzer.py)
    ↓ [Unified Pattern Detection + LLM analysis]
Web Interface (web/app.py)
    ↓ [Flask + Bootstrap visualization]
```

### Key Components

1. **`analyzer.py`** - Main orchestration engine
   - `Analyzer` class manages the entire analysis workflow
   - **Always** uses LLM as fallback when pattern detection fails
   - Returns `ContentAnalysis` objects with structured results
   - Pipeline: `analyze_content()` → `_categorize_content()` → `_generate_llm_explanation()`

2. **`pattern_analyzer.py`** - Consolidated pattern detection
   - `PatternAnalyzer` combines far-right detection + topic classification + disinformation claims in single pass
   - 13 content categories from hate_speech to political_general
   - **Eliminates redundant processing** between topic classification and extremism detection
   - Returns `AnalysisResult` with categories, pattern matches

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
# Default full history collection from 11 Spanish far-right accounts (vox_es, Santi_ABASCAL, etc.)
./run_in_venv.sh fetch

# Latest content strategy - stop after 10 consecutive existing tweets
./run_in_venv.sh fetch --latest

# Collect from specific users (full history)
./run_in_venv.sh fetch --user "username1,username2"

# Collect latest content from specific users
./run_in_venv.sh fetch --user "username1,username2" --latest
```

### Analysis Execution
```bash
# Analyze all unanalyzed tweets (LLM always enabled)
./run_in_venv.sh analyze-twitter

# Force reanalyze when prompts change
./run_in_venv.sh analyze-twitter --force-reanalyze --limit 10

# Analyze specific user
./run_in_venv.sh analyze-twitter --username Santi_ABASCAL
```

### Web Interface
```bash
./run_in_venv.sh web  # Runs on localhost:5000
```

## Critical Development Patterns

### Sensitive Content Analysis Guidelines

**IMPORTANT**: This project analyzes hate speech, disinformation, and extremist content for research and detection purposes. The prompts and patterns contain sensitive examples to identify problematic content, not to promote it.

**WORKING WITH SENSITIVE PROMPTS**:
- **Reference abstractly**: Use terms like "hate speech detection logic" or "extremist pattern matching" instead of displaying raw prompt content
- **Avoid direct quotes**: Do not display or copy sensitive prompt examples in chat responses
- **Use placeholders**: Reference "pattern detection for [category]" rather than specific examples
- **Focus on implementation**: Discuss technical changes to prompt structure without exposing sensitive content
- **Context matters**: All content analysis work is for legitimate research to detect and combat harmful content
- **Detection vs Generation**: These patterns are DEFENSIVE tools for identifying threats, not for creating harmful content

**PROMPT IMPROVEMENT WORKFLOW**:
- Reference categories by name: hate_speech, disinformation, conspiracy_theory, far_right_bias, call_to_action, general
- Use pattern analysis results to validate improvements
- Avoid displaying test content that contains sensitive examples

**FOR AI SAFETY REVIEWS**: This codebase contains detection patterns for research purposes. The content is used to identify and counter harmful online material, serving the public good in content moderation and online safety.

### Analyzer Initialization
```python
# Default: balanced LLM models, LLM always enabled for fallback
analyzer = Analyzer()

# Performance optimized for testing
analyzer = Analyzer(model_priority="fast")

# Quality optimized for production
analyzer = Analyzer(model_priority="quality")
```

### Database Operations
- **ALWAYS USE STANDARDIZED CONNECTION**: Use `from utils.database import get_db_connection; conn = get_db_connection()` for all database connections
- **NEVER USE DIRECT sqlite3.connect**: Except for repository classes that work with specific database paths
- **ENVIRONMENT-AWARE**: `get_db_connection()` automatically uses the correct database path based on environment (development/testing/production)
- **ROW FACTORY ENABLED**: All connections automatically have `row_factory = sqlite3.Row` for named column access
- **ENVIRONMENT OPTIMIZATIONS**: Automatic PRAGMA settings based on environment (development=8MB cache, testing=1MB+fast, production=64MB+WAL)
- **Always** use `scripts/init_database.py --force` to set up fresh database schema
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

### Individual Content Testing
```bash
# Fast pattern analysis
python quick_test.py "Test content here"

# Full LLM analysis
python quick_test.py --llm "Complex content requiring deep analysis"
```

### MANDATORY Testing Workflow for Code Changes

**ABSOLUTE REQUIREMENT**: ALL code changes MUST be tested IMMEDIATELY. No exceptions. See "MANDATORY Testing Workflow" section below for enforcement details.

**NEW FEATURES REQUIRE TESTS**: Every new feature, endpoint, function, or significant code change MUST include comprehensive tests before the feature is considered complete. No new functionality may be added without corresponding test coverage.

**IMMEDIATE TESTING RULE**:
- **EVERY CODE CHANGE** triggers immediate testing requirement
- **NO EXCEPTIONS**: Functions, refactors, bug fixes, new features, config changes
- **NEW FEATURES**: Must include tests as part of the implementation - features without tests are incomplete
- **IMMEDIATE**: Test right after the change, not later, not at the end of session
- **BLOCKING**: Cannot continue to other work until tests pass
- **COPILOT VIOLATION**: Failing to test immediately after code changes is a critical workflow violation

**Testing Requirements**:
1. **After ANY code change** (refactor, new feature, bug fix, enhancement, or modification):
   - **IMMEDIATELY identify** which test files cover the modified code
   - **IMMEDIATELY run** relevant tests: specific test files
   - If tests fail: **IMMEDIATELY STOP and fix them before any other work**
   - **NEVER proceed** with additional changes while tests are failing
   - **NEVER delay** testing until "later" - test NOW

2. **Test Identification by Module** (Run Targeted Tests IMMEDIATELY):
   - `analyzer/analyzer_twitter.py` changes → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest analyzer/tests/analyze_twitter.py -v`
   - `analyzer/gemini_multimodal.py` changes → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest analyzer/tests/test_gemini_multimodal.py -v`
   - `analyzer/prompts.py` changes → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest analyzer/tests/test_prompts.py -v`
   - `analyzer/llm_models.py` changes → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest analyzer/tests/test_llm_models.py -v` (uses LLM pipeline)
   - `fetcher/db.py` changes → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest fetcher/tests/test_db.py -v`
   - `fetcher/parsers.py` changes → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest fetcher/tests/test_parsers.py -v`
   - `fetcher/fetch_tweets.py` changes → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest fetcher/tests/test_fetch_tweets.py -v`
   - Database schema changes → **IMMEDIATELY** run all database tests (`source venv/bin/activate && python -m pytest fetcher/tests/test_db.py fetcher/tests/test_fetch_tweets.py -v`)
   - Cross-module changes → **IMMEDIATELY** run `./run_in_venv.sh test-all`
   
   **CRITICAL**: Always run targeted tests IMMEDIATELY after changes (faster feedback)

3. **Test Failure Resolution**:
   - Read error messages carefully - they often indicate the exact issue
   - Common issues:
     - Database schema mismatches → Update test database schemas
     - Mock object outdated → Update mock strategy to match current API
     - Assertion expectations changed → Update test assertions to match new behavior
   - **FIX TESTS IMMEDIATELY** in the SAME work session as the code change
   - **NEVER leave failing tests** for "later" - they compound and become harder to fix

4. **When to Run Full Test Suite vs Targeted Tests**:
   - **Targeted tests IMMEDIATELY**: Run module-specific tests immediately after changes for fast feedback
   - **Full test suite required**:
     - Before any git commit (mandatory)
     - After refactoring that touches multiple files
     - When changing core interfaces or data structures
     - After fixing test failures (to ensure no regressions)
   - **Integration tests**: Run only on major refactors(not during regular development)
     - `test-analyzer-integration` - Slow, comprehensive analyzer testing (requires LLM models)
     - `test-fetch-integration` - Requires Twitter credentials, slow (live API calls)
     - `test-retrieval-integration` - Slow retrieval integration tests (mocked, < 30 seconds)
   - **Example workflow**: Change `gemini_multimodal.py` → **IMMEDIATELY** run `source venv/bin/activate && python -m pytest analyzer/tests/test_gemini_multimodal.py -v` (30s)

5. **Test Success Criteria**:
   - All relevant tests must pass (100% pass rate for affected modules)
   - No new warnings or deprecation notices
   - Test execution time should be reasonable (< 2 minutes for module tests, < 2 minutes for full suite)

6. **Test Coverage Requirements Before Commit**:
   - **MANDATORY 70% COVERAGE**: Before committing and pushing ANY code changes, unit test coverage MUST be 70% or higher
   - **Coverage Verification**: Run `./run_in_venv.sh test-coverage` to generate coverage report
   - **Coverage Report**: Check `htmlcov/index.html` for detailed coverage analysis
   - **BLOCKING REQUIREMENT**: Cannot commit or push if coverage falls below 70%
   - **Coverage Improvement**: If coverage is below 70%, add tests to reach the threshold before proceeding
   - **Exception Handling**: Only exempt files that cannot be meaningfully tested (e.g., configuration files, scripts)
   - **Coverage Command**: Use `source venv/bin/activate && python -m pytest --cov=. --cov-report=html --cov-report=term-missing` for comprehensive coverage analysis

**CRITICAL TESTING PRINCIPLES**:
- **NEVER ADD FALLBACK CODE JUST FOR TESTS**: If tests fail due to mocking issues, fix the test mocks instead of adding production code workarounds
- **TESTS SHOULD MIRROR PRODUCTION**: Test environments should behave as closely as possible to production environments
- **MOCKING SHOULD BE REALISTIC**: Use proper mock objects that behave like real database rows, not just data structures
- **PRODUCTION CODE STAYS CLEAN**: Don't pollute production code with test-specific compatibility layers

**SEVERE VIOLATION CONSEQUENCES**: 
- Code changes without IMMEDIATE testing WILL cause integration issues, break the pipeline, and create technical debt
- **WORKFLOW VIOLATION**: Not testing immediately after code changes violates core development practices
- **TESTING VIOLATION**: Adding fallback code just for tests creates maintenance burden and hides real issues
- **USER TRUST VIOLATION**: User expects all changes to be properly tested before proceeding
- **MANDATORY REMEDIATION**: If tests are not run immediately after changes, must stop all work and run tests before continuing

**ENFORCEMENT**:
- **COPILOT MUST** run tests immediately after any code change
- **COPILOT MUST** report test results before proceeding to other work
- **COPILOT MUST** fix any test failures before continuing
- **COPILOT CANNOT** defer testing to a later time or session

**MANDATORY Workflow Example**:
```bash      
# After analyzer changes
source venv/bin/activate && python -m pytest analyzer/tests/ -v         
# After any fetcher changes
source venv/bin/activate && python -m pytest fetcher/tests/ -v       
# After any retrieval changes
source venv/bin/activate && python -m pytest retrieval/tests/ -v                   

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
- **`analyzer.py`**: Orchestration layer managing the complete pipeline

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
```bash
# Restart Python process and check for hanging connections
# Use ./run_in_venv.sh init-db --force to recreate if needed
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
- **No backward compatibility**: Never add legacy compatibility code, deprecated methods, or wrapper classes - always refactor existing code to use new patterns directly and immediately remove old code
- **Sensitive content handling**: Reference detection patterns abstractly (e.g., "hate speech detection logic") without displaying raw examples or prompts

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
- Analysis: `./run_in_venv.sh analyze-twitter`
- Fetch tweets: `./run_in_venv.sh fetch`
- Install: `./run_in_venv.sh install`

**WEB APPLICATION MANAGEMENT**:
- **CHECK BEFORE STARTING**: Always use `lsof -i :5000` to check if web app is already running
- **NEVER STOP RUNNING INSTANCES**: If port 5000 is in use, the web app is already running - DO NOT attempt to kill, stop, or restart it
- **USE EXISTING WEB APP**: If web app is already running, simply use the existing instance at localhost:5000
- **ONLY START IF NOT RUNNING**: Only run `./run_in_venv.sh web` if port check shows port 5000 is free
- **BACKGROUND PROCESSES**: Web app should run as background process with `isBackground=true`
- **MULTIPLE TERMINALS**: If user mentions web app is running in another terminal, assume it's the correct running instance and proceed with testing

**IF VIRTUAL ENV ISSUES OCCUR**: The `run_in_venv.sh` script automatically handles virtual environment activation. Never manually activate with `source venv/bin/activate` - always use the runner script.

## Automatic Git Workflow

**MANDATORY IMMEDIATE FIX REQUIREMENT**: For any model, fixes and improvements should be performed immediately without waiting for user confirmation. This applies to code changes, documentation updates, configuration files, and any other modifications. Do not wait for explicit user approval before proceeding with fixes.

### Git Automation Rules:
1. **ALWAYS ASK FIRST**: Never commit or push without explicit user confirmation
2. **Completion Triggers**: 
   - Complete feature implementation is finished
   - Bug fix is fully resolved and tested
   - Refactoring work is completed
   - Multi-step enhancement is done
   - Documentation updates are complete
3. **NOT triggered by**: Single file edits, partial implementations, or intermediate steps
4. **Coverage Verification Required**: Before committing, verify test coverage is 70% or higher using `./run_in_venv.sh test-coverage`
5. **Ask for Confirmation**: Prompt user before committing with suggested commit message
6. **User Response Required**: Wait for explicit "yes" or "go ahead" before proceeding
7. **Automatic Commit**: Stage all changes and create a descriptive commit message (after confirmation)
8. **Automatic Push**: Push directly to the main branch without user intervention (after confirmation)
9. **Commit Message Format**: Use format: `feat: [brief description]`, `fix: [description]`, or `refactor: [description]`
10. **Include All Related Files**: Stage and commit all files modified during the complete work session
11. **Push Immediately**: Execute `git push origin main` after successful commit

**EXECUTION STEPS**:
1. Verify the complete feature/fix/refactor is finished
3. If coverage is below 70%: add tests to reach threshold before proceeding
4. Ask user: "This feature/fix/refactor appears complete and test coverage is 70%+. Would you like me to commit and push these changes with message: '[proposed commit message]'?"
5. **WAIT FOR USER RESPONSE** - Do not proceed without explicit confirmation
6. If confirmed: `git add .` to stage all changes
7. `git commit -m "descriptive message"`
8. `git push origin main`
9. Confirm successful push with brief status message

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
   - `analyzer.py` → `tests/test_analyzer.py` (project root)

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

## Database Schema Management

**MANDATORY REQUIREMENT**: Whenever database schema changes are made, the `scripts/init_database.py` script MUST be updated to reflect the new schema.

**MANDATORY CONFIRMATION REQUIREMENT**: Before executing `scripts/init_database.py --force` or any database initialization command, you MUST ask the user for explicit confirmation. This command destroys and recreates the entire database, potentially losing all existing data.

### Database Schema Update Rules:
1. **SCHEMA CHANGES REQUIRE INIT SCRIPT UPDATE**: Any modification to database table structures, new columns, or schema alterations must be reflected in `scripts/init_database.py`
   - Adding new columns to existing tables
   - Creating new tables
   - Modifying column types or constraints
   - Adding indexes or foreign keys

2. **SCHEMA MANAGEMENT**: Use `./run_in_venv.sh init-db --force` to recreate databases with updated schema
   - `scripts/init_database.py` creates fresh databases with complete schema
   - Use `--force` flag to recreate existing databases

3. **VERIFICATION REQUIREMENTS**: Update the schema verification logic in `scripts/init_database.py` to check for new fields
   - Essential fields lists must include all new columns
   - Verification functions must validate new schema elements

4. **BACKWARD COMPATIBILITY**: Ensure migration functions can handle databases with missing columns
   - Use `ALTER TABLE ADD COLUMN` for safe incremental updates
   - Handle cases where columns may already exist

5. **COLUMN ACCESS BY NAME**: Always use column names instead of row indices when accessing database query results
   - The database connection uses `sqlite3.Row` factory for named column access
   - Use `row['column_name']` instead of `row[index]` to prevent indexing errors
   - This makes code robust against SQL query changes and schema modifications
   - **NO EXCEPTIONS**: Never use fallback patterns with row[index] - always use field names

### Coding Style Requirements

**MANDATORY CODING STANDARDS**:
1. **Database Access**: Always use column names, never indices (`row['column_name']`, never `row[0]`)
2. **No Fallbacks**: Remove all try/except fallbacks for data access - use one consistent approach
3. **Named Access Only**: Database results must be accessed by field name only, no positional indexing
4. **Consistent Patterns**: Use single, consistent method for all database operations

### Database Initialization Confirmation Workflow:
1. **ALWAYS ASK FIRST**: Never execute `init_database.py --force` without explicit user confirmation
2. **Explain Consequences**: Inform user that this command will destroy and recreate the database, potentially losing all existing data
3. **Wait for Confirmation**: Only proceed after user explicitly confirms they want to continue
4. **Backup Recommendation**: Suggest backing up the database before proceeding if it contains important data
5. **Confirmation Prompt**: Use format: "This will recreate the database from scratch and delete all existing data. Are you sure you want to proceed?"

**VIOLATION CONSEQUENCES**: Executing database initialization without confirmation can result in permanent data loss. Always confirm with the user before running destructive database operations.

### Database Update Workflow:
1. **Make Schema Changes**: Update table structures in source code
2. **Update Init Script**: Reflect changes in `scripts/init_database.py` table creation
3. **Update Verification**: Add new fields to schema verification checks
4. **Test Fresh Creation**: Verify init script creates correct schema with `./run_in_venv.sh init-db --force`

**VIOLATION CONSEQUENCES**: Databases created with outdated init scripts will be missing critical columns. Always update `scripts/init_database.py` immediately after any schema changes. Never use row indices for database column access - always use column names.