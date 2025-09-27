# DiMeTuVerdad AI Agent Instructions

## Project Overview

**DiMeTuVerdad** is a Spanish far-right content analysis system that combines pattern matching, machine learning, and LLMs to detect hate speech, disinformation, and extremist content in Twitter/X data. The system follows a **multi-stage pipeline**: data collection → storage → analysis → web visualization.

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
- **Preload models**: `ollama run gpt-oss:20b --keepalive 24h` (critical for performance)
- Check model status: `ollama ps`
- Model loading takes 3+ minutes without preloading vs 30-60 seconds with preload

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
python comprehensive_test_suite.py --quick

# Full validation (all cases, ~6 minutes)
python comprehensive_test_suite.py --full

# Pattern-only testing (fastest)
python comprehensive_test_suite.py --patterns-only
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

**AUTOMATIC COMMITS**: When a complete feature, fix, or refactor is finished, ask for confirmation then commit and push changes to GitHub.

### Git Automation Rules:
1. **Completion Triggers**: 
   - Complete feature implementation is finished
   - Bug fix is fully resolved and tested
   - Refactoring work is completed
   - Multi-step enhancement is done
2. **NOT triggered by**: Single file edits, partial implementations, or intermediate steps
3. **Ask for Confirmation**: Prompt user before committing with suggested commit message
4. **Automatic Commit**: Stage all changes and create a descriptive commit message (after confirmation)
5. **Automatic Push**: Push directly to the main branch without user intervention (after confirmation)
6. **Commit Message Format**: Use format: `feat: [brief description]`, `fix: [description]`, or `refactor: [description]`
7. **Include All Related Files**: Stage and commit all files modified during the complete work session
8. **Push Immediately**: Execute `git push origin main` after successful commit

**COMMIT MESSAGE EXAMPLES**:
- `feat: add political_general category with distinct UI styling and filtering`
- `fix: resolve tweet sorting to prioritize political content over general posts`
- `refactor: consolidate terminal usage rules and eliminate multiple terminal creation`
- `feat: implement automatic git workflow for completed features`

**EXECUTION STEPS**:
1. Verify the complete feature/fix/refactor is finished
2. Ask user: "This feature/fix/refactor appears complete. Would you like me to commit and push these changes with message: '[proposed commit message]'?"
3. If confirmed: `git add .` to stage all changes
4. `git commit -m "descriptive message"`
5. `git push origin main`
6. Confirm successful push with brief status message