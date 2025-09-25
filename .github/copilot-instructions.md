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

2. **`unified_pattern_analyzer.py`** - Consolidated pattern detection
   - `UnifiedPatternAnalyzer` combines far-right detection + topic classification in single pass
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
- **`unified_pattern_analyzer.py`**: Single-pass analysis combining topic classification + extremism detection
- **`claim_detector.py`**: Specialized factual statement verification
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

When working on this codebase, prioritize understanding the multi-stage analysis pipeline and always test with both pattern-only and LLM-enhanced modes to ensure comprehensive coverage.

## Terminal Usage Best Practices

**CRITICAL**: Always use existing terminals instead of creating new ones unnecessarily.

### Terminal Management Rules:
1. **Check existing terminals first**: Use `get_terminal_output` or `terminal_last_command` to see available terminals
2. **Continue in same terminal**: For sequential commands, use the same terminal session to maintain context
3. **Only create new terminals when**: 
   - Running background processes that need isolation
   - Existing terminal is occupied by a blocking process
   - Need different working directories simultaneously
4. **Background process management**: Use `isBackground=true` only for long-running services (web servers, watchers)
5. **Working directory**: Use `cd` commands within existing terminals instead of creating new ones