# Command Reference

Complete reference of all available commands for dimetuverdad operations.

## Project Setup Commands

```bash
# Install all dependencies and setup environment
./run_in_venv.sh install

# Initialize database schema
./run_in_venv.sh init-db --force

# Initialize database (safe, won't overwrite existing data)
./run_in_venv.sh init-db

# Complete setup workflow
./run_in_venv.sh full  # install + fetch + analyze
```

## Data Collection Commands

```bash
# Default full history collection from 11 Spanish far-right accounts
./run_in_venv.sh fetch

# Latest content strategy - stops after 10 consecutive existing tweets
./run_in_venv.sh fetch --latest

# Collect from specific users (full history)
./run_in_venv.sh fetch --user "username1,username2"

# Collect latest content from specific users
./run_in_venv.sh fetch --user "username1,username2" --latest
```

## Analysis Commands

```bash
# Analyze all unanalyzed tweets
./run_in_venv.sh analyze-twitter

# Analyze tweets from specific user
./run_in_venv.sh analyze-twitter --username Santi_ABASCAL

# Force reanalyze existing analyses (useful when prompts change)
./run_in_venv.sh analyze-twitter --force-reanalyze --limit 10

# Analyze specific tweet by ID
./run_in_venv.sh analyze-twitter --tweet-id 1975540692899537249

# Limit number of tweets to process
./run_in_venv.sh analyze-twitter --limit 50

# Reanalyze all tweets from a specific user (force reanalysis)
./run_in_venv.sh analyze-twitter --username Santi_ABASCAL --force-reanalyze
```

## Database Management Commands

```bash
# Initialize or reset database schema
./run_in_venv.sh init-db --force

# Initialize database (safe, won't overwrite existing data)
./run_in_venv.sh init-db

# Create timestamped database backup
./run_in_venv.sh backup-db

# List existing database backups
./run_in_venv.sh backup-db list

# Clean up old backups (keep last 10)
./run_in_venv.sh backup-db cleanup
```

## Testing Commands

```bash
# Full test suite with parallel execution
./run_in_venv.sh test-all

# Quick integration test (2 cases per category, ~1 minute)
./run_in_venv.sh test-analyzer-integration --quick

# Full integration test suite (all cases, ~6 minutes)
./run_in_venv.sh test-analyzer-integration --full

# Pattern-only tests (fast, ~10 seconds)
./run_in_venv.sh test-analyzer-integration --patterns-only

# LLM-only tests
./run_in_venv.sh test-analyzer-integration --llm-only

# Test specific categories
./run_in_venv.sh test-analyzer-integration --categories hate_speech disinformation

# Test fetch integration (requires Twitter/X credentials)
./run_in_venv.sh test-fetch-integration

# Generate test coverage report
./run_in_venv.sh test-coverage
```

## Application Commands

```bash
# Start web interface
./run_in_venv.sh web

# Performance analysis
./run_in_venv.sh benchmarks

# Database backup operations
./run_in_venv.sh backup-db              # Create backup
./run_in_venv.sh backup-db list         # List backups
./run_in_venv.sh backup-db cleanup      # Clean old backups
```

## Quick Content Analysis

Test individual posts or text content:

```bash
# Activate environment
source venv/bin/activate

# Fast pattern-only analysis (~2-5 seconds)
python quick_test.py "Los moros nos están invadiendo"

# Full LLM analysis (~30-60 seconds)
python quick_test.py --llm "Ya sabéis cómo son esa gente..."

# JSON output for integration
python quick_test.py --llm --json "Content to analyze"

# Interactive mode
python quick_test.py --interactive
```

## Performance Analysis

```bash
# Run comprehensive performance benchmarks
./run_in_venv.sh benchmarks
```

## Parallel Test Execution

The test suite supports parallel execution for improved performance:

```bash
# Run tests with automatic parallelization (recommended)
./run_in_venv.sh test-all  # Uses -n auto for optimal core utilization

# Manual parallel control
pytest -n 4  # Use 4 worker processes
pytest -n auto  # Automatic worker count based on CPU cores

# Run specific test modules in parallel
pytest analyzer/tests/ fetcher/tests/ -n auto
```

**Performance Improvements**:
- **Sequential execution**: ~50 seconds
- **Parallel execution**: ~23 seconds (2.2x speedup)
- **Test isolation**: Thread-safe database connections with unique test databases
- **Coverage**: 75%+ test coverage maintained across all modules

## Database Backup System

The project includes an automated backup system for the SQLite database:

- **Automatic Timestamping**: Backups include date and time in filename (e.g., `accounts_20251012_122943.db`)
- **Backup Directory**: All backups stored in `./backups/` (gitignored)
- **Automatic Cleanup**: Keeps last 10 backups, removes older ones
- **File Size Display**: Shows backup size in MB for verification
- **Safe Operation**: Uses `shutil.copy2()` to preserve file metadata