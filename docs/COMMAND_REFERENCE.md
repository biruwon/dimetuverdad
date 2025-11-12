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
# Run complete test suite (unit + integration tests)
./run_in_venv.sh test-suite

# Run all unit tests only
./run_in_venv.sh test-unit

# Run all integration tests
./run_in_venv.sh test-integration

# Run analyzer integration tests specifically
./run_in_venv.sh test-analyzer-integration

# Run fetch integration tests (requires Twitter/X credentials)
./run_in_venv.sh test-fetch-integration

# Run retrieval integration tests
./run_in_venv.sh test-retrieval-integration
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
./run_in_venv.sh test-suite  # Uses optimized parallel execution

# Test execution is automatically optimized for performance
# Parallel execution provides 2.2x speedup on multi-core systems
```

**Performance Improvements**:
- **Sequential execution**: ~50 seconds
- **Parallel execution**: ~23 seconds (2.2x speedup)
- **Test isolation**: Thread-safe database connections
- **Coverage**: Comprehensive test coverage maintained across all modules

## Database Backup System

The project includes an automated backup system for the SQLite database:

- **Automatic Timestamping**: Backups include date and time in filename (e.g., `accounts_20251012_122943.db`)
- **Backup Directory**: All backups stored in `./backups/` (gitignored)
- **Automatic Cleanup**: Keeps last 10 backups, removes older ones
- **File Size Display**: Shows backup size in MB for verification
- **Safe Operation**: Uses `shutil.copy2()` to preserve file metadata