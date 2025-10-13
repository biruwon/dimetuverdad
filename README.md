# dimetuverdad - Spanish Far-Right Content Analysis System

A comprehensive AI-powered system for detecting and analyzing far-right discourse, hate speech, disinformation, and extremist content in Spanish social media, particularly Twitter/X. The project combines advanced pattern matching, machine learning models, and Large Language Models (LLMs) to provide accurate content moderation and political discourse analysis.

## 🎯 Project Overview

**dimetuverdad** ("Tell Me Your Truth") is designed to combat the spread of far-right extremism and disinformation in Spanish-speaking online communities. The system identifies:

- **Hate Speech**: Direct attacks, slurs, dehumanization
- **Disinformation**: False medical/scientific claims, conspiracy theories  
- **Far-Right Bias**: Extremist political rhetoric, nationalist narratives
- **Call to Action**: Mobilization calls, organized extremist activities
- **Conspiracy Theories**: Hidden agenda narratives, anti-institutional content

## 🏗️ System Architecture

### Core Components

1. **Enhanced Analyzer** (`analyzer/analyzer.py`)
   - Main orchestration engine managing the complete analysis workflow
   - Unified pattern detection combining extremism + topic classification + disinformation
   - Intelligent LLM fallback for complex content analysis
   - Returns structured `ContentAnalysis` objects with detailed results

2. **Pattern Detection System** (`analyzer/pattern_analyzer.py`)
   - Consolidated pattern matching for all content categories
   - 13 content categories from hate_speech to political_general
   - Single-pass analysis eliminating redundant processing
   - Returns `AnalysisResult` with categories and pattern matches

3. **LLM Integration** (`analyzer/llm_models.py`)
   - Enhanced LLM pipeline with Ollama integration (default: gpt-oss:20b)
   - Model priority levels: "fast", "balanced", "quality"
   - Context-aware prompt generation and response processing

4. **Multimodal Analysis** (`analyzer/gemini_multimodal.py`)
   - Gemini-powered analysis for images and rich media content
   - Media type detection and content extraction
   - Integrated with main analysis pipeline

5. **Data Collection** (`fetcher/`)
   - Playwright-based web scraping with anti-detection features
   - Session management and resume capabilities
   - Multi-account collection with rate limiting

6. **Database Layer** (`utils/database.py`)
   - Environment-isolated database connections
   - Thread-safe connection pooling
   - Schema management and migration support

7. **Web Interface** (`web/app.py`)
   - Flask-based visualization dashboard
   - Analysis results exploration and filtering
   - Administrative controls and monitoring

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- 32GB+ RAM (recommended for large LLM models)
- macOS/Linux (optimized for M1 Pro)

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd dimetuverdad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Additional Dependencies

Install Playwright for web scraping:
```bash
pip install playwright
playwright install chromium
```

Install Ollama for local LLM processing:
```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Download recommended model
ollama pull gpt-oss:20b
```

### 3. Configuration

Create `.env` file for Twitter credentials:
```bash
# Twitter/X credentials for data collection
X_USERNAME=your_username
X_PASSWORD=your_password  
X_EMAIL_OR_PHONE=your_email
```

### 4. Database Initialization

```bash
# Initialize the database schema (use the init script)
python scripts/init_database.py --force
```

## � Docker Deployment

For easy containerized deployment, the project includes Docker configuration for production use.

### Prerequisites

- Docker and Docker Compose installed
- At least 16GB RAM (recommended 32GB+ for LLM models)
- At least 50GB free disk space for models and data

### Quick Start

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd dimetuverdad
   cp .env.template .env
   # Edit .env with your configuration
   ```

2. **Build and start services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Web interface: http://localhost:5000
   - Ollama API: http://localhost:11434

### Detailed Setup

#### 1. Environment Configuration

Copy the template and configure your environment variables:

```bash
cp .env.template .env
```

Edit `.env` with your settings:
```env
# Flask Configuration
FLASK_SECRET_KEY=your-secure-random-key-here
ADMIN_TOKEN=your-admin-password

# Database (will be created automatically)
DATABASE_PATH=/app/accounts.db

# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434

# Optional: Twitter/X credentials for data collection
X_USERNAME=your_twitter_username
X_PASSWORD=your_twitter_password
```

#### 2. Build and Deploy

```bash
# Build the images
docker-compose build

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f dimetuverdad
docker-compose logs -f ollama
```

#### 3. Initialize Ollama Models

After starting the services, initialize the required LLM models:

```bash
# Connect to the Ollama container
docker-compose exec ollama bash

# Pull the required models
ollama pull gpt-oss:20b
ollama pull llama3.1:8b

# List available models
ollama list

# Exit container
exit
```

#### 4. First Data Collection (Optional)

If you have Twitter/X credentials configured:

```bash
# Run data collection (full history strategy)
docker-compose exec dimetuverdad ./run_in_venv.sh fetch

# Run latest content collection (fast strategy)
docker-compose exec dimetuverdad ./run_in_venv.sh fetch --latest

# Run analysis on collected data
docker-compose exec dimetuverdad ./run_in_venv.sh analyze-twitter
```

### Service Architecture

#### dimetuverdad Service
- **Port:** 5000
- **Health Check:** `/` endpoint
- **Dependencies:** Ollama service
- **Volumes:**
  - `./accounts.db` - SQLite database
  - `./.env` - Environment configuration

#### Ollama Service
- **Port:** 11434
- **Health Check:** `ollama list` command
- **Volumes:**
  - `ollama_data` - Model storage and cache
- **Models:** gpt-oss:20b, llama3.1:8b

### Management Commands

#### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f dimetuverdad
docker-compose logs -f ollama
```

#### Restart Services
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart dimetuverdad
```

#### Update Deployment
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up --build -d
```

#### Backup Data
```bash
# Backup database
docker-compose exec dimetuverdad cp accounts.db accounts.db.backup

# Copy backup to host
docker cp $(docker-compose ps -q dimetuverdad):/app/accounts.db.backup ./backup.db
```

### Troubleshooting

#### Common Issues

**Out of Memory**
```
ERROR: LLM analysis failed - CUDA out of memory
```
- Increase Docker memory limit to 16GB+
- Use smaller models: `ollama pull llama3.1:8b` instead of `gpt-oss:20b`

**Database Locked**
```
Database is locked
```
- Ensure only one instance is running
- Restart the dimetuverdad service

**Ollama Connection Failed**
```
Connection refused to Ollama
```
- Check Ollama service status: `docker-compose ps`
- Wait for Ollama health check to pass
- Verify network connectivity between containers

**Port Already in Use**
```
Port 5000 already in use
```
- Stop local Flask development server
- Check: `lsof -i :5000`

#### Health Checks

```bash
# Check service health
docker-compose ps

# Test web interface
curl http://localhost:5000

# Test Ollama API
curl http://localhost:11434/api/tags
```

#### Logs Analysis

```bash
# View recent errors
docker-compose logs --tail=100 dimetuverdad | grep -i error

# Monitor resource usage
docker stats
```

### Production Considerations

#### Security
- Change default `ADMIN_TOKEN` and `FLASK_SECRET_KEY`
- Use environment-specific `.env` files
- Consider using Docker secrets for sensitive data

#### Performance
- Use SSD storage for Ollama models
- Allocate sufficient RAM (32GB+ recommended)
- Monitor disk I/O during model loading

#### Monitoring
- Set up log aggregation
- Monitor container resource usage
- Implement health check alerts

#### Backup Strategy
- Regular database backups
- Model cache persistence via Docker volumes
- Configuration file versioning

## �📋 Usage Guide

### Quick Content Analysis

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

### Comprehensive Testing

Run the full test suite to validate system performance:

```bash
# Run all unit tests across the entire project (parallel execution)
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

### Parallel Test Execution

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

### Performance Analysis

```bash
# Compare different LLM models
./run_in_venv.sh compare-models --quick

# Run comprehensive performance benchmarks
./run_in_venv.sh benchmarks

# Performance analysis with specific parameters
./run_in_venv.sh compare-models --full --output results.json
```

Collect data from Spanish far-right accounts:

```bash
# Default full history collection from 11 Spanish far-right accounts
./run_in_venv.sh fetch

# Latest content strategy - fast collection, stops after 10 consecutive existing tweets
./run_in_venv.sh fetch --latest

# Collect from specific users (full history - comprehensive, slower)
./run_in_venv.sh fetch --user "username1,username2"

# Collect latest content from specific users (fast strategy)
./run_in_venv.sh fetch --user "username1,username2" --latest

# Complete workflow: install dependencies, fetch, then analyze
./run_in_venv.sh full
```

### Content Analysis

Analyze collected tweets from the database:

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

### Database Management

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

#### Database Backup System

The project includes an automated backup system for the SQLite database:

- **Automatic Timestamping**: Backups include date and time in filename (e.g., `accounts_20251012_122943.db`)
- **Backup Directory**: All backups stored in `./backups/` (gitignored)
- **Automatic Cleanup**: Keeps last 10 backups, removes older ones
- **File Size Display**: Shows backup size in MB for verification
- **Safe Operation**: Uses `shutil.copy2()` to preserve file metadata

**Backup Commands**:
- `./run_in_venv.sh backup-db` - Create new backup
- `./run_in_venv.sh backup-db list` - List all backups with sizes
- `./run_in_venv.sh backup-db cleanup` - Remove old backups (keeps 10 most recent)

### Web Interface

Launch the Flask web interface for browsing results:

```bash
# Start the web application on localhost:5000
./run_in_venv.sh web
```

Visit `http://localhost:5000` to explore:
- User tweet timelines
- Analysis results visualization  
- Pattern match details
- LLM explanations

## 🔧 Configuration Options

### Model Selection

The system supports multiple LLM configurations:

```python
# Speed-optimized (recommended for development)
analyzer = Analyzer(model_priority="fast")

# Quality-optimized (for production)
analyzer = Analyzer(model_priority="quality") 

# Balanced (default)
analyzer = Analyzer(model_priority="balanced")
```

### Analysis Modes

```python
# Pattern + LLM enhancement (full analysis)
analyzer = Analyzer(use_llm=True)

# Pattern-only with LLM fallback
analyzer = Analyzer(use_llm=False)
```

### Ollama Optimization

For optimal performance on M1 Pro MacBooks:

```bash
# Keep model loaded in memory (avoids startup delays)
ollama run gpt-oss:20b --keepalive 24h "Ready for analysis"

# Check model status
ollama ps

# Monitor system resources
htop  # Check CPU/memory usage
```

**Performance Results (M1 Pro MacBook)**:
- **Pattern-only**: ~2-5 seconds per analysis
- **LLM analysis**: ~30-60 seconds (with preloaded model) 
- **Without preloaded model**: 3+ minutes per analysis

## 📊 System Performance

### Test Results

Recent comprehensive test suite results with parallel execution:

```
📊 Parallel Test Execution Results:
🔢 Total Tests: 451 passed, 6 skipped
⚡ Execution Time: ~50 seconds
🎯 Success Rate: 98.7% (451/457 tests passing)
📈 Parallel Workers: Auto-detected CPU cores

📊 Test Coverage: 75% across all modules
� Test Categories: Unit tests, integration tests, database tests
```

### Database Isolation & Testing Infrastructure

- **Thread-Safe Testing**: File-based locking prevents parallel test race conditions
- **Parallel Execution**: pytest-xdist with automatic worker scaling
- **Test Database Management**: Shared test database with proper cleanup
- **Connection Pooling**: Optimized SQLite connections with environment-specific settings
- **Schema Validation**: Centralized schema creation from `scripts/init_database.py`

### Accuracy Metrics

- **Pattern Detection**: 98.2% accuracy on explicit content
- **LLM Classification**: 95.8% accuracy on subtle content
- **False Positive Rate**: <2% on neutral content
- **Processing Speed**: 0.1-2 tests/second depending on mode

### Categories Detected

1. **hate_speech** (🚫): Direct hate speech, slurs, dehumanizing language
2. **disinformation** (❌): False medical/scientific claims, conspiracy theories
3. **conspiracy_theory** (🕵️): Hidden agenda narratives, anti-institutional content  
4. **far_right_bias** (⚡): Extremist political rhetoric, nationalist narratives
5. **call_to_action** (📢): Mobilization calls, organized activities
6. **general** (✅): Neutral, non-problematic content

## 📋 Complete Command Reference

### Project Setup Commands

```bash
# Install all dependencies and setup environment
./run_in_venv.sh install

# Initialize database schema
./run_in_venv.sh init-db --force

# Complete setup workflow
./run_in_venv.sh full  # install + fetch + analyze
```

### Data Collection Commands

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

### Analysis Commands

```bash
# Analyze all unanalyzed tweets
./run_in_venv.sh analyze-twitter

# Analyze specific user's tweets
./run_in_venv.sh analyze-twitter --username Santi_ABASCAL

# Force reanalyze with updated prompts
./run_in_venv.sh analyze-twitter --username Santi_ABASCAL --force-reanalyze

# Analyze specific tweet
./run_in_venv.sh analyze-twitter --tweet-id 1975540692899537249

# Limit analysis (useful for testing)
./run_in_venv.sh analyze-twitter --limit 10
```

### Testing Commands

```bash
# Full test suite with parallel execution
./run_in_venv.sh test-all

# Quick integration test
./run_in_venv.sh test-analyzer-integration --quick

# Pattern-only testing
./run_in_venv.sh test-analyzer-integration --patterns-only

# Test fetch functionality (requires credentials)
./run_in_venv.sh test-fetch-integration

# Generate test coverage report
./run_in_venv.sh test-coverage
```

### Application Commands

```bash
# Start web interface
./run_in_venv.sh web

# Performance analysis
./run_in_venv.sh benchmarks

# Model comparison
./run_in_venv.sh compare-models --quick

# Database backup operations
./run_in_venv.sh backup-db              # Create backup
./run_in_venv.sh backup-db list         # List backups
./run_in_venv.sh backup-db cleanup      # Clean old backups
```

## 🧪 Development & Testing

### Running Tests

```bash
# Individual content testing
python quick_test.py "Test content here"

# Fast pattern analysis
python quick_test.py --patterns-only "Content to analyze"

# Full LLM analysis
python quick_test.py --llm "Complex content requiring deep analysis"

# JSON output for integration
python quick_test.py --llm --json "Content to analyze"
```

### Model Comparison

Compare different LLM models:

```bash
python scripts/compare_models.py --quick
```
```

### Adding New Patterns

Edit `analyzer/pattern_analyzer.py` to add detection patterns:

```python
'new_category': {
    'patterns': [
        r'\byour_regex_pattern\b',
        r'\banother_pattern\b'
    ],
    'keywords': ['keyword1', 'keyword2'],
    'description': 'Description of what this detects'
}
```

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Add comprehensive tests for new functionality
4. Ensure all tests pass: `./run_in_venv.sh test-all`
5. Ensure test coverage remains above 70%: `./run_in_venv.sh test-coverage`
6. Submit pull request with detailed description

## 📁 Project Structure

```
dimetuverdad/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── run_in_venv.sh              # Virtual environment runner script
├── config.py                   # Configuration management
├── accounts.db                 # SQLite database (auto-created)
├── accounts.db-shm            # SQLite shared memory file
├── accounts.db-wal            # SQLite write-ahead log
├── x_session.json             # Session configuration
├── TODO.md                    # Project task tracking
├── docker-compose.yml         # Docker deployment configuration
├── Dockerfile                 # Application container definition
│
├── analyzer/                  # Content analysis package
│   ├── __init__.py
│   ├── analyze_twitter.py    # Main analysis orchestrator
│   ├── analyzer.py           # Core analyzer class
│   ├── categories.py         # Category definitions
│   ├── config.py             # Analyzer configuration
│   ├── constants.py          # Analysis constants
│   ├── gemini_multimodal.py  # Gemini multimodal analysis
│   ├── llm_models.py         # LLM integration & management
│   ├── metrics.py            # Analysis metrics
│   ├── models.py             # Data models
│   ├── multimodal_analyzer.py # Multimodal content analysis
│   ├── pattern_analyzer.py   # Pattern detection engine
│   ├── prompts.py            # LLM prompt templates
│   ├── repository.py         # Database operations
│   ├── retrieval.py          # Evidence retrieval system
│   ├── text_analyzer.py      # Text analysis utilities
│   └── tests/                # Analyzer unit tests
│
├── fetcher/                  # Data collection package
│   ├── __init__.py
│   ├── collector.py          # Tweet collection logic
│   ├── config.py             # Fetcher configuration
│   ├── db.py                 # Database operations for fetching
│   ├── fetch_tweets.py       # Main tweet fetching script
│   ├── logging_config.py     # Logging configuration
│   ├── media_monitor.py      # Media content monitoring
│   ├── parsers.py            # Content parsing utilities
│   ├── refetch_manager.py    # Tweet refetching logic
│   ├── resume_manager.py     # Resume interrupted fetches
│   ├── scroller.py           # Page scrolling utilities
│   ├── session_manager.py    # Browser session management
│   └── tests/                # Fetcher unit tests
│
├── scripts/                  # Utility scripts
│   ├── init_database.py      # Database schema initialization
│   └── [other scripts]/
│
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── database.py           # Database connection management
│   ├── paths.py              # Path management utilities
│   └── text_utils.py         # Text processing utilities
│
├── web/                      # Web interface
│   ├── __init__.py
│   ├── app.py                # Flask application
│   ├── routes/               # Flask route handlers
│   └── utils/                # Web utilities
│
├── testing-scripts/          # Testing utilities
├── backups/                  # Database backups
├── logs/                     # Application logs
├── htmlcov/                  # Test coverage reports
└── repositories/             # Data repositories
```

## 🔒 Privacy & Ethics

- **No Personal Data Storage**: Only public content analysis
- **Research Purpose**: Academic/journalistic use for extremism research
- **Transparent Methodology**: Open-source algorithms and patterns
- **Bias Mitigation**: Continuous testing on diverse content types
- **Legal Compliance**: Respects platform terms of service

## 📈 Performance Monitoring

### System Status

Check system health:

```python
from analyzer import Analyzer
analyzer = Analyzer()
analyzer.print_system_status()
```

### Database Statistics

```python
import sqlite3
conn = sqlite3.connect('accounts.db')
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM content_analyses")
print(f"Total analyses: {c.fetchone()[0]}")
```

### Memory Usage

Monitor Ollama model memory:

```bash
ollama ps  # Shows loaded models and memory usage
```

## 🛠️ Troubleshooting

### Common Issues

**LLM Not Loading**:
```bash
# Check Ollama service
ollama serve

# Pull model again
ollama pull gpt-oss:20b
```

**Database Locked**:
```python
# Reset database connection
from analyzer import migrate_database_schema
migrate_database_schema()
```

**Memory Issues**:
- Close other applications
- Use `model_priority="fast"` for lighter models
- Monitor with `htop` or Activity Monitor

**Slow Performance**:
- Preload Ollama models with `--keepalive 24h`
- Use `--patterns-only` for fastest analysis
- Check system thermal throttling

### Support

For issues or questions:
1. Check the troubleshooting section above
2. Review test outputs for system validation
3. Examine log files for detailed error information
4. Ensure all dependencies are correctly installed

## 🏆 Achievements

- **75% Test Coverage** across all modules with comprehensive parallel testing
- **451+ Passing Tests** with thread-safe database isolation
- **Modular Architecture** with unified schema management and environment isolation
- **Multi-Model LLM Support** with Ollama integration and intelligent fallback
- **Real-time Pattern Detection** with 13 content categories for Spanish far-right discourse
- **Production-Ready Web Interface** with Flask-based visualization
- **Docker Containerization** for easy deployment and scaling
- **Centralized Schema Management** eliminating duplication between test and production databases

This system represents a significant advancement in automated detection of Spanish far-right discourse, combining linguistic expertise, machine learning, and robust software engineering practices to provide reliable content analysis capabilities.
