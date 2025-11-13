# Development Guide

This guide covers development workflows, testing strategies, and contribution guidelines for dimetuverdad.

## Development Environment Setup

### Prerequisites

- Python 3.8+
- 32GB+ RAM (recommended for large LLM models)
- macOS/Linux (optimized for M1 Pro)

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd dimetuverdad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Setup Ollama for LLM analysis
brew install ollama  # macOS
ollama serve
ollama pull gpt-oss:20b
```

### Environment Configuration

The project supports multiple environments (development, testing, production) with separate configuration files:

#### Environment Files Setup

1. **Copy the example configuration:**
   ```bash
   cp .env.example .env.development
   cp .env.example .env.testing
   cp .env.example .env.production
   ```

2. **Configure each environment file:**
   - **`.env.development`**: Your local development settings
   - **`.env.testing`**: Test environment with mock data
   - **`.env.production`**: Production credentials and settings

3. **Update credentials in each file:**
   ```bash
   # Edit .env.development with your actual credentials
   X_USERNAME=your_twitter_username
   X_PASSWORD=your_twitter_password
   X_EMAIL_OR_PHONE=your_email
   GEMINI_API_KEY=your_gemini_api_key
   ```

#### Environment Selection

**Default Environment**: `development`

**Switch Environments:**
TODO

**Environment Priority:**
2. Environment-specific `.env.{environment}` file
3. Fallback to `.env` file
4. Auto-detection (pytest → testing, otherwise development)

**Important**: Environment files (`.env.*`) are gitignored and should never be committed to version control.

### Database Initialization

```bash
# Initialize the database schema (use the init script)
python scripts/init_database.py --force
```

## Testing Strategy

### MANDATORY Testing Workflow for Code Changes

**ABSOLUTE REQUIREMENT**: ALL code changes MUST be tested IMMEDIATELY. No exceptions.

**IMMEDIATE TESTING RULE**:
- **EVERY CODE CHANGE** triggers immediate testing requirement
- **NO EXCEPTIONS**: Functions, refactors, bug fixes, new features, config changes
- **IMMEDIATE**: Test right after the change, not later, not at the end of session
- **BLOCKING**: Cannot continue to other work until tests pass

**Testing Requirements**:
1. **After ANY code change** (refactor, new feature, bug fix, enhancement, or modification):
   - **IMMEDIATELY identify** which test files cover the modified code
   - **IMMEDIATELY run** relevant tests: specific test files
   - If tests fail: **IMMEDIATELY STOP and fix them before any other work**
   - **NEVER proceed** with additional changes while tests are failing

2. **Test Identification by Module** (Run Targeted Tests IMMEDIATELY):
   - `analyzer/` module changes → **IMMEDIATELY** run `./run_in_venv.sh test-analyzer-integration`
   - `fetcher/` module changes → **IMMEDIATELY** run `./run_in_venv.sh test-fetch-integration`
   - `retrieval/` module changes → **IMMEDIATELY** run `./run_in_venv.sh test-retrieval-integration`
   - Database schema changes → **IMMEDIATELY** run `./run_in_venv.sh test-integration`
   - Cross-module changes → **IMMEDIATELY** run `./run_in_venv.sh test-suite`

### Test Coverage Requirements

- **MANDATORY 70% COVERAGE**: Before committing and pushing ANY code changes, unit test coverage MUST be 70% or higher
- **Coverage Verification**: Run `./run_in_venv.sh test-coverage` to generate coverage report
- **Coverage Report**: Check `htmlcov/index.html` for detailed coverage analysis
- **BLOCKING REQUIREMENT**: Cannot commit or push if coverage falls below 70%
- **Coverage Improvement**: If coverage is below 70%, add tests to reach the threshold before proceeding

### Running Tests

```bash
# Complete test suite (unit + integration)
./run_in_venv.sh test-suite

# Unit tests only
./run_in_venv.sh test-unit

# Integration tests only
./run_in_venv.sh test-integration

# Individual component tests
./run_in_venv.sh test-analyzer-integration
./run_in_venv.sh test-fetch-integration
./run_in_venv.sh test-retrieval-integration
```

### Model Comparison

Compare different LLM models:

```bash
python scripts/compare_models.py --quick
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

## Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Add comprehensive tests for new functionality
4. Ensure all tests pass: `./run_in_venv.sh test-suite`
5. Ensure test coverage remains above 70%: Run tests and verify they pass
6. Submit pull request with detailed description

### Code Standards

- **Python Style**: Follow PEP 8 guidelines
- **Imports**: All imports at the top of files, never inside functions/methods
- **Database Access**: Always use column names, never indices (`row['column_name']`, never `row[0]`)
- **Documentation**: Comprehensive docstrings for all public functions
- **Error Handling**: Proper exception handling with meaningful messages

### Commit Guidelines

- Use clear, descriptive commit messages
- Reference issue numbers when applicable
- Keep commits focused on single changes
- Test before committing

### Pull Request Process

1. **Create PR**: Open pull request with clear description
2. **Code Review**: Address reviewer feedback
3. **Testing**: Ensure all tests pass and coverage maintained
4. **Merge**: PR merged after approval

## Architecture Guidelines

### Component Design

- **Separation of Concerns**: Each component has a single responsibility
- **Dependency Injection**: Components receive dependencies rather than creating them
- **Interface Consistency**: Similar components follow consistent interfaces
- **Error Propagation**: Errors bubble up appropriately with context

### Database Operations

- **Standardized Connections**: Use `from database import get_db_connection`
- **Environment Isolation**: Automatic database path selection by environment
- **Connection Pooling**: Optimized SQLite connections with environment-specific settings
- **Schema Management**: Centralized schema creation from `scripts/init_database.py`

### LLM Integration

- **Model Management**: Consistent model loading and caching
- **Prompt Engineering**: Structured prompts with clear instructions
- **Error Handling**: Graceful degradation when LLM services unavailable
- **Performance Optimization**: Model preloading and connection reuse

## Performance Optimization

### Memory Management

- **Model Preloading**: Keep Ollama models loaded with `--keepalive 24h`
- **Resource Monitoring**: Use `htop` or Activity Monitor to track usage
- **Batch Processing**: Process multiple items efficiently
- **Connection Reuse**: Reuse database connections and API clients

### Analysis Optimization

```python
# Speed-optimized (recommended for development)
analyzer = Analyzer(model_priority="fast")

# Quality-optimized (for production)
analyzer = Analyzer(model_priority="quality")

# Balanced (default)
analyzer = Analyzer(model_priority="balanced")
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

## Debugging and Troubleshooting

### Common Issues

**LLM Not Loading**:
```bash
# Check Ollama service with debug logging
ollama serve --debug

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

### Logging

The system provides comprehensive logging:

- **Application Logs**: Stored in `logs/` directory
- **Database Operations**: Connection and query logging
- **Analysis Results**: Detailed analysis logging with timestamps
- **Error Tracking**: Comprehensive error logging with stack traces

### Performance Monitoring

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

## Security and Ethics

### Responsible AI Usage

- **Research Focus**: Content analysis for disinformation detection
- **Privacy Protection**: Public data only, no personal information collection
- **Ethical Guidelines**: Transparent methodology and bias mitigation
- **Platform Compliance**: Respectful of platform terms and rate limits

### Security Measures

- **Anti-detection**: Sophisticated web scraping with human-like behavior
- **Data Protection**: Secure credential management and environment isolation
- **Access Control**: Role-based permissions in web interface
- **Audit Logging**: Comprehensive activity and error logging

### Bias Mitigation

- **Diverse Testing**: Test on varied content types and perspectives
- **Transparent Methodology**: Open-source algorithms and decision processes
- **Continuous Validation**: Regular accuracy testing and improvement
- **Human Oversight**: Manual review capabilities for edge cases