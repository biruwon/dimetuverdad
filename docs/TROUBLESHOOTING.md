# Troubleshooting Guide

Common issues and solutions for dimetuverdad deployment and operation.

## Installation Issues

### Python Version Compatibility

**Issue**: `ModuleNotFoundError` or import errors
```
ModuleNotFoundError: No module named 'package_name'
```

**Solutions**:
- Ensure Python 3.8+ is installed: `python --version`
- Use virtual environment: `python -m venv venv && source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Check pip version: `pip --version`

### Virtual Environment Issues

**Issue**: Virtual environment not activating
```
Command not found: python
```

**Solutions**:
- Activate virtual environment: `source venv/bin/activate` (Linux/macOS)
- Or: `venv\Scripts\activate` (Windows)
- Check environment: `which python` should point to venv
- Recreate venv if corrupted: `rm -rf venv && python -m venv venv`

### Playwright Installation

**Issue**: Browser automation fails
```
Browser not found or incompatible
```

**Solutions**:
```bash
# Install Playwright
pip install playwright

# Install browser binaries
playwright install chromium

# Update browsers
playwright install --force
```

## LLM and Ollama Issues

### Ollama Service Not Running

**Issue**: LLM analysis fails with connection errors
```
Connection refused to Ollama
```

**Solutions**:
```bash
# Start Ollama service
ollama serve

# Check service status
ollama ps

# Restart service
brew services restart ollama  # macOS
```

### Model Download Issues

**Issue**: Model pull fails or is corrupted
```
Error: pull model manifest: Get "https://...": dial tcp: lookup registry.ollama.ai: no such host
```

**Solutions**:
```bash
# Check internet connection
ping google.com

# Pull model again
ollama pull gpt-oss:20b

# List available models
ollama list

# Remove and re-download if corrupted
ollama rm gpt-oss:20b
ollama pull gpt-oss:20b
```

### Memory Issues

**Issue**: Out of memory during LLM analysis
```
CUDA out of memory
ERROR: LLM analysis failed - CUDA out of memory
```

**Solutions**:
- Close other applications
- Use smaller models: `ollama pull llama3.1:8b`
- Increase system memory (32GB+ recommended)
- Use pattern-only analysis: `--patterns-only` flag

### Slow LLM Performance

**Issue**: LLM analysis takes 3+ minutes
```
Analysis taking too long
```

**Solutions**:
```bash
# Preload model in memory
ollama run gpt-oss:20b --keepalive 24h

# Check model status
ollama ps

# Use faster models for development
ollama pull llama3.1:8b
```

## Database Issues

### Database Locked

**Issue**: SQLite database locked errors
```
Database is locked
```

**Solutions**:
- Close other Python processes using the database
- Restart Python interpreter
- Check for hanging connections: `lsof accounts.db`
- Use database migration: `python scripts/init_database.py --force`

### Schema Mismatches

**Issue**: Database schema errors after updates
```
Table/column does not exist
```

**Solutions**:
```bash
# Reset database schema
./run_in_venv.sh init-db --force

# Check schema version
python -c "import sqlite3; conn = sqlite3.connect('accounts.db'); c = conn.cursor(); c.execute('SELECT sql FROM sqlite_master WHERE type=\"table\"'); print(c.fetchall())"
```

### Connection Issues

**Issue**: Database connection failures
```
Unable to connect to database
```

**Solutions**:
- Check file permissions: `ls -la accounts.db`
- Verify database path in environment variables
- Use absolute paths for database connections
- Check disk space: `df -h`

## Web Scraping Issues

### Twitter/X Authentication

**Issue**: Login failures or rate limiting
```
Authentication failed
Rate limit exceeded
```

**Solutions**:
- Verify credentials in `.env` file
- Use different account if rate limited
- Wait for rate limit reset (usually 15 minutes)
- Check Twitter/X status: https://twitterstatus.com

### Anti-Detection Measures

**Issue**: Content collection blocked
```
Content not loading or access denied
```

**Solutions**:
- Update Playwright: `pip install --upgrade playwright`
- Reinstall browser: `playwright install --force`
- Use different user agents or proxies
- Reduce collection frequency

### Network Issues

**Issue**: Connection timeouts or DNS failures
```
Connection timeout
DNS resolution failed
```

**Solutions**:
- Check internet connection: `ping google.com`
- Use different DNS: `8.8.8.8`
- Configure proxy if needed
- Wait and retry (may be temporary outage)

## Analysis Issues

### Pattern Detection Failures

**Issue**: No patterns detected in content
```
No categories detected
```

**Solutions**:
- Check content language (must be Spanish)
- Verify content length (minimum requirements)
- Update patterns in `analyzer/pattern_analyzer.py`
- Test with known problematic content

### Analysis Pipeline Errors

**Issue**: Pipeline fails at specific stages
```
Stage X failed in analysis pipeline
```

**Solutions**:
- Check individual components: pattern analyzer, LLM, external API
- Verify API keys and credentials
- Check system resources (memory, CPU)
- Review error logs for specific failure points

## Docker Issues

### Container Startup Failures

**Issue**: Docker containers fail to start
```
Container exited with code 1
```

**Solutions**:
```bash
# Check container logs
docker-compose logs dimetuverdad
docker-compose logs ollama

# Verify environment file
cat .env

# Rebuild containers
docker-compose build --no-cache

# Check system resources
docker system df
```

### Port Conflicts

**Issue**: Port already in use
```
Port 5000 already in use
```

**Solutions**:
```bash
# Find process using port
lsof -i :5000

# Kill process or change port
docker-compose down

# Change port in docker-compose.yml
ports:
  - "5001:5000"
```

### Volume Mount Issues

**Issue**: File permission errors in containers
```
Permission denied on volume mount
```

**Solutions**:
- Check file permissions: `ls -la accounts.db`
- Change ownership: `chown -R $USER:$USER .`
- Use Docker volumes instead of bind mounts
- Run container as current user

## Performance Issues

### System Resource Monitoring

**Issue**: High CPU/memory usage
```
System running slow
```

**Solutions**:
```bash
# Monitor resources (macOS)
htop
# or Activity Monitor

# Check Ollama resource usage
ollama ps

# Monitor Docker containers
docker stats

# Limit container resources in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G
      cpus: '4.0'
```

### Thermal Throttling

**Issue**: Performance degrades over time
```
Analysis getting slower
```

**Solutions**:
- Check system temperature
- Improve cooling/ventilation
- Reduce concurrent operations
- Use lighter models during development

## Testing Issues

### Test Failures

**Issue**: Tests failing unexpectedly
```
FAILED test_something.py::test_function - AssertionError
```

**Solutions**:
- Run specific test: `pytest path/to/test.py::test_function -v`
- Check test environment setup
- Verify test data and mocks
- Update tests to match code changes

### Coverage Issues

**Issue**: Test coverage below 70%
```
Coverage is below required threshold
```

**Solutions**:
```bash
# Generate coverage report
./run_in_venv.sh test-coverage

# Check coverage details
open htmlcov/index.html

# Add missing tests for uncovered code
# Focus on public methods and error paths
```

### Parallel Test Issues

**Issue**: Tests fail when run in parallel
```
Database locked during parallel tests
```

**Solutions**:
- Use unique test databases per worker
- Implement proper test isolation
- Run tests sequentially: `pytest --maxfail=1`
- Check for shared state between tests

## Configuration Issues

### Environment Variables

**Issue**: Configuration not loading
```
Environment variable not found
```

**Solutions**:
- Check `.env` file exists and is readable
- Verify variable names match code expectations
- Use absolute paths for file references
- Check environment precedence (development > testing > production)

### API Key Issues

**Issue**: External API failures
```
Invalid API key
```

**Solutions**:
- Verify API keys in environment files
- Check API key permissions and quotas
- Rotate keys if compromised
- Test API connectivity independently

## Logging and Debugging

### Log Analysis

```bash
# View application logs
tail -f logs/fetch_runner.log

# Search for errors
grep -r "ERROR" logs/

# Check database operations
python -c "
import sqlite3
conn = sqlite3.connect('accounts.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM tweets')
print(f'Tweets: {c.fetchone()[0]}')
c.execute('SELECT COUNT(*) FROM content_analyses')
print(f'Analyses: {c.fetchone()[0]}')
"
```

### System Status Checks

```bash
# Check Ollama status
ollama ps

# Test web interface
curl http://localhost:5000

# Verify database connectivity
python -c "from database import get_db_connection; conn = get_db_connection(); print('Database connected')"

# Check system resources
df -h  # Disk space
free -h  # Memory (Linux)
vm_stat  # Memory (macOS)
```

### Health Checks

```bash
# Docker health checks
docker-compose ps

# Application health
curl -f http://localhost:5000/health || echo "Health check failed"

# Ollama API
curl http://localhost:11434/api/tags
```

## Database Queries and Analysis

### Database Overview

The system uses SQLite with two main tables:
- **`tweets`**: Raw tweet data collected from Twitter/X
- **`content_analyses`**: Analysis results and classifications

### Useful Database Queries

#### Basic Statistics

```sql
-- Total tweets and analyses
SELECT 
    (SELECT COUNT(*) FROM tweets) as total_tweets,
    (SELECT COUNT(*) FROM content_analyses) as total_analyses,
    ROUND((SELECT COUNT(*) FROM content_analyses) * 100.0 / (SELECT COUNT(*) FROM tweets), 1) as analysis_coverage_percent;

-- Top 10 users by tweet count
SELECT username, COUNT(*) as tweet_count 
FROM tweets 
GROUP BY username 
ORDER BY tweet_count DESC 
LIMIT 10;
```

#### Content Analysis Statistics

```sql
-- Category distribution
SELECT category, COUNT(*) as count 
FROM content_analyses 
WHERE category IS NOT NULL 
GROUP BY category 
ORDER BY count DESC;

-- Analysis stages usage
SELECT analysis_stages, COUNT(*) as count 
FROM content_analyses 
WHERE analysis_stages IS NOT NULL 
GROUP BY analysis_stages 
ORDER BY count DESC;

-- External analysis usage
SELECT 
    CASE WHEN external_analysis_used = 1 THEN 'Used' ELSE 'Not used' END as external_analysis_status,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM content_analyses), 1) as percentage
FROM content_analyses 
GROUP BY external_analysis_used;
```

#### Media Analysis Statistics

```sql
-- Media content statistics
SELECT 
    (SELECT COUNT(*) FROM tweets WHERE media_count > 0) as tweets_with_media,
    (SELECT COUNT(*) FROM tweets) as total_tweets,
    ROUND((SELECT COUNT(*) FROM tweets WHERE media_count > 0) * 100.0 / (SELECT COUNT(*) FROM tweets), 1) as media_percentage,
    (SELECT COUNT(*) FROM content_analyses WHERE multimodal_analysis = 1) as multimodal_analyses;

-- Media types distribution
SELECT media_type, COUNT(*) as count 
FROM content_analyses 
WHERE media_type IS NOT NULL AND media_type != '' 
GROUP BY media_type 
ORDER BY count DESC;
```

#### Specific Tweet Investigation

```sql
-- Check specific tweet details (replace '1935239994274370004' with actual tweet ID)
SELECT tweet_id, username, content, media_count, media_links 
FROM tweets 
WHERE tweet_id = '1935239994274370004';

-- Check analysis details for specific tweet
SELECT author_username, category, external_explanation, multimodal_analysis, analysis_stages, media_urls 
FROM content_analyses 
WHERE post_id = '1935239994274370004';
```

#### Recent Activity and Errors

```sql
-- Recent analyses (last 24 hours)
SELECT 
    COUNT(*) as analyses_last_24h,
    MAX(analysis_timestamp) as latest_analysis
FROM content_analyses 
WHERE analysis_timestamp > datetime('now', '-1 day');

-- Failed analyses (no category assigned)
SELECT COUNT(*) as analyses_without_category
FROM content_analyses 
WHERE category IS NULL OR category = '';

-- Analyses with error indicators in explanations
SELECT COUNT(*) as analyses_with_errors
FROM content_analyses 
WHERE external_explanation LIKE '%error%' 
   OR external_explanation LIKE '%failed%'
   OR external_explanation LIKE '%Error%'
   OR external_explanation LIKE '%Failed%';
```

#### Performance Analysis

```sql
-- Daily analysis counts (last 7 days)
SELECT 
    DATE(analysis_timestamp) as analysis_date, 
    COUNT(*) as daily_count
FROM content_analyses 
WHERE analysis_timestamp >= date('now', '-7 days')
GROUP BY DATE(analysis_timestamp) 
ORDER BY analysis_date DESC;

-- Analysis stages performance
SELECT 
    analysis_stages, 
    COUNT(*) as count,
    ROUND(AVG(LENGTH(external_explanation)), 0) as avg_explanation_length
FROM content_analyses 
WHERE analysis_stages IS NOT NULL
GROUP BY analysis_stages;
```

### Advanced Queries

#### Cross-table Analysis

```sql
-- Tweets with media that weren't analyzed with multimodal
SELECT 
    t.tweet_id, 
    t.username, 
    t.media_count, 
    CASE WHEN ca.multimodal_analysis = 1 THEN 'Yes' ELSE 'No' END as multimodal_analysis,
    ca.category
FROM tweets t
LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
WHERE t.media_count > 0 
  AND (ca.multimodal_analysis = 0 OR ca.multimodal_analysis IS NULL)
LIMIT 10;

-- Unanalyzed tweets
SELECT 
    t.tweet_id, 
    t.username, 
    t.content,
    t.tweet_timestamp
FROM tweets t
LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
WHERE ca.post_id IS NULL
ORDER BY t.tweet_timestamp DESC
LIMIT 20;
```

#### Pattern Matching Analysis

```sql
-- Most common pattern matches (requires JSON parsing)
SELECT pattern_matches, COUNT(*) as frequency
FROM content_analyses 
WHERE pattern_matches IS NOT NULL 
  AND pattern_matches != '[]'
GROUP BY pattern_matches 
ORDER BY frequency DESC 
LIMIT 10;

-- Categories by user
SELECT 
    ca.author_username,
    ca.category,
    COUNT(*) as count
FROM content_analyses ca
GROUP BY ca.author_username, ca.category
ORDER BY ca.author_username, count DESC;
```

### Database Maintenance Queries

```sql
-- Check database integrity
PRAGMA integrity_check;

-- Analyze query performance
PRAGMA analyze;

-- Check table sizes
SELECT 
    name as table_name,
    sql
FROM sqlite_master 
WHERE type = 'table';

-- Get row counts for all tables
SELECT 
    'tweets' as table_name, COUNT(*) as row_count FROM tweets
UNION ALL
SELECT 
    'content_analyses' as table_name, COUNT(*) as row_count FROM content_analyses;

-- Database file information
SELECT 
    page_count * page_size as database_size_bytes,
    ROUND(page_count * page_size / 1024.0 / 1024.0, 2) as database_size_mb
FROM pragma_page_count(), pragma_page_size();
```

### Quick Reference Commands

To connect to the database using SQLite CLI:
```bash
sqlite3 accounts.db
```

To run these queries:
1. Open SQLite CLI: `sqlite3 accounts.db`
2. Copy and paste any SQL query above
3. Use `.quit` to exit SQLite CLI

To export query results to CSV:
```bash
sqlite3 -header -csv accounts.db "SELECT * FROM tweets LIMIT 10;" > tweets_sample.csv
```

## Getting Help

### Support Resources

1. **Check Documentation**: Review this troubleshooting guide and main README
2. **Search Issues**: Check GitHub Issues for similar problems
3. **Log Analysis**: Examine logs for error details and stack traces
4. **Test Isolation**: Reproduce issues with minimal test cases
5. **Community**: Ask questions in GitHub Discussions

### When to Report Issues

- **Bug Reports**: Include reproduction steps, error logs, and system information
- **Performance Issues**: Include system specs, resource usage, and benchmark results
- **Configuration Problems**: Share relevant configuration (without sensitive data)
- **Feature Requests**: Describe use case and expected behavior

### Emergency Procedures

**Data Loss**:
- Check backups in `./backups/` directory
- Restore from timestamped backup: `cp backups/accounts_*.db accounts.db`

**System Unresponsive**:
- Stop all processes: `pkill -f python`
- Restart services: `./run_in_venv.sh web`
- Clear temporary files and caches

**Corrupted Database**:
- Restore from backup: `./run_in_venv.sh backup-db list`
- Reinitialize schema: `./run_in_venv.sh init-db --force`