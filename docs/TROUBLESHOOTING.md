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
python -c "from utils.database import get_db_connection; conn = get_db_connection(); print('Database connected')"

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