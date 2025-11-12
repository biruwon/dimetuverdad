# Docker Deployment Guide

This guide covers containerized deployment of dimetuverdad using Docker and Docker Compose for production use.

## Prerequisites

- Docker and Docker Compose installed
- At least 16GB RAM (recommended 32GB+ for LLM models)
- At least 50GB free disk space for models and data

## Quick Start

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

## Detailed Setup

### Environment Configuration

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

### Build and Deploy

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

### Initialize Ollama Models

After starting the services, initialize the required LLM models:

```bash
# Connect to the Ollama container
docker-compose exec ollama bash

# Pull the required models
ollama pull gemma3:4b
ollama pull gemma3:27b-it-q4_K_M

# List available models
ollama list

# Exit container
exit
```

### First Data Collection (Optional)

If you have Twitter/X credentials configured:

```bash
# Run data collection (full history strategy)
docker-compose exec dimetuverdad ./run_in_venv.sh fetch

# Run latest content collection (fast strategy)
docker-compose exec dimetuverdad ./run_in_venv.sh fetch --latest

# Run analysis on collected data
docker-compose exec dimetuverdad ./run_in_venv.sh analyze-twitter
```

## Service Architecture

### dimetuverdad Service
- **Port:** 5000
- **Health Check:** `/` endpoint
- **Dependencies:** Ollama service
- **Volumes:**
  - `./accounts.db` - SQLite database
  - `./.env` - Environment configuration

### Ollama Service
- **Port:** 11434
- **Health Check:** `ollama list` command
- **Volumes:**
  - `ollama_data` - Model storage and cache
- **Models:** gemma3:4b, gemma3:27b-it-q4_K_M

## Management Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f dimetuverdad
docker-compose logs -f ollama
```

### Restart Services
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart dimetuverdad
```

### Update Deployment
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up --build -d
```

### Backup Data
```bash
# Backup database
docker-compose exec dimetuverdad cp accounts.db accounts.db.backup

# Copy backup to host
docker cp $(docker-compose ps -q dimetuverdad):/app/accounts.db.backup ./backup.db
```

## Troubleshooting

### Common Issues

**Out of Memory**
```
ERROR: LLM analysis failed - CUDA out of memory
```
- Increase Docker memory limit to 16GB+
- Use smaller models: `ollama pull gemma3:4b` instead of `gemma3:27b-it-q4_K_M`

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

### Health Checks

```bash
# Check service health
docker-compose ps

# Test web interface
curl http://localhost:5000

# Test Ollama API
curl http://localhost:11434/api/tags
```

### Logs Analysis

```bash
# View recent errors
docker-compose logs --tail=100 dimetuverdad | grep -i error

# Monitor resource usage
docker stats
```

## Production Considerations

### Security
- Change default `ADMIN_TOKEN` and `FLASK_SECRET_KEY`
- Use environment-specific `.env` files
- Consider using Docker secrets for sensitive data

### Performance
- Use SSD storage for Ollama models
- Allocate sufficient RAM (32GB+ recommended)
- Monitor disk I/O during model loading

### Monitoring
- Set up log aggregation
- Monitor container resource usage
- Implement health check alerts

### Backup Strategy
- Regular database backups
- Model cache persistence via Docker volumes
- Configuration file versioning