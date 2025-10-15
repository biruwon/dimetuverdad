# dimetuverdad - Spanish Far-Right Content Analysis System

A comprehensive AI-powered system for detecting and analyzing far-right discourse, hate speech, disinformation, and extremist content in Spanish social media, particularly Twitter/X.

## 🎯 Overview

**dimetuverdad** ("Tell Me Your Truth") combats far-right extremism and disinformation in Spanish-speaking online communities by identifying:
- **Hate Speech**: Direct attacks, slurs, dehumanization
- **Disinformation**: False medical/scientific claims, conspiracy theories
- **Far-Right Bias**: Extremist political rhetoric, nationalist narratives
- **Call to Action**: Mobilization calls, organized extremist activities
- **Conspiracy Theories**: Hidden agenda narratives, anti-institutional content

## 📚 Documentation

For detailed information:
- **[🔍 System Architecture](docs/ANALYZER_PIPELINE_ARCHITECTURE.md)** - Analysis pipeline and components
- **[🕷️ Data Collection](docs/FETCHER_SYSTEM_ARCHITECTURE.md)** - Web scraping and data gathering
- **[🐳 Docker Deployment](docs/DOCKER_DEPLOYMENT.md)** - Containerized deployment
- **[📋 Command Reference](docs/COMMAND_REFERENCE.md)** - Complete usage guide
- **[🔧 Development](docs/DEVELOPMENT.md)** - Setup and testing guide
- **[🛠️ Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+, 32GB+ RAM, macOS/Linux

### Installation
```bash
git clone <repository-url>
cd dimetuverdad

# Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# Setup LLM (Ollama)
brew install ollama && ollama serve
ollama pull gpt-oss:20b

# Initialize database
python scripts/init_database.py --force
```

### Basic Usage
```bash
# Collect data from far-right accounts
./run_in_venv.sh fetch

# Analyze collected content
./run_in_venv.sh analyze-twitter

# Start web interface
./run_in_venv.sh web  # Visit http://localhost:5000
```

## 🏗️ Architecture

### Analysis Pipeline
The system uses a 3-stage pipeline:
1. **Pattern Detection** (2-5s): Fast rule-based detection
2. **Local LLM Analysis** (30-60s): Ollama gpt-oss:20b for complex cases
3. **External Analysis** (Optional): Gemini 2.5 Flash for multimodal content

### Key Components
- **Flow Manager**: Orchestrates analysis pipeline
- **Pattern Analyzer**: Fast detection for explicit content
- **Local LLM Analyzer**: Ollama-based analysis
- **External Analyzer**: Gemini multimodal wrapper
- **Web Scraper**: Playwright-based data collection
- **Database Layer**: SQLite with environment isolation

## 📊 Performance

- **Accuracy**: 98.2% pattern detection, 95.8% LLM classification
- **Speed**: 0.1-2 analyses/second depending on mode
- **Coverage**: 75%+ test coverage, 476+ passing tests
- **Categories**: 13 content categories detected

## 🔒 Security & Ethics

- **Research Focus**: Academic use for extremism detection
- **Privacy**: Public data only, no personal information
- **Compliance**: Respects platform terms and rate limits
- **Transparency**: Open-source algorithms and methodology

## 🤝 Contributing

See **[Development Guide](docs/DEVELOPMENT.md)** for:
- Environment setup and testing requirements
- Code standards (70% coverage minimum)
- Pull request process

## 📁 Project Structure

```
dimetuverdad/
├── README.md              # This overview
├── docs/                  # Detailed documentation
├── analyzer/              # Content analysis package
├── fetcher/               # Data collection package
├── web/                   # Flask web interface
├── scripts/               # Utility scripts
├── utils/                 # Shared utilities
└── tests/                 # Comprehensive test suite
```

---

*Advanced AI-powered detection of Spanish far-right discourse combining pattern matching, LLMs, and multimodal analysis.*
