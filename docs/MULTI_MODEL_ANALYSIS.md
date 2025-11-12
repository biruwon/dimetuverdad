# Multi-Model Analysis System

## Overview

The multi-model analysis system enables parallel analysis of content using multiple local LLM models (gemma3:4b, gemma3:27b-it-qat, and gpt-oss:20b) to provide comparative insights and consensus-based categorization.

## Architecture

### Components

1. **Database Layer** (`database/database_multi_model.py`)
   - `model_analyses` table: Stores individual model results
   - `content_analyses` extensions: Tracks multi-model consensus
   - Functions for saving, retrieving, and analyzing model comparisons

2. **Analyzer Layer** (`analyzer/multi_model_analyzer.py`)
   - `MultiModelAnalyzer`: Orchestrates parallel analysis with multiple models
   - Multi-stage analysis approach: category detection → media description → explanation generation
   - Model selection and sequential execution management

3. **CLI Layer** (`scripts/analyze_multi_model.py`)
   - Command-line interface for multi-model analysis
   - Progress tracking and result visualization
   - Flexible model selection

4. **Web Layer** (`web/routes/models.py`)
   - `/models/tweet/<tweet_id>`: Side-by-side model comparison view
   - `/models/comparison`: Performance dashboard
   - `/models/stats`: API for model statistics

## Usage

### Command Line

#### Analyze tweets with all models:
```bash
./run_in_venv.sh analyze-twitter-multi
```

#### Analyze specific user:
```bash
./run_in_venv.sh analyze-twitter-multi --username Santi_ABASCAL
```

#### Analyze with specific models:
```bash
./run_in_venv.sh analyze-twitter-multi --models gemma3:4b,gpt-oss:20b
```

#### Analyze limited number of tweets:
```bash
./run_in_venv.sh analyze-twitter-multi --limit 5
```

#### Force reanalysis:
```bash
./run_in_venv.sh analyze-twitter-multi --force-reanalyze --limit 10
```

#### Verbose output (shows per-model details):
```bash
./run_in_venv.sh analyze-twitter-multi -v
```

### Web Interface

#### View Model Comparison Dashboard:
```
http://localhost:5000/models/comparison
```

#### View Specific Tweet Comparison:
```
http://localhost:5000/models/tweet/<tweet_id>
```

#### API Endpoints:
- `GET /models/stats` - Model performance statistics
- `GET /models/api/tweet/<tweet_id>/models` - JSON model analyses
- `GET /models/api/comparison/category/<category>` - Category agreement stats

## Database Schema

### `model_analyses` Table
```sql
CREATE TABLE model_analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    category TEXT NOT NULL,
    explanation TEXT NOT NULL,
    confidence_score REAL,
    processing_time_seconds REAL,
    analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT,
    UNIQUE(post_id, model_name)
);
```

### `content_analyses` Extensions
```sql
-- New fields added:
multi_model_analysis BOOLEAN DEFAULT FALSE
model_consensus_category TEXT
```

## Available Models

### gemma3:4b (Fast)
- **Type**: Fast multimodal model
- **Parameters**: 4 billion
- **Capabilities**: Text + images
- **Speed**: ~25-35 seconds per analysis
- **Best for**: Quick initial analysis, high-throughput scenarios

### gemma3:27b-it-q4_K_M (Accurate)
- **Type**: Large multimodal model with quantization
- **Parameters**: 27 billion
- **Capabilities**: Text + images
- **Speed**: ~60-90 seconds per analysis
- **Best for**: Detailed analysis, complex content

## Consensus Calculation

The system uses majority voting to calculate consensus:

1. Each model analyzes the content independently
2. Category votes are tallied
3. The category with the most votes becomes the consensus
4. Agreement score = (votes for consensus category) / (total models)

### Agreement Levels:
- **Full Agreement** (100%): All models agree
- **Partial Agreement** (≥50%): Majority of models agree
- **No Agreement** (<50%): Models significantly disagree

## Performance Metrics

The system tracks:
- **Per-model statistics**: Total analyses, success rate, avg processing time
- **Category distribution**: Which categories each model detects most
- **Agreement rates**: Overall and per-category model consensus
- **Error rates**: Failed analyses per model

## Examples

### Multi-Model Analysis Output:
```
📝 [1/10] Analyzing: 1234567890123456789
    📝 Este contenido incita al odio...
    🎯 Consensus: 🚫 hate_speech (100% agreement)
```

### Dashboard Metrics:
- **Tweets Analyzed**: 150
- **Average Agreement**: 85%
- **Full Agreement**: 120 tweets (80%)
- **Partial Agreement**: 25 tweets (17%)
- **No Agreement**: 5 tweets (3%)

## Best Practices

1. **Use all models for critical content**: Provides highest confidence
2. **Use fast model (gemma3:4b) for large batches**: Faster processing
3. **Check agreement scores**: Low agreement indicates ambiguous content
4. **Review disagreements**: Manual review of low-agreement cases recommended

## Troubleshooting

### Models not loading:
```bash
# Ensure models are pulled
ollama pull gemma3:4b
ollama pull gemma3:27b-it-q4_K_M

# Keep models loaded for faster analysis
ollama run gemma3:4b --keepalive 24h
```

### Database schema missing:
```bash
# Recreate database with new schema
./run_in_venv.sh init-db --force
```

### Slow performance:
```bash
# Use specific models only
./run_in_venv.sh analyze-twitter-multi --models gemma3:4b

# Reduce batch size
./run_in_venv.sh analyze-twitter-multi --limit 5
```

## Future Enhancements

- [ ] Confidence score calculation per model
- [ ] Weighted consensus based on model accuracy
- [ ] Model performance learning and adaptation
- [ ] Real-time model comparison in web UI
- [ ] Export comparison reports (PDF, CSV)
- [ ] A/B testing framework for model evaluation

## API Reference

### Python API

```python
from analyzer.local_analyzer import LocalMultimodalAnalyzer

# Initialize analyzer
analyzer = LocalMultimodalAnalyzer(verbose=True)

# Analyze with all models
results = await analyzer.analyze_with_multiple_models(
    content="Content to analyze",
    media_urls=["https://example.com/image.jpg"],
    models=["gemma3:4b", "gpt-oss:20b"]  # Optional: specific models
)

# Results format:
# {
#     "gemma3:4b": ("category", "explanation", processing_time),
#     "gpt-oss:20b": ("category", "explanation", processing_time)
# }
```

### Database API

```python
from utils import database_multi_model
from database import get_db_connection_context

with get_db_connection_context() as conn:
    # Save model analysis
    database_multi_model.save_model_analysis(
        conn, 
        post_id="123",
        model_name="gemma3:4b",
        category="hate_speech",
        explanation="...",
        processing_time=25.5
    )
    
    # Get consensus
    consensus = database_multi_model.get_model_consensus(conn, "123")
    # Returns: {'category': '...', 'agreement_score': 0.66, ...}
    
    # Get performance stats
    stats = database_multi_model.get_model_performance_stats(conn)
```

## Testing

```bash
# Run multi-model tests
source venv/bin/activate
python -m pytest tests/test_multi_model_basic.py -v

# Test with real content (requires models loaded)
./run_in_venv.sh analyze-twitter-multi --limit 1 --verbose
```
