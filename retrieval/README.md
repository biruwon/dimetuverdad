# Retrieval System for Evidence-Based Content Verification

A comprehensive evidence retrieval and verification system that transforms the dimetuverdad analyzer into a "source of trust" for validating disinformation and numerical claims in Spanish political content.

## Overview

The retrieval system provides multi-source verification capabilities by:

- **Extracting verifiable claims** from text (numerical, temporal, attribution, causal)
- **Searching multiple data sources** (statistical APIs, web sources, official records)
- **Scoring source credibility** using multi-factor assessment
- **Aggregating evidence** with weighted decision making
- **Validating temporal claims** against known events and logical consistency
- **Integrating seamlessly** with the main analyzer pipeline

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Claim         │    │   Evidence       │    │   Credibility   │
│   Extraction    │───▶│   Aggregation    │───▶│   Scoring       │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query         │    │   Multi-Source   │    │   Temporal      │
│   Building      │    │   Verification   │    │   Validation    │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Analyzer      │    │   Performance    │    │   Main API      │
│   Integration   │    │   Optimization   │    │   Interface     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Basic Usage

```python
import asyncio
from retrieval import verify_text_content, verify_single_claim

async def main():
    # Verify a piece of content
    result = await verify_text_content(
        "Según datos oficiales, hay 47 millones de españoles y el PIB creció un 2%.",
        category="general",
        language="es"
    )

    print(f"Verification successful: {result.success}")
    if result.verification_report:
        print(f"Overall verdict: {result.verification_report.overall_verdict}")
        print(f"Confidence: {result.verification_report.confidence_score:.1f}%")

    # Verify a single claim
    claim_result = await verify_single_claim(
        "La población de España es de 47 millones",
        claim_type="numerical"
    )

    print(f"Claim verdict: {claim_result.verdict}")

asyncio.run(main())
```

### Advanced Usage with Full API

```python
from retrieval import create_retrieval_api, RetrievalConfig, VerificationRequest

# Create configured API instance
config = RetrievalConfig(
    max_parallel_requests=4,
    verification_timeout=45.0
)

api = create_retrieval_api(config)

async def verify_content():
    request = VerificationRequest(
        content="El gobierno dice que la inflación bajó del 10% al 3% en 6 meses.",
        content_category="disinformation",
        language="es",
        priority_level="balanced"
    )

    result = await api.verify_content(request)

    if result.success and result.verification_report:
        report = result.verification_report
        print(f"Verdict: {report.overall_verdict}")
        print(f"Claims verified: {len(report.claims_verified)}")
        print(f"Evidence sources: {len(report.evidence_sources)}")

        if report.contradictions_found:
            print("Contradictions detected:")
            for contradiction in report.contradictions_found:
                print(f"  - {contradiction}")

asyncio.run(verify_content())
```

## Integration with Main Analyzer

### Automatic Verification

```python
from retrieval import create_retrieval_api

class AnalyzerWithVerification:
    def __init__(self):
        self.retrieval_api = create_retrieval_api()

    async def analyze_content(self, content: str) -> dict:
        # Run original analysis
        original_result = self._run_original_analysis(content)

        # Verify with verification if appropriate
        verification_result = await self.retrieval_api.analyze_with_verification(
            content, original_result
        )

        return {
            "category": verification_result.original_result["category"],
            "confidence": verification_result.verification_data.get("verification_confidence", 0),
            "explanation": verification_result.explanation_with_verification,
            "sources_cited": verification_result.verification_data.get("sources_cited", []),
            "verification_data": verification_result.verification_data
        }
```

### Conditional Triggering

```python
from retrieval.integration import create_analyzer_hooks

hooks = create_analyzer_hooks()

# Check if verification should be triggered
should_verify, reason = hooks.should_trigger_verification(
    content="Hay 50 millones de españoles según esta estadística.",
    analyzer_result={"category": "disinformation", "confidence": 0.85}
)

if should_verify:
    print(f"Triggering verification: {reason}")
    result = await hooks.analyze_with_verification(content, analyzer_result)
    # Use result
```

## Component Details

### Claim Extraction

Extracts verifiable claims from text:

```python
from retrieval.core import ClaimExtractor

extractor = ClaimExtractor()
claims = extractor.extract_claims("Según el INE, hay 47.4 millones de españoles.")

for claim in claims:
    print(f"Type: {claim.claim_type}, Text: {claim.claim_text}, Confidence: {claim.confidence}")
```

**Supported claim types:**
- `numerical`: Statistics, percentages, quantities
- `temporal`: Dates, time periods, sequences
- `attribution`: Source citations, references
- `causal`: Cause-effect relationships
- `statistical`: Data analysis claims

### Evidence Aggregation

Combines evidence from multiple sources:

```python
from retrieval.core import EvidenceAggregator

aggregator = EvidenceAggregator()
result = aggregator.aggregate_evidence(claim, evidence_sources)

print(f"Verdict: {result.verdict}")
print(f"Confidence: {result.confidence_score}")
print(f"Explanation: {result.explanation}")
```

### Credibility Scoring

Multi-factor source assessment:

```python
from retrieval.verification import CredibilityScorer

scorer = CredibilityScorer()
scored_sources = scorer.score_sources_batch(sources)

for source in scored_sources:
    print(f"{source.source_name}: {source.credibility_score}/100")
```

**Scoring factors:**
- Base reputation (government, academic, news sources)
- Content quality (fact-checking, peer review)
- Freshness (publication date relevance)
- Context relevance (topic alignment)
- Verdict consistency (agreement with other sources)

### Temporal Verification

Validates dates and time claims:

```python
from retrieval.verification import TemporalVerifier

verifier = TemporalVerifier()
is_verified, explanation, date = verifier.verify_temporal_claim(
    "La pandemia empezó en marzo de 2020"
)

print(f"Verified: {is_verified}, Explanation: {explanation}")
```

### Statistical API Integration

Queries official data sources:

```python
from retrieval.sources import StatisticalAPIManager

api_manager = StatisticalAPIManager()
results = await api_manager.query_all_sources("población españa", "es")

for result in results:
    print(f"Source: {result['source_name']}")
    print(f"Data: {result['description']}")
```

**Supported APIs:**
- INE (Instituto Nacional de Estadística)
- Eurostat (European statistics)
- WHO (World Health Organization)
- World Bank (economic indicators)

## Configuration

### RetrievalConfig Options

```python
config = RetrievalConfig(
    max_parallel_requests=4,       # Concurrent verification limit
    verification_timeout=30.0,     # Timeout in seconds
    enable_statistical_apis=True,  # Enable official data APIs
    enable_web_search=False,       # Web search (placeholder)
    default_language="es",         # Default language
    log_level="INFO"               # Logging level
)
```

### Performance Tuning

```python
# Fast mode for development
fast_config = RetrievalConfig(
    max_parallel_requests=2,
    verification_timeout=15.0,
    priority_level="fast"
)

# Quality mode for production
quality_config = RetrievalConfig(
    max_parallel_requests=8,
    verification_timeout=60.0,
    priority_level="quality"
)
```

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest retrieval/tests/ -v

# Run specific test categories
python -m pytest retrieval/tests/test_retrieval_system.py::TestClaimExtractor -v
python -m pytest retrieval/tests/test_retrieval_system.py::TestCredibilityScorer -v

# Run performance tests
python -m pytest retrieval/tests/test_retrieval_system.py::TestPerformance -v
```

### Test Coverage

The test suite covers:
- ✅ Claim extraction accuracy
- ✅ Credibility scoring algorithms
- ✅ Temporal verification logic
- ✅ Evidence aggregation consistency
- ✅ Multi-source verification pipeline
- ✅ API integration (mocked)
- ✅ Performance characteristics
- ✅ Integration with analyzer hooks

## API Reference

### Main Classes

- `RetrievalAPI`: Main interface for verification
- `MultiSourceVerifier`: Core verification engine
- `ClaimExtractor`: Claim identification and extraction
- `EvidenceAggregator`: Evidence combination and weighting
- `CredibilityScorer`: Source trustworthiness assessment
- `TemporalVerifier`: Date and time validation
- `AnalyzerHooks`: Integration with main analyzer

### Key Methods

- `verify_content(request)`: Full content verification
- `verify_claim(text, type)`: Single claim verification
- `analyze_with_verification(content, result)`: Analyzer verification
- `extract_claims(text)`: Claim extraction only

### Data Models

- `VerificationRequest`: Input for verification
- `RetrievalResult`: Verification output
- `VerificationReport`: Detailed verification report
- `Claim`: Extracted claim with metadata
- `EvidenceSource`: Source with credibility score
- `VerificationVerdict`: VERIFIED, QUESTIONABLE, DEBUNKED, UNVERIFIED

## Error Handling

The system includes comprehensive error handling:

```python
try:
    result = await api.verify_content(request)
    if not result.success:
        print(f"Verification failed: {result.error_message}")
        # Continue with original analysis
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Fallback behavior
```

## Performance Considerations

### Optimization Tips

1. **Batch processing** for multiple content pieces
2. **Priority levels** (fast/balanced/quality) based on needs
3. **Timeout configuration** to prevent hanging requests
4. **Parallel requests** limited to prevent API rate limiting

### Monitoring

```python
# Get system health
health = await api.health_check()
print(f"System status: {health['status']}")

# Get component status
status = api.get_component_status()
operational = sum(1 for s in status.values() if s == "operational")
print(f"Components operational: {operational}/{len(status)}")
```

## Deployment

### Requirements

```txt
# requirements.txt
requests>=2.28.0
beautifulsoup4>=4.11.0
dataclasses>=0.6
typing>=3.7.4
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### Installation

```bash
# Install the retrieval system
pip install -e .

# Or add to your existing requirements
echo "requests beautifulsoup4" >> requirements.txt
```

### Production Configuration

```python
# production_config.py
PRODUCTION_CONFIG = RetrievalConfig(
    max_parallel_requests=10,
    verification_timeout=45.0,
    enable_statistical_apis=True,
    log_level="WARNING"
)
```

## Troubleshooting

### Common Issues

1. **Slow verification**: Reduce parallel requests, check network connectivity
2. **API timeouts**: Increase timeout, check network connectivity
3. **Memory usage**: Monitor component status, reduce parallel requests
4. **Import errors**: Ensure proper Python path configuration

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = RetrievalConfig(log_level="DEBUG")
api = create_retrieval_api(config)
```

### Health Checks

```python
# Regular health monitoring
health = await api.health_check()
if health["status"] != "healthy":
    print("System issues detected:")
    for issue in health.get("issues", []):
        print(f"  - {issue}")
```

## Contributing

### Development Setup

```bash
# Clone and setup
git clone <repository>
cd dimetuverdad
pip install -r requirements.txt

# Run tests
python -m pytest retrieval/tests/ -v

# Add new components following the existing patterns
```

### Code Standards

- Type hints for all function parameters and returns
- Comprehensive docstrings
- Async/await for I/O operations
- Exception handling with specific error types
- Unit tests for all new functionality

## License

This retrieval system is part of the dimetuverdad project for research and educational purposes. See main project license for details.

## Changelog

### v1.0.0
- Initial release with full multi-source verification
- Claim extraction for numerical, temporal, and attribution claims
- Credibility scoring with multi-factor assessment
- Evidence aggregation with weighted decision making
- Temporal verification against known events
- Statistical API integration (INE, Eurostat, WHO, World Bank)
- Integration hooks for analyzer verification
- Comprehensive test suite
- Performance optimization and monitoring