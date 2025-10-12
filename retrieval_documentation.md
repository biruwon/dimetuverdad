# Evidence Retrieval Module

This module provides functionality to search for evidence and fact-checking information from curated sources, particularly focused on Spanish news outlets and fact-checking websites.

## Overview

The `retrieval.py` module is designed to:
- Extract relevant search terms from text content
- Query multiple trusted sources for related information
- Fetch and analyze links for potential debunking content
- Format results for easy consumption

## Key Components

### Spanish Stopwords
A curated list of common Spanish words that are filtered out when creating search queries:
- Common words like 'que', 'de', 'la', 'el', 'y', 'a', 'en', etc.
- Helps improve search quality by focusing on meaningful terms

### Source Templates
Predefined search URL templates for various trusted sources:

#### Fact-Checking Sites
- **Maldita.es** - Spanish fact-checking organization
- **Newtral** - Spanish verification platform
- **Snopes** - International fact-checking website
- **PolitiFact** - Political fact-checking service
- **FactCheck.org** - Non-partisan fact-checking organization

#### News Sources
- **Google News RSS** - News aggregation via RSS
- **Bing News RSS** - Microsoft's news search via RSS
- **El País** - Spanish newspaper search
- **El Mundo** - Spanish newspaper search
- **ABC** - Spanish newspaper search
- **La Vanguardia** - Spanish newspaper search

### Debunk Detection
The module includes keywords that commonly indicate debunking or fact-checking content:
- 'falso', 'falsedad', 'desmiente', 'desmentido'
- 'no es cierto', 'mentira', 'error', 'corrección'
- 'verifica', 'verificado', 'comprobado', etc.

## Main Functions

### `extract_query_terms(text: str, max_terms: int = 6) -> str`
Extracts the most relevant search terms from input text by:
- Removing URLs and filtering stopwords
- Analyzing word frequency
- Returning the most common meaningful terms

### `build_search_urls(query: str, sources: List[str] = None) -> List[Dict[str,str]]`
Constructs search URLs for specified sources using the query terms.

### `fetch_links_from_search_url(url: str, max_links: int = 3, timeout: int = 6) -> List[Dict[str,str]]`
Fetches and parses search results from a given URL, supporting both:
- RSS/XML feeds (for news aggregators)
- HTML pages (for website searches)

Each result includes automatic debunk detection by analyzing page content.

### `retrieve_evidence_for_post(text: str, max_per_source: int = 2, sources: List[str] = None) -> List[Dict[str,str]]`
Main function that orchestrates the evidence retrieval process:
- Extracts query terms from input text
- Searches across trusted sources
- Returns structured results with links and metadata

### `format_evidence(results: List[Dict[str,str]]) -> str`
Formats the search results into a readable string format for display.

## Usage Example

```python
from retrieval import retrieve_evidence_for_post, format_evidence

# Analyze a piece of text for fact-checking
text = "Nueva vacuna causa efectos secundarios graves"
evidence = retrieve_evidence_for_post(text)
formatted_output = format_evidence(evidence)
print(formatted_output)
```

---

# Evidence Retrieval Enhancement Proposal

## Executive Summary

The current `retrieval.py` module provides basic fact-checking capabilities by searching curated sources for debunking content. However, it lacks the sophistication needed to serve as a "source of trust" for verifying disinformation and numerical/statistical claims with up-to-date data. This proposal outlines a comprehensive enhancement plan to transform the retrieval system into a robust evidence verification engine that can be seamlessly integrated into the dimetuverdad analyzer pipeline.

## Current State Analysis

### Strengths
- **Curated Sources**: Well-selected mix of Spanish and international fact-checking sites
- **Multi-format Support**: Handles both RSS feeds and HTML search results
- **Debunk Detection**: Basic keyword-based identification of corrective content
- **Modular Design**: Clean separation of concerns with focused functions

### Critical Limitations

#### 1. **Query Generation Issues**
- Generic frequency-based term extraction misses numerical claims
- No special handling for statistical data, dates, or quantitative assertions
- Fails to identify verification-worthy claims automatically

#### 2. **Source Coverage Gaps**
- Limited to journalistic and fact-checking sources
- No access to statistical databases, government data, or academic sources
- Missing real-time data APIs for current statistics
- No source credibility scoring or freshness assessment

#### 3. **Verification Logic Deficiencies**
- Primitive keyword matching for debunk detection
- No confidence scoring or evidence strength assessment
- Cannot verify numerical claims against authoritative data
- No temporal awareness (data freshness validation)

#### 4. **Integration Limitations**
- Not integrated with analyzer pipeline
- No automatic triggering based on content analysis
- Results not structured for LLM consumption
- No caching or performance optimization

## Proposed Enhancements

### Phase 1: Core Retrieval Improvements

#### Enhanced Query Generation
```python
def extract_verification_targets(text: str) -> List[VerificationTarget]:
    """
    Extract claims that warrant verification:
    - Numerical/statistical claims
    - Dated information
    - Attribution claims
    - Causal assertions
    """
```

**Key Features:**
- **Numerical Claim Detection**: Regex patterns for percentages, counts, dates
- **Claim Type Classification**: Statistics, attributions, temporal claims
- **Context Preservation**: Maintain surrounding context for accurate verification
- **Priority Scoring**: Rank claims by verification importance

#### Expanded Source Ecosystem
```python
SOURCE_CATEGORIES = {
    'fact_checkers': [...],  # Current fact-checking sites
    'statistical_agencies': [
        'ine.es',  # Spanish National Statistics Institute
        'eurostat.europa.eu',
        'worldbank.org/data',
        'oecd.org/statistics'
    ],
    'government_sources': [
        'boe.es',  # Spanish Official Gazette
        'mscbs.gob.es',  # Health Ministry
        'exteriores.gob.es'  # Foreign Affairs
    ],
    'academic_databases': [
        'scholar.google.es',
        'dialnet.unirioja.es'
    ]
}
```

#### Real-time Data APIs
- **INE (Spanish Statistics Institute)**: Population, economic indicators
- **WHO/OMS**: Health statistics and pandemic data
- **European Commission**: EU policy and implementation data
- **World Bank/Open Data**: Global development indicators

### Phase 2: Advanced Verification Logic

#### Multi-layered Verification
1. **Pattern-based Verification**: Check against known false claims databases
2. **Statistical Verification**: Compare numbers against authoritative sources
3. **Temporal Verification**: Validate date-sensitive information
4. **Source Credibility Assessment**: Weighted scoring based on source reputation

#### Confidence Scoring System
```python
@dataclass
class VerificationResult:
    claim: str
    verdict: Literal['verified', 'debunked', 'unclear']
    confidence: float  # 0.0 to 1.0
    evidence_sources: List[EvidenceSource]
    explanation: str
    last_updated: datetime
```

### Phase 3: Analyzer Integration

#### Trigger Conditions
The retrieval system should be automatically triggered when:

1. **Pattern Analysis Results**:
   - Category: `disinformation` (confidence > 0.7)
   - Category: `conspiracy_theory` (any detection)
   - Numerical claims detected in text

2. **LLM Analysis Enhancement**:
   - When LLM identifies claims needing verification
   - For complex statistical claims
   - When pattern analysis is inconclusive

3. **Content Characteristics**:
   - Contains numbers/statistics
   - Makes temporal claims (dates, timelines)
   - Attributes information to sources
   - Contains policy or implementation claims

#### Integration Architecture
```python
class EnhancedAnalyzer:
    def analyze_content(self, ...):
        # Step 1: Pattern + LLM analysis (current)
        initial_result = self._run_initial_analysis(content)
        
        # Step 2: Conditional evidence retrieval
        if self._should_verify_claims(initial_result, content):
            evidence_results = self.retrieval_engine.verify_claims(content)
            enhanced_explanation = self._enhance_with_evidence(
                initial_result.llm_explanation, 
                evidence_results
            )
            initial_result.llm_explanation = enhanced_explanation
        
        return initial_result
```

## When to Use Evidence Retrieval

### Optimal Trigger Conditions

#### 1. **High-Confidence Disinformation Detection**
```python
# Trigger when pattern analysis detects disinformation with high confidence
if analysis_result.category == Categories.DISINFORMATION and pattern_confidence > 0.7:
    trigger_evidence_retrieval(content)
```

#### 2. **Numerical/Statistical Claims**
```python
# Trigger for content containing quantitative claims
if contains_numerical_claims(content):
    trigger_evidence_retrieval(content, focus='numerical')
```

#### 3. **LLM-Requested Verification**
```python
# Trigger when LLM analysis indicates need for evidence checking
if llm_response_contains_verification_request(llm_explanation):
    trigger_evidence_retrieval(content)
```

#### 4. **Conspiracy Theory Detection**
```python
# Trigger for conspiracy-related content
if Categories.CONSPIRACY_THEORY in detected_categories:
    trigger_evidence_retrieval(content, focus='conspiracy')
```

### Anti-Trigger Conditions (When NOT to Use)

#### 1. **Low-Value Content**
- General political discussion without specific claims
- Pure opinion pieces
- Content already verified in recent analysis

#### 2. **Already Verified Content**
- Content from trusted sources with established credibility
- Factual reporting from mainstream media
- Official government announcements

## Implementation Roadmap

### Month 1: Foundation (Weeks 1-4)
- [ ] Enhance query generation for numerical claims
- [ ] Add statistical source APIs
- [ ] Implement basic claim extraction
- [ ] Create verification result data structures

### Month 2: Core Verification (Weeks 5-8)
- [ ] Build multi-source verification logic
- [ ] Implement confidence scoring
- [ ] Add temporal validation
- [ ] Create evidence aggregation system

### Month 3: Integration (Weeks 9-12)
- [ ] Integrate with analyzer pipeline
- [ ] Implement conditional triggering
- [ ] Add LLM prompt enhancement
- [ ] Performance optimization and caching

### Month 4: Production Readiness (Weeks 13-16)
- [ ] Comprehensive testing
- [ ] Error handling and resilience
- [ ] Monitoring and metrics
- [ ] Documentation and training

## Technical Architecture

### Component Structure
```
retrieval/
├── core/
│   ├── claim_extractor.py      # Extract verification targets
│   ├── query_builder.py        # Build search queries
│   └── evidence_aggregator.py  # Combine multiple sources
├── sources/
│   ├── fact_checkers.py        # Existing + enhanced
│   ├── statistical_apis.py     # New real-time APIs
│   └── web_scrapers.py         # Enhanced web scraping
├── verification/
│   ├── numerical_verifier.py   # Statistical claim checking
│   ├── temporal_verifier.py    # Date/time validation
│   └── credibility_scorer.py   # Source reputation
└── integration/
    ├── analyzer_hooks.py       # Analyzer integration points
    └── result_formatter.py     # Format for LLM consumption
```

### Data Flow
```
Content Input → Claim Extraction → Query Generation → Multi-Source Search → Evidence Aggregation → Verification Scoring → Result Enhancement → Analyzer Output
```

## Success Metrics

### Accuracy Metrics
- **Verification Precision**: % of correct verdicts on known claims
- **False Positive Rate**: % of incorrect debunkings
- **Coverage**: % of verifiable claims detected and processed

### Performance Metrics
- **Response Time**: Average time for evidence retrieval (< 5 seconds)
- **Cache Hit Rate**: % of requests served from cache (> 70%)
- **API Reliability**: % of successful API calls (> 95%)

### Integration Metrics
- **Trigger Accuracy**: % of cases where verification was appropriately triggered
- **Analysis Enhancement**: Improvement in LLM explanation quality
- **User Trust**: Reduction in false positives from analyzer

## Risk Assessment & Mitigation

### Technical Risks
- **API Rate Limiting**: Implement caching, request pooling, and fallback sources
- **Source Reliability**: Multi-source verification with credibility weighting
- **Performance Impact**: Asynchronous processing and smart caching

### Content Risks
- **Verification Bias**: Use diverse sources and transparent methodology
- **Cultural Context**: Spanish-specific fact-checking with international validation
- **Temporal Decay**: Implement freshness checks and update mechanisms

## Testing Strategy

### Unit Testing
- Individual component testing with mock data
- API integration testing with rate limit simulation
- Claim extraction accuracy testing

### Integration Testing
- End-to-end verification workflows
- Analyzer pipeline integration testing
- Performance benchmarking

### Validation Testing
- Known false claims database testing
- Statistical claim verification testing
- Temporal validation testing

## Conclusion

This enhancement proposal transforms the basic retrieval module into a sophisticated evidence verification system that can serve as a trusted source for validating disinformation and numerical claims. The phased approach ensures incremental value delivery while building toward a comprehensive solution.

The integration with the analyzer pipeline will significantly improve the accuracy and trustworthiness of content analysis results, particularly for Spanish far-right content where disinformation and statistical manipulation are common tactics.

## Next Steps

1. **Immediate Action**: Begin Phase 1 implementation with claim extraction enhancements
2. **Stakeholder Review**: Present proposal to development team for feedback
3. **Resource Allocation**: Identify team members for each phase
4. **Timeline Confirmation**: Adjust roadmap based on available resources

---

*This proposal represents a strategic enhancement to make dimetuverdad a more robust and trustworthy content analysis system for combating disinformation in Spanish political discourse.*