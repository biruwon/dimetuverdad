# Analyzer Pipeline Architecture

## Overview

The **dimetuverdad analyzer pipeline** is a sophisticated multi-stage content analysis system designed to detect hate speech, disinformation, conspiracy theories, and far-right extremist content in Spanish social media. It combines pattern matching, machine learning, multimodal analysis (text + images), and evidence retrieval to provide comprehensive content categorization with detailed explanations.

---

## 🏗️ Core Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     AnalysisFlowManager                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Components:                                            │  │
│  │ • PatternAnalyzer (fast regex-based detection)        │  │
│  │ • LocalMultimodalAnalyzer (Ollama gpt-oss:20b)        │  │
│  │ • ExternalAnalyzer (Gemini 2.5 Flash API)             │  │
│  │ • AnalyzerHooks (evidence retrieval integration)      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Analysis Pipeline Flow

### 1. **Entry Point: `AnalysisFlowManager.analyze_full()`**

The main orchestration method that routes content through the complete 3-stage pipeline:

```python
async def analyze_full(
    self,
    content: str,
    media_urls: Optional[List[str]] = None,
    admin_override: bool = False,
    force_disable_external: bool = False
) -> AnalysisResult:
```

**Decision Logic:**
- **Always runs**: Pattern Detection + Local LLM Analysis
- **Conditional**: External Analysis (Gemini) when:
  - Category is NOT `general` OR `political_general`
  - OR `admin_override=True` (admin-triggered analysis)

**Key Features:**
- Async execution with proper error handling
- Dual explanation architecture (local + external)
- Evidence retrieval integration
- Performance tracking and logging

---

## 🔍 3-Stage Analysis Pipeline

### Stage 1: Pattern Detection
**Component:** `PatternAnalyzer`

```
Input: Normalized text content
  ↓
Regex-based Pattern Matching (13 categories)
  ↓
Category Detection with Priority Order:
  1. hate_speech (highest priority)
  2. anti_immigration
  3. anti_lgbtq
  4. anti_feminism
  5. disinformation
  6. conspiracy_theory
  7. call_to_action
  8. nationalism
  9. anti_government
  10. historical_revisionism
  11. political_general
  12. general (fallback)
  ↓
Output: PatternResult with categories, pattern_matches, political_context
```

**Pattern Matching Categories:**
- **Hate Speech**: Xenophobic language, dehumanization, violence threats, anti-immigrant scapegoating
- **Anti-Immigration**: Xenophobic rhetoric, anti-immigrant narratives, invasion metaphors
- **Anti-LGBTQ**: Attacks on LGBTQ community, gender ideology criticism, traditional values defense
- **Anti-Feminism**: Anti-feminist rhetoric, traditional gender roles, patriarchy defense
- **Disinformation**: False medical/scientific claims, fabricated facts, conspiracy claims presented as fact
- **Conspiracy Theory**: Hidden agenda narratives, anti-institutional content, "deep state" theories
- **Call to Action**: Mobilization calls, protest organization, organized activities
- **Nationalism**: National identity emphasis, patriotic rhetoric, cultural preservation themes
- **Anti-Government**: Government criticism, anti-establishment views, institutional distrust
- **Historical Revisionism**: Historical reinterpretation, authoritarian regime glorification
- **Political General**: General political discourse, neutral political commentary
- **General**: Neutral content, no problematic patterns detected

### Stage 2: Local LLM Analysis
**Component:** `LocalMultimodalAnalyzer` (Ollama gpt-oss:20b)

```
Pattern Results Available?
  ↓
YES → LLM Explanation Only (for known category)
  ↓
NO → LLM Categorization + Explanation
  ↓
Enhanced Prompt Generation (Spanish-optimized)
  ↓
Ollama API Call (gpt-oss:20b, ~30-60s)
  ↓
Structured Response Parsing:
  "CATEGORÍA: category_name
   EXPLICACIÓN: explanation_text"
  ↓
Output: Local category + explanation (Spanish, 2-3 sentences)
```

**LLM Operation Modes:**
- **Categorize + Explain**: When patterns insufficient (full analysis)
- **Explain Only**: When patterns found specific category (targeted explanation)

**Model Configuration:**
- **Primary Model**: `gemma3:4b` (fast, multimodal-capable)
- **Fallback Model**: `gpt-oss:20b` (quality explanations)
- **Temperature**: 0.3 (consistent categorization)
- **Max Tokens**: 512 (concise explanations)

### Stage 3: External Analysis (Conditional)
**Component:** `ExternalAnalyzer` (Gemini 2.5 Flash)

```
Category NOT general/political_general OR admin_override=True?
  ↓
YES → Independent Gemini Analysis
  ↓
Gemini API Call (gemini-2.5-flash)
  ↓
Multimodal Analysis (text + images/videos)
  ↓
Category Override Logic:
  • If Gemini detects different category → Override local result
  • Special handling for disinformation detection
  ↓
Output: External category + explanation (independent verification)
```

**Trigger Conditions:**
1. **Automatic**: Content categorized as problematic (not general/political_general)
2. **Admin Override**: Manual trigger for any content
3. **Disinformation Priority**: External analysis takes precedence for disinformation detection

**External Analysis Benefits:**
- Independent verification (no bias from local analysis)
- Advanced multimodal understanding
- Higher accuracy for complex cases
- Cross-validation of local results

---

## 🔬 Evidence Retrieval Integration

### Conditional Enhancement
**Component:** `AnalyzerHooks`

**Trigger Conditions:**
1. **Text-only analysis** (multimodal excluded)
2. **High-confidence disinformation detection**
3. **Conspiracy theory content**
4. **Content with numerical/statistical claims**
5. **Far-right bias with potential factual claims**

### Enhancement Process

```
Local Analysis Complete
  ↓
Claim Extraction (numerical, temporal, causal claims)
  ↓
Multi-Source Verification:
  • Statistical APIs (INE, Eurostat, WHO, World Bank)
  • Web scraping (Wikipedia, fact-checkers)
  • Temporal consistency checks
  ↓
Credibility Scoring & Verdict Generation:
  • Overall verdict (VERIFIED/REFUTED/UNVERIFIED)
  • Confidence score (0-1)
  • Individual claim verdicts
  • Evidence sources with credibility ratings
  ↓
Enhanced Explanation with Verification Data
  ↓
Output: Updated AnalysisResult with verification_data
```

**Verification Components:**
- **ClaimExtractor**: Identifies verifiable claims
- **QueryBuilder**: Constructs fact-checking queries
- **EvidenceAggregator**: Combines evidence from multiple sources
- **CredibilityScorer**: Assigns reliability scores to sources
- **ClaimVerifier**: Produces final verification verdict

---

## 💾 Database Operations

### ContentAnalysisRepository

**Save Operation:**
```sql
INSERT OR REPLACE INTO content_analyses
(post_id, post_url, author_username, platform, post_content,
 category, categories_detected, local_explanation, external_explanation,
 analysis_stages, external_analysis_used, analysis_timestamp,
 analysis_json, pattern_matches, topic_classification,
 media_urls, media_type, verification_data, verification_confidence)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Key Fields:**
- **`local_explanation`**: Explanation from Ollama (gpt-oss:20b)
- **`external_explanation`**: Explanation from Gemini (when triggered)
- **`analysis_stages`**: Comma-separated stages executed ("pattern,local_llm,external")
- **`external_analysis_used`**: Boolean flag for external analysis
- **`verification_data`**: JSON with evidence retrieval results
- **`verification_confidence`**: Confidence score (0-1)

**Features:**
- Automatic retry logic (max 5 attempts)
- Exponential backoff for database locks
- Connection pooling via `get_db_connection_context()`
- JSON serialization for complex fields
- Proper error handling and logging

---

## 📈 Performance Tracking

### MetricsCollector

**Tracked Metrics:**
- Analysis duration (seconds)
- Method used (pattern/local_llm/external)
- Category detected
- Model used (ollama, gemini)
- Multimodal flag (true/false)
- Evidence retrieval triggered
- Verification confidence scores

**Performance Summary Output:**
```
📊 Analysis Performance Summary
==================================================
⏱️  Duration: 45.23 seconds
🧠 Peak Memory: 420.7 MB
⚡ CPU Usage: 0.0%
🔢 Operations: 1
🚀 Throughput: 0.02 ops/sec
📈 Status: ✅ Success
🔍 Stages: pattern,local_llm,external
📋 Category: disinformation
```

---

## 🎯 Category Detection System

### Category Hierarchy (Priority Order)

1. **`hate_speech`** 🚫
   - Direct attacks, slurs, dehumanization
   - Xenophobia, anti-immigrant rhetoric
   - Violence threats, eliminationist language
   - **Highest Priority**: Overrides all other categories

2. **`anti_immigration`** 🚫
   - Xenophobic rhetoric, anti-immigrant narratives
   - Invasion metaphors, border security alarmism
   - Cultural replacement fears, demographic concerns

3. **`anti_lgbtq`** 🏳️‍🌈
   - Attacks on LGBTQ community and rights
   - Gender ideology criticism, traditional values defense
   - Anti-trans rhetoric, conversion therapy promotion

4. **`anti_feminism`** 👩
   - Anti-feminist rhetoric, traditional gender roles
   - Patriarchy defense, masculinity crisis narratives
   - Criticism of equality movements

5. **`disinformation`** ❌
   - False medical/scientific claims
   - Fabricated statistics
   - Misleading information
   - Conspiracy claims presented as fact

6. **`conspiracy_theory`** 🕵️
   - Hidden agenda narratives
   - Anti-institutional claims
   - "Deep state" / cabal theories
   - Secretive elite control narratives

7. **`call_to_action`** 📢
   - Mobilization calls
   - Protest organization
   - Collective action requests
   - Event promotion

8. **`nationalism`** 🇪�
   - National identity emphasis
   - Patriotic rhetoric
   - Cultural preservation themes

9. **`anti_government`** 🏛️
   - Government criticism
   - Anti-establishment views
   - Institutional distrust

10. **`historical_revisionism`** 📜
    - Reinterpretation of historical events
    - Minimization of atrocities
    - Glorification of authoritarian regimes

11. **`political_general`** 🗳️
    - General political discourse
    - Neutral political commentary

12. **`general`** ✅
    - Neutral content
    - No problematic patterns detected
    - **Fallback Category**: Default when no patterns match

---

## 🔄 Async Pipeline Execution

### Concurrency Control

**Configuration:**
```python
max_concurrency = 10          # Total concurrent analyses
max_llm_concurrency = 3       # LLM inference limit
max_retries = 5               # Failed analysis retries
retry_delay = 2               # Initial retry delay (seconds)
```

**Semaphore-Based Rate Limiting:**
```python
analysis_sema = asyncio.Semaphore(max_concurrency)
llm_sema = asyncio.Semaphore(max_llm_concurrency)

async with analysis_sema:
    async with llm_sema:
        result = await flow_manager.analyze_full(content, media_urls)
```

**Benefits:**
- Prevents resource exhaustion
- Balances throughput vs. system load
- Graceful degradation under load
- Exponential backoff on failures

---

## 🖼️ Multimodal Analysis Capabilities

### Local Multimodal (Ollama)
**Component:** `LocalMultimodalAnalyzer`

```
Text + Media URLs
  ↓
Media Content Preparation:
  • Download images (skip videos)
  • Convert to base64
  • Limit to 3 media files
  ↓
Ollama Multimodal API Call (gemma3:4b)
  ↓
Combined Text + Visual Analysis
  ↓
Output: Unified explanation considering both modalities
```

**Supported Media:**
- Images: JPG, PNG, GIF, WebP, BMP
- Videos: Skipped (Ollama limitation)
- Mixed content: Images processed, videos ignored

### External Multimodal (Gemini)
**Component:** `ExternalAnalyzer`

```
Text + Media URLs
  ↓
Gemini 2.5 Flash API Call
  ↓
Native Multimodal Understanding
  ↓
Independent Category + Explanation
  ↓
Cross-Validation with Local Results
```

**Advantages:**
- Native video support
- Superior multimodal understanding
- Independent verification
- Higher accuracy for visual content

---

## 📝 Output Data Model

### AnalysisResult Object

```python
@dataclass
class AnalysisResult:
    category: str                          # Final category (may be overridden by external)
    local_explanation: str                 # Explanation from Ollama
    stages: AnalysisStages                 # Which stages were executed
    pattern_data: dict                     # Pattern detection results
    verification_data: dict                # Evidence retrieval results
    external_explanation: Optional[str]    # Explanation from Gemini (if triggered)
```

### AnalysisStages Tracking

```python
@dataclass
class AnalysisStages:
    pattern: bool = False      # Pattern detection executed
    local_llm: bool = False    # Local LLM analysis executed
    external: bool = False     # External analysis executed
    
    def to_string(self) -> str:
        """Convert to database format: 'pattern,local_llm,external'"""
```

### Database ContentAnalysis Record

```python
# Key fields in content_analyses table
{
    "post_id": "1234567890",
    "category": "disinformation",
    "local_explanation": "Este contenido presenta información falsa...",
    "external_explanation": "El contenido difunde datos manipulados...",
    "analysis_stages": "pattern,local_llm,external",
    "external_analysis_used": true,
    "verification_data": {...},           # Evidence retrieval results
    "verification_confidence": 0.85,      # Confidence score
    "pattern_matches": [...],             # Detailed regex matches
    "topic_classification": {...}         # Political context
}
```

---

## 🚀 Usage Examples

### Complete Analysis Pipeline

```python
from analyzer.flow_manager import AnalysisFlowManager

# Initialize flow manager
flow_manager = AnalysisFlowManager(verbose=True)

# Run complete 3-stage analysis
result = await flow_manager.analyze_full(
    content="Los inmigrantes están destruyendo España según datos oficiales",
    media_urls=["https://example.com/chart.jpg"],
    admin_override=False  # Let system decide external analysis
)

print(f"Category: {result.category}")
print(f"Stages: {result.stages.to_string()}")
print(f"Local: {result.local_explanation}")
if result.external_explanation:
    print(f"External: {result.external_explanation}")
```

### Local Analysis Only

```python
# Run pattern + local LLM only (no external)
result = await flow_manager.analyze_local(
    content="Contenido político normal",
    media_urls=None
)

print(f"Category: {result.category}")
print(f"Local Explanation: {result.local_explanation}")
print(f"Verification Data: {result.verification_data}")
```

### External Analysis Only

```python
# Run independent external analysis
external_result = await flow_manager.analyze_external(
    content="Contenido sospechoso",
    media_urls=["https://example.com/image.jpg"]
)

print(f"External Category: {external_result.category}")
print(f"External Explanation: {external_result.explanation}")
```

---

## 🔧 Error Handling & Resilience

### Graceful Degradation Strategy

1. **Pattern Detection Fails** → Continue to LLM analysis
2. **Local LLM Fails** → Return pattern-based result with error explanation
3. **External Analysis Fails** → Continue with local result only
4. **Evidence Retrieval Fails** → Continue with original analysis
5. **Database Lock** → Retry with exponential backoff (max 5 attempts)

### Error Message System

**Constants (ErrorMessages):**
- `LLM_PIPELINE_NOT_AVAILABLE`: "Análisis LLM no disponible..."
- `EXTERNAL_ANALYSIS_FAILED`: "Error en análisis externo: {error}"
- `EVIDENCE_RETRIEVAL_FAILED`: "Error en verificación de evidencia: {error}"

---

## 📊 Performance Characteristics

### Typical Processing Times

**Pattern-Only Analysis:**
- Processing: 2-5 seconds
- Memory: ~50MB
- CPU: Minimal

**Local LLM Analysis:**
- Pattern + LLM: 10-30 seconds (preloaded model)
- Pattern + LLM (cold start): 3-5 minutes
- Memory: ~200-400MB
- CPU: Moderate

**Complete Pipeline (with External):**
- Full analysis: 30-60 seconds
- Memory: ~400-600MB
- CPU: Moderate to high

**Evidence Retrieval:**
- Claim extraction: 1-2 seconds
- Multi-source search: 5-20 seconds
- Total enhancement: 10-30 seconds

### Optimization Strategies

1. **Model Preloading**: `ollama run gemma3:4b --keepalive 24h`
2. **Concurrent Processing**: Semaphore-based rate limiting
3. **Pattern-First Approach**: Fast detection before expensive LLM calls
4. **Selective External Analysis**: Only for high-value content
5. **Connection Pooling**: Reuse database connections

---

## 🧪 Testing & Validation

### Test Coverage

**Core Components:**
- `AnalysisFlowManager`: 89% coverage (pipeline orchestration)
- `LocalMultimodalAnalyzer`: 82% coverage (Ollama integration)
- `ExternalAnalyzer`: 76% coverage (Gemini API)
- `PatternAnalyzer`: 96% coverage (regex validation)
- `Repository`: 89% coverage (database operations)

**Overall Project Coverage:** 78.77% (above 70% requirement)

### Test Categories

1. **Unit Tests**: Individual component behavior
2. **Integration Tests**: Pipeline flow validation
3. **Pattern Detection Tests**: Regex accuracy validation
4. **LLM Tests**: Model loading, inference, prompt validation
5. **Database Tests**: CRUD operations, retry logic
6. **Performance Tests**: Benchmarking and load testing

---

## 🔄 Future Enhancements

### Planned Improvements

1. **Real-Time Analysis**: WebSocket streaming for live content analysis
2. **Multi-Language Support**: Extend beyond Spanish (English, French, German)
3. **Enhanced Evidence Retrieval**: More fact-checking sources and APIs
4. **Model Fine-Tuning**: Custom Spanish far-right detection models
5. **Temporal Analysis**: Track narrative evolution over time
6. **Network Analysis**: User interaction patterns and influence mapping
7. **API Endpoints**: RESTful API for external integrations
8. **Advanced Dashboard**: Real-time monitoring and analytics
9. **Batch Processing**: High-throughput analysis for large datasets
10. **Model Comparison**: A/B testing between different LLM configurations

---

## 📚 Related Documentation

- **Pattern Detection**: See `analyzer/pattern_analyzer.py` for regex patterns
- **LLM Integration**: See `analyzer/local_analyzer.py` for Ollama integration
- **External Analysis**: See `analyzer/external_analyzer.py` for Gemini integration
- **Flow Orchestration**: See `analyzer/flow_manager.py` for pipeline logic
- **Database Schema**: See `scripts/init_database.py` for table structures
- **Evidence Retrieval**: See `retrieval/README.md` for verification system
- **Web Interface**: See `web/app.py` for visualization components

---

## 🎯 Key Takeaways

### Architecture Strengths

✅ **3-Stage Pipeline**: Pattern → Local LLM → Conditional External Analysis
✅ **Dual Explanations**: Independent local (Ollama) and external (Gemini) verification
✅ **Evidence Integration**: Fact-checking for claims-based content
✅ **Modular Design**: Clear separation of concerns, easy component swapping
✅ **Performance Optimized**: Async execution, rate limiting, connection pooling
✅ **Comprehensive Categories**: 13 categories with nuanced detection
✅ **Production-Ready**: 78% test coverage, retry logic, error handling

### Workflow Summary

```
Content → Pattern Detection (2-5s) → Local LLM Analysis (10-30s)
  → Conditional External Analysis (30-60s) → Evidence Retrieval (if needed)
  → Database Storage → Web Visualization
```

**Critical Success Factors:**
1. **Pattern-First Approach**: Fast, rule-based detection for obvious cases
2. **LLM Fallback**: Handles nuanced content with Spanish-optimized prompts
3. **External Validation**: Independent verification for high-confidence cases
4. **Evidence Enhancement**: Fact-checking integration for factual claims
5. **Dual Explanations**: Multiple perspectives for comprehensive analysis

---

**Last Updated:** October 29, 2025
**Version:** 2.0
**Author:** dimetuverdad development team
