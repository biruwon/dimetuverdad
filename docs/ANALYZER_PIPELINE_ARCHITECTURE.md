# Analyzer Pipeline Architecture

## Overview

The **dimetuverdad analyzer pipeline** is a sophisticated multi-stage content analysis system designed to detect hate speech, disinformation, conspiracy theories, and far-right extremist content in Spanish social media. It combines pattern matching, machine learning, multimodal analysis (text + images), and evidence retrieval to provide comprehensive content categorization with detailed explanations.

---

## 🏗️ Core Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     Analyzer (Main Orchestrator)             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Components:                                            │  │
│  │ • MetricsCollector (performance tracking)             │  │
│  │ • ContentAnalysisRepository (database operations)     │  │
│  │ • TextAnalyzer (text-only content)                    │  │
│  │ • MultimodalAnalyzer (text + media content)           │  │
│  │ • AnalyzerHooks (evidence retrieval integration)      │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Analysis Pipeline Flow

### 1. **Entry Point: `analyze_content()`**

The main orchestration method that routes content to appropriate analyzers:

```python
async def analyze_content(self, tweet_id, tweet_url, username, content, media_urls=None):
    """
    Main pipeline entry point - routes to text or multimodal analysis
    """
```

**Decision Logic:**
- **Has media URLs?** → Route to `MultimodalAnalyzer`
- **Text only?** → Route to `TextAnalyzer`

**Key Features:**
- Async execution with `asyncio.to_thread()` to avoid blocking
- Conditional evidence retrieval integration
- Performance metrics tracking
- Error handling with graceful fallbacks

---

## 🔍 Text Analysis Pipeline (TextAnalyzer)

### Stage 1: Pattern Analysis
**Component:** `PatternAnalyzer`

```
Input: Normalized text content
  ↓
Pattern Detection (regex-based)
  ↓
Category Detection: 13 categories including:
  • hate_speech (highest priority)
  • disinformation
  • conspiracy_theory
  • anti_immigration
  • anti_lgbtq
  • anti_feminism
  • nationalism
  • anti_government
  • historical_revisionism
  • political_general
  • call_to_action
  • general (fallback)
  ↓
Output: AnalysisResult with categories, pattern matches, political context
```

**Pattern Matching Categories:**
- **Hate Speech**: Xenophobic language, dehumanization, violence threats, anti-immigrant scapegoating
- **Disinformation**: False medical/scientific claims, fabricated facts, conspiracy claims
- **Conspiracy Theory**: Hidden agenda narratives, anti-institutional content, cabal theories
- **Anti-Immigration**: Xenophobic rhetoric, anti-immigrant narratives, invasion metaphors
- **Anti-LGBTQ**: Attacks on LGBTQ community, gender ideology criticism, traditional values defense
- **Anti-Feminism**: Anti-feminist rhetoric, traditional gender roles, patriarchy defense
- **Nationalism**: National pride, cultural preservation, patriotic narratives
- **Anti-Government**: Government criticism, institutional distrust, anti-establishment views
- **Historical Revisionism**: Historical reinterpretation, authoritarian regime glorification
- **Political General**: General political discourse, neutral political commentary
- **Call to Action**: Mobilization calls, protest organization, organized activities

### Stage 2: Content Categorization
**Method:** `_categorize_content()`

```
Pattern Results Available?
  ↓
YES → Return primary pattern category (method: "pattern")
  ↓
NO → LLM Fallback Available?
     ↓
     YES → Use LLM for category detection (method: "llm")
     ↓
     NO → Return "general" category
```

**Category Priority (when multiple patterns detected):**
1. `hate_speech` (highest priority)
2. `anti_immigration`
3. `anti_lgbtq`
4. `anti_feminism`
5. `disinformation`
6. `conspiracy_theory`
7. `call_to_action`
8. `nationalism`
9. `anti_government`
10. `historical_revisionism`
11. `political_general`
12. `general` (lowest priority)

### Stage 3: LLM Explanation Generation
**Component:** `EnhancedLLMPipeline`

```
Input: Normalized content + detected category
  ↓
Category-Specific Prompt Generation
  (using EnhancedPromptGenerator)
  ↓
LLM Inference (Ollama: gpt-oss:20b or HuggingFace models)
  ↓
Explanation Parsing & Validation
  ↓
Output: Detailed Spanish explanation (2-3 sentences)
```

**LLM Model Priority Levels:**
- **Fast**: Lightweight models (DistilBERT, TinyBERT) - for development/testing
- **Balanced**: Medium models (RoBERTa Spanish) - default production
- **Quality**: Large models (gpt-oss:20b via Ollama) - best explanations

**Prompt Engineering:**
- Category-specific prompts for nuanced analysis
- Context-aware based on pattern detection results
- Structured output format enforcement
- Spanish language optimization

### Stage 4: Analysis Data Construction
**Method:** `_build_analysis_data()`

```
Pattern Results + LLM Output
  ↓
Structured Analysis Data:
  • category (primary)
  • categories_detected (all found)
  • pattern_matches (detailed regex matches)
  • topic_classification (political context)
  • llm_explanation (human-readable)
  • analysis_method ("pattern" or "llm")
  ↓
Output: ContentAnalysis object
```

---

## 🖼️ Multimodal Analysis Pipeline (MultimodalAnalyzer)

### Stage 1: Media Content Analysis
**Component:** `GeminiMultimodal`

```
Input: Media URLs + text content
  ↓
Media URL Selection (best quality detection)
  ↓
Gemini API Call (gemini-2.5-flash)
  ↓
Structured Response Parsing:
  "CATEGORÍA: [category]
   EXPLICACIÓN: [explanation]"
  ↓
Output: Media analysis with category + description
```

**Media Type Support:**
- Images (JPG, PNG, GIF, WebP)
- Videos (MP4, WebM, AVI, MOV)
- Mixed media detection

**Gemini Model Benefits:**
- Native multimodal understanding (text + images)
- Visual content interpretation (memes, graphics, photos)
- Contextual analysis combining visual and text elements

### Stage 2: Category Determination
**Method:** `_determine_combined_category()`

```
Media Analysis Results
  ↓
Media Category Detected?
  ↓
YES → Use media-determined category
  ↓
NO → Fallback to "general"
```

### Stage 3: Multimodal Explanation
**Method:** `_generate_multimodal_explanation()`

```
Media Description + Text Content
  ↓
Combined Explanation Generation
  ↓
Output: Unified explanation (Spanish)
```

---

## 🔬 Evidence Retrieval Integration

### Conditional Triggering
**Method:** `_should_trigger_evidence_retrieval()`

**Trigger Conditions:**
1. Text-only analysis (multimodal excluded - already has visual verification)
2. High-confidence disinformation detection
3. Conspiracy theory content
4. Content with numerical/statistical claims
5. Far-right bias with potential factual claims

### Enhancement Process
**Method:** `_enhance_with_evidence_retrieval()`

```
Original Analysis Result
  ↓
Claim Extraction
  ↓
Multi-Source Evidence Search:
  • Statistical APIs (INE, Eurostat, WHO, World Bank)
  • Web scraping (Wikipedia, fact-checkers)
  • Temporal consistency checks
  ↓
Verification Report Generation:
  • Overall verdict (VERIFIED/REFUTED/UNVERIFIED)
  • Confidence score (0-1)
  • Claims verified (individual verdicts)
  • Evidence sources (with credibility scores)
  • Contradictions found
  ↓
Enhanced Explanation with Verification Data
  ↓
Output: Updated ContentAnalysis with verification_data
```

**Verification Components:**
- **ClaimExtractor**: Identifies verifiable claims (numerical, temporal, causal)
- **QueryBuilder**: Constructs search queries for fact-checking
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
 category, llm_explanation, analysis_method, analysis_json,
 analysis_timestamp, categories_detected, media_urls, media_analysis,
 media_type, multimodal_analysis, verification_data, verification_confidence)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
```

**Features:**
- Automatic retry logic (max 5 attempts)
- Exponential backoff for database locks
- Connection pooling via `get_db_connection_context()`
- JSON serialization for complex fields
- Proper error handling and logging

**Tweet Retrieval:**
```python
def get_tweets_for_analysis(self, username=None, max_tweets=None, force_reanalyze=False):
    """
    Retrieves unanalyzed tweets or all tweets (if force_reanalyze=True)
    Filters by username if specified
    Limits results if max_tweets provided
    """
```

---

## 📈 Performance Tracking

### MetricsCollector

**Tracked Metrics:**
- Analysis duration (seconds)
- Method used (pattern/llm/multimodal)
- Category detected
- Model used (ollama, gemini, pattern-only)
- Multimodal flag (true/false)

**Performance Summary Output:**
```
📊 Tweet Analyzer Performance Summary
==================================================
⏱️  Duration: 17.05 seconds
🧠 Peak Memory: 350.3 MB
⚡ CPU Usage: 0.0%
🔢 Operations: 1
🚀 Throughput: 0.06 ops/sec
📈 Status: ✅ Success
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

**Configuration (AnalyzerConfig):**
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
        result = await analyzer.analyze_content(...)
```

**Benefits:**
- Prevents resource exhaustion
- Balances throughput vs. system load
- Graceful degradation under load
- Exponential backoff on failures

---

## 🛠️ Configuration System

### AnalyzerConfig

**Key Parameters:**
- `use_llm` (bool): Enable/disable LLM analysis
- `model_priority` (str): "fast", "balanced", "quality"
- `max_concurrency` (int): Concurrent analysis limit
- `max_llm_concurrency` (int): LLM inference limit
- `max_retries` (int): Failure retry attempts
- `retry_delay` (int): Initial retry delay
- `verbose` (bool): Enable detailed logging

**Model Priority Mapping:**
```python
fast = {
    "ollama": "llama3.2:1b",
    "classification": "distilbert-multilingual"
}

balanced = {
    "ollama": "gpt-oss:20b",
    "classification": "roberta-hate"
}

quality = {
    "ollama": "llama3.1:70b",
    "classification": "roberta-spanish"
}
```

---

## 📝 Output Data Model

### ContentAnalysis Object

```python
@dataclass
class ContentAnalysis:
    post_id: str                          # Tweet ID
    post_url: str                         # Tweet URL
    author_username: str                  # Author handle
    post_content: str                     # Original text
    analysis_timestamp: str               # ISO datetime
    category: str                         # Primary category
    categories_detected: List[str]        # All detected categories
    llm_explanation: str                  # Human-readable explanation
    analysis_method: str                  # "pattern" / "llm" / "multimodal"
    pattern_matches: List[Dict]           # Regex pattern details
    topic_classification: Dict            # Political context
    analysis_json: str                    # Full structured data
    media_urls: List[str]                 # Media URLs (if any)
    media_analysis: str                   # Media description (if multimodal)
    media_type: str                       # "image" / "video" / "mixed"
    multimodal_analysis: bool             # Multimodal flag
    verification_data: Optional[Dict]     # Evidence retrieval results
    verification_confidence: float        # Verification confidence (0-1)
    analysis_time_seconds: float          # Processing time
    model_used: str                       # Model identifier
```

---

## 🚀 Usage Examples

### Basic Analysis (Text Only)

```python
from analyzer.analyze_twitter import create_analyzer

# Create analyzer with default config
analyzer = create_analyzer()

# Analyze single tweet
result = await analyzer.analyze_content(
    tweet_id="123456789",
    tweet_url="https://twitter.com/user/status/123456789",
    username="user",
    content="Los inmigrantes nos están invadiendo y colapsan nuestros servicios."
)

print(f"Category: {result.category}")  # hate_speech
print(f"Method: {result.analysis_method}")  # pattern
print(f"Explanation: {result.llm_explanation}")
```

### Multimodal Analysis (Text + Image)

```python
# Analyze tweet with media
result = await analyzer.analyze_content(
    tweet_id="987654321",
    tweet_url="https://twitter.com/user/status/987654321",
    username="user",
    content="Mira este meme sobre la inmigración",
    media_urls=["https://pbs.twimg.com/media/image.jpg"]
)

print(f"Media Type: {result.media_type}")  # image
print(f"Multimodal: {result.multimodal_analysis}")  # True
print(f"Media Analysis: {result.media_analysis}")
```

### Batch Analysis with Evidence Retrieval

```python
# Analyze multiple tweets from database
await analyze_tweets_from_db(
    username="Santi_ABASCAL",
    max_tweets=50,
    force_reanalyze=True  # Reanalyze with updated prompts
)
```

---

## 🔧 Error Handling & Resilience

### Graceful Degradation Strategy

1. **Pattern Detection Fails** → LLM fallback
2. **LLM Fails** → Return "general" category with error explanation
3. **Multimodal Analysis Fails** → Return error message with context
4. **Evidence Retrieval Fails** → Continue with original analysis
5. **Database Lock** → Retry with exponential backoff (max 5 attempts)

### Error Message System

**Constants (ErrorMessages):**
- `LLM_PIPELINE_NOT_AVAILABLE`: "LLM pipeline no disponible..."
- `LLM_CATEGORY_ERROR`: "Error en categorización LLM: {error}"
- `LLM_EXPLANATION_FAILED`: "LLM no generó explicación válida..."
- `MULTIMODAL_ANALYSIS_FAILED`: "Error en análisis multimodal: {error}"

---

## 📊 Performance Characteristics

### Typical Processing Times

**Text-Only Analysis:**
- Pattern-only: 2-5 seconds
- Pattern + LLM: 10-30 seconds (preloaded model)
- Pattern + LLM (cold start): 3-5 minutes (first load)

**Multimodal Analysis:**
- Gemini API call: 5-15 seconds
- Total with text: 15-45 seconds

**Evidence Retrieval:**
- Claim extraction: 1-2 seconds
- Multi-source search: 5-20 seconds
- Total enhancement: 10-30 seconds

### Optimization Strategies

1. **Model Preloading**: `ollama run gpt-oss:20b --keepalive 24h`
2. **Concurrent Processing**: Semaphore-based rate limiting
3. **Pattern-First Approach**: Fast pattern detection before LLM
4. **Selective Evidence Retrieval**: Only for high-value content
5. **Connection Pooling**: Reuse database connections

---

## 🧪 Testing & Validation

### Test Coverage

**Core Components:**
- `TextAnalyzer`: 76% coverage (unit + integration)
- `MultimodalAnalyzer`: 69% coverage (multimodal tests)
- `PatternAnalyzer`: 96% coverage (comprehensive pattern tests)
- `Repository`: 89% coverage (database operations)
- `LLM Models`: 80% coverage (model loading, inference)

**Overall Project Coverage:** 78.77% (above 70% requirement)

### Test Categories

1. **Unit Tests**: Individual component behavior
2. **Integration Tests**: Component interaction flows
3. **Pattern Detection Tests**: Regex validation
4. **LLM Tests**: Model loading, inference, prompts
5. **Database Tests**: CRUD operations, retries
6. **Performance Tests**: Benchmarking utilities

---

## 🔄 Future Enhancements

### Planned Improvements

1. **Real-Time Analysis**: WebSocket streaming for live tweets
2. **Multi-Language Support**: Extend beyond Spanish
3. **Enhanced Evidence Retrieval**: More fact-checking sources
4. **Model Fine-Tuning**: Custom Spanish far-right detection models
5. **Temporal Analysis**: Track narrative evolution over time
6. **Network Analysis**: User interaction patterns
7. **API Endpoints**: RESTful API for external integrations
8. **Dashboard Analytics**: Real-time monitoring interface

---

## 📚 Related Documentation

- **Pattern Detection**: See `analyzer/pattern_analyzer.py` for regex patterns
- **LLM Models**: See `analyzer/llm_models.py` for model configurations
- **Prompts**: See `analyzer/prompts.py` for category-specific prompts
- **Database Schema**: See `scripts/init_database.py` for table structures
- **Web Interface**: See `web/app.py` for visualization components
- **Evidence Retrieval**: See `retrieval/README.md` for verification system

---

## 🎯 Key Takeaways

### Architecture Strengths

✅ **Modular Design**: Clear separation of concerns (text/multimodal/database)
✅ **Flexible Configuration**: Easy model swapping and priority tuning
✅ **Robust Error Handling**: Graceful degradation at every stage
✅ **Performance Optimized**: Async execution, connection pooling, caching
✅ **Evidence-Based**: Verification integration for factual claims
✅ **Comprehensive Coverage**: 13 content categories with nuanced detection
✅ **Production-Ready**: 78% test coverage, retry logic, monitoring

### Workflow Summary

```
Tweet → Route (Text/Multimodal) → Pattern Detection → LLM Analysis
  → Evidence Retrieval (conditional) → Database Storage → Web Visualization
```

**Critical Success Factors:**
1. Pattern-first approach (fast + accurate for obvious cases)
2. LLM fallback (handles nuanced content)
3. Multimodal support (visual content analysis)
4. Evidence retrieval (fact-checking integration)
5. Performance tracking (monitoring and optimization)

---

**Last Updated:** October 14, 2025
**Version:** 1.0
**Author:** dimetuverdad development team
