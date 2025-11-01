# Multi-Category Support Implementation Plan

## 📋 Overview

This document outlines the comprehensive plan for implementing multi-category support in the dimetuverdad content analysis system. Currently, the system forces each post to have a single primary category. This plan enables posts to be assigned multiple categories simultaneously, providing more accurate and nuanced content analysis.

### Example Use Case

**Post 1979108828265250874** (VOX tweet about Arabic classes):
- Content: "En 121 centros de Cataluña se imparten clases de Lengua Árabe y Cultura Marroquí. FIRMA PARA PONERLE FIN [link to VOX petition]"
- **Current classification:** `disinformation` (single category)
- **Desired classification:** `anti_immigration`, `call_to_action`, `disinformation` (multiple categories)

---

## 📊 Current State Analysis

### Current Architecture Limitations

1. **Pattern Analyzer** - Already detects multiple categories (`detected_categories` list) but only uses first one as `primary_category`
2. **Flow Manager** - Reduces analysis to single `category` string
3. **LLM Prompts** - Request "UNA categoría" (one category only)
4. **Database** - Has `categories_detected` TEXT field (currently unused) but `category` is single value
5. **Web Display** - Only shows single category badge per tweet
6. **Multi-model** - Each model votes for single category only

### What Already Works

- ✅ Pattern analyzer detects multiple patterns simultaneously
- ✅ Database has `categories_detected` field (JSON array storage ready)
- ✅ Category badge styling already defined for all categories
- ✅ `AnalysisResult.categories` field exists in pattern analyzer

---

## 🔧 Implementation Plan

### Phase 1: Category Priority System

**Goal:** Define severity hierarchy for selecting primary category from multiple detected categories.

#### 1.1 Add Priority Configuration

Create priority hierarchy in `analyzer/categories.py`:

```python
# Category priority hierarchy (1 = highest severity/priority)
CATEGORY_PRIORITY = {
    # Identity-based hate & discrimination (highest priority)
    Categories.HATE_SPEECH: 1,
    
    # Information warfare
    Categories.DISINFORMATION: 2,
    Categories.CONSPIRACY_THEORY: 3,
    
    # Identity attacks
    Categories.ANTI_IMMIGRATION: 4,
    Categories.ANTI_LGBTQ: 4,
    Categories.ANTI_FEMINISM: 4,
    
    # Political mobilization
    Categories.CALL_TO_ACTION: 5,
    
    # Political categories
    Categories.NATIONALISM: 6,
    Categories.ANTI_GOVERNMENT: 6,
    Categories.HISTORICAL_REVISIONISM: 6,
    
    # General political
    Categories.POLITICAL_GENERAL: 7,
    
    # Fallback
    Categories.GENERAL: 8
}

def select_primary_category(categories: List[str]) -> str:
    """
    Select primary category based on priority hierarchy.
    Returns the category with highest priority (lowest number).
    """
    if not categories:
        return Categories.GENERAL
    
    # Filter out invalid categories
    valid_categories = [c for c in categories if c in CATEGORY_PRIORITY]
    
    if not valid_categories:
        return Categories.GENERAL
    
    # Return category with lowest priority number (highest severity)
    return min(valid_categories, key=lambda c: CATEGORY_PRIORITY.get(c, 99))

def get_category_priority(category: str) -> int:
    """Get priority level for a category (lower = higher priority)."""
    return CATEGORY_PRIORITY.get(category, 99)
```

#### 1.2 Files to Modify
- `analyzer/categories.py` - Add priority system
- Add unit tests for priority selection logic

**Estimated time:** 0.5 days

---

### Phase 2: Pattern Analyzer Updates

**Goal:** Enhance pattern analyzer to return both category list and primary category.

#### 2.1 Update AnalysisResult

Already has `categories` list - just need to add primary selection:

```python
@dataclass
class AnalysisResult:
    categories: List[str]  # All detected categories
    pattern_matches: List[PatternMatch]
    primary_category: str  # Selected based on priority
    political_context: List[str]
    keywords: List[str]
```

#### 2.2 Modify `analyze_content()` Method

```python
def analyze_content(self, text: str) -> AnalysisResult:
    # ... existing detection logic ...
    
    # Select primary category based on priority
    from .categories import select_primary_category
    primary_category = select_primary_category(detected_categories) if detected_categories else "non_political"
    
    return AnalysisResult(
        categories=detected_categories,
        pattern_matches=pattern_matches,
        primary_category=primary_category,
        political_context=political_context,
        keywords=keywords
    )
```

#### 2.3 Files to Modify
- `analyzer/pattern_analyzer.py` - Update primary category selection
- `analyzer/tests/test_pattern_analyzer.py` - Add multi-category tests

**Estimated time:** 0.5 days

---

### Phase 3: LLM Prompt Updates

**Goal:** Modify prompts to request and parse multiple categories from LLMs.

#### 3.1 Update System Prompts

Modify `analyzer/prompts.py`:

**Current:**
```python
Clasifica este texto en UNA categoría: {categories}
```

**New:**
```python
Clasifica este texto en UNA O MÁS categorías (si aplican múltiples, separa con comas): {categories}

⚠️ IMPORTANTE SOBRE MÚLTIPLES CATEGORÍAS:
- Un mismo contenido puede pertenecer a MÚLTIPLES categorías simultáneamente
- Identifica TODAS las categorías que apliquen, no solo la más prominente
- Ejemplos de combinaciones comunes:
  * Anti-inmigración + Llamada a acción = contenido con retórica xenófoba que pide firmar petición
  * Desinformación + Teoría conspirativa = información falsa basada en narrativas conspiratorias
  * Discurso de odio + Llamada a acción = contenido agresivo que moviliza contra grupos
  * Anti-inmigración + Desinformación = retórica xenófoba con datos falsos
  * Nacionalismo + Anti-gubernamental = exaltación nacional criticando al gobierno

FORMATO DE RESPUESTA:
CATEGORÍAS: category1, category2, category3
EXPLICACIÓN: [2-3 oraciones explicando por qué pertenece a CADA categoría detectada]
```

#### 3.2 Update Response Parser

Modify `analyzer/response_parser.py`:

```python
def parse_ollama_response(response: str) -> Tuple[List[str], str]:
    """
    Parse Ollama response supporting multiple categories.
    
    Returns:
        Tuple of (categories_list, explanation)
    
    Example input:
        "CATEGORÍAS: anti_immigration, call_to_action
         EXPLICACIÓN: Este contenido combina..."
    """
    lines = response.strip().split('\n')
    categories = []
    explanation = ""
    
    for line in lines:
        line = line.strip()
        
        # Look for categories (singular or plural)
        if line.startswith('CATEGORÍA:') or line.startswith('CATEGORÍAS:'):
            cat_text = line.split(':', 1)[1].strip()
            # Split by comma and clean
            categories = [c.strip() for c in cat_text.split(',')]
        
        # Look for explanation
        elif line.startswith('EXPLICACIÓN:'):
            explanation = line.split(':', 1)[1].strip()
        elif explanation and line:
            # Multi-line explanation
            explanation += ' ' + line
    
    # Fallback if no categories found
    if not categories:
        categories = [Categories.GENERAL]
    
    return categories, explanation
```

#### 3.3 Files to Modify
- `analyzer/prompts.py` - Update text and multimodal prompts
- `analyzer/response_parser.py` - Update parsing logic
- `analyzer/tests/test_response_parser.py` - Add multi-category parsing tests

**Estimated time:** 1 day

---

### Phase 4: Flow Manager Updates

**Goal:** Update analysis flow to handle and store multiple categories.

#### 4.1 Update AnalysisResult Dataclass

```python
@dataclass
class AnalysisResult:
    categories: List[str]  # All detected categories
    primary_category: str  # Most severe/important category
    local_explanation: str
    stages: AnalysisStages
    pattern_data: dict
    verification_data: dict
    external_explanation: Optional[str] = None
```

#### 4.2 Update `analyze_local()` Method

```python
async def analyze_local(self, content: str, media_urls: Optional[List[str]] = None) -> AnalysisResult:
    # ... existing pattern detection ...
    
    # Stage 2: Local LLM Analysis
    if patterns_found_specific_category:
        primary_category = pattern_result.primary_category
        categories = pattern_result.categories
        local_explanation = await self.local_llm.explain_only(content, categories, media_urls)
    else:
        categories, local_explanation = await self.local_llm.categorize_and_explain(content, media_urls)
        primary_category = select_primary_category(categories)
    
    # Combine pattern and LLM categories (union)
    all_categories = list(set(pattern_result.categories + categories))
    
    # Re-select primary from combined set
    final_primary = select_primary_category(all_categories)
    
    # ... verification logic ...
    
    return AnalysisResult(
        categories=all_categories,
        primary_category=final_primary,
        local_explanation=local_explanation,
        stages=stages,
        pattern_data=pattern_data,
        verification_data=verification_data
    )
```

#### 4.3 Files to Modify
- `analyzer/flow_manager.py` - Update to handle category lists
- `analyzer/local_analyzer.py` - Update methods to return category lists
- `analyzer/external_analyzer.py` - Update to handle multiple categories
- `analyzer/tests/test_flow_manager.py` - Add multi-category flow tests

**Estimated time:** 1 day

---

### Phase 5: Database Repository Updates

**Goal:** Store and retrieve multiple categories from database.

#### 5.1 Update `save_analysis()` Method

In `analyzer/repository.py`:

```python
def save_analysis(
    self,
    post_id: str,
    categories: List[str],  # NEW: list of all categories
    primary_category: str,  # NEW: most important category
    local_explanation: str,
    external_explanation: Optional[str] = None,
    analysis_stages: str = "",
    pattern_data: Optional[dict] = None,
    verification_data: Optional[dict] = None,
    **kwargs
) -> bool:
    """Save analysis with multiple categories."""
    
    # Serialize categories as JSON
    categories_json = json.dumps(categories)
    pattern_json = json.dumps(pattern_data) if pattern_data else None
    verification_json = json.dumps(verification_data) if verification_data else None
    
    query = """
        INSERT INTO content_analyses (
            post_id, category, categories_detected, 
            local_explanation, external_explanation,
            analysis_stages, pattern_matches, 
            topic_classification, verification_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(post_id) DO UPDATE SET
            category = excluded.category,
            categories_detected = excluded.categories_detected,
            local_explanation = excluded.local_explanation,
            external_explanation = excluded.external_explanation,
            analysis_stages = excluded.analysis_stages,
            analysis_timestamp = CURRENT_TIMESTAMP
    """
    
    self.conn.execute(query, (
        post_id,
        primary_category,  # Store as 'category' for backwards compatibility
        categories_json,   # Store full list in 'categories_detected'
        local_explanation,
        external_explanation,
        analysis_stages,
        pattern_json,
        pattern_json,
        verification_json
    ))
    self.conn.commit()
```

#### 5.2 Add Query Methods

```python
def get_tweets_with_any_category(self, categories: List[str]) -> List[Dict]:
    """Get tweets that have any of the specified categories."""
    # Query against categories_detected JSON field
    pass

def get_tweets_with_all_categories(self, categories: List[str]) -> List[Dict]:
    """Get tweets that have all of the specified categories."""
    pass

def get_category_combinations(self) -> Dict[str, int]:
    """Get statistics on category combinations."""
    pass
```

#### 5.3 Files to Modify
- `analyzer/repository.py` - Update save/retrieve methods
- `database/repositories/interfaces.py` - Update interface definitions
- `analyzer/tests/test_repository.py` - Add multi-category storage tests

**Estimated time:** 1 day

---

### Phase 6: Web Interface Updates

**Goal:** Display multiple category badges in web interface.

#### 6.1 Update Template Display Logic

In `web/templates/user.html` and `web/templates/index.html`:

```html
<!-- Old: Single category badge -->
<span class="badge category-{{ tweet.analysis_category }}">
    {{ tweet.analysis_category }}
</span>

<!-- New: Multiple category badges -->
<div class="category-badges-container d-flex flex-wrap gap-2">
    {% if tweet.categories_detected %}
        {% for category in tweet.categories_detected %}
        <span class="badge category-{{ category }} category-badge 
                     {% if category == tweet.analysis_category %}primary-badge{% else %}secondary-badge{% endif %}"
              title="{% if category == tweet.analysis_category %}Categoría principal{% else %}Categoría secundaria{% endif %}">
            {{ get_category_emoji(category) }} 
            {{ get_category_display_name(category) }}
        </span>
        {% endfor %}
    {% else %}
        <!-- Fallback for old single-category format -->
        <span class="badge category-{{ tweet.analysis_category }} category-badge primary-badge">
            {{ get_category_emoji(tweet.analysis_category) }} 
            {{ get_category_display_name(tweet.analysis_category) }}
        </span>
    {% endif %}
</div>
```

#### 6.2 Add CSS Styling

```css
/* Category badge container */
.category-badges-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

/* Primary badge - most important category */
.primary-badge {
    font-size: 0.9rem;
    font-weight: 700;
    border: 2px solid rgba(255, 255, 255, 0.9);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
    padding: 0.4rem 0.8rem;
}

/* Secondary badges - additional categories */
.secondary-badge {
    font-size: 0.8rem;
    font-weight: 500;
    opacity: 0.85;
    border: 1px solid rgba(255, 255, 255, 0.6);
    padding: 0.3rem 0.6rem;
}

/* Hover effects */
.category-badge:hover {
    transform: scale(1.05);
    opacity: 1;
}
```

#### 6.3 Update Helper Functions

In `web/utils/helpers.py`:

```python
def get_category_emoji(category: str) -> str:
    """Get emoji icon for category."""
    emoji_map = {
        'hate_speech': '🚫',
        'disinformation': '❌',
        'conspiracy_theory': '🕵️',
        'anti_immigration': '🚫',
        'anti_lgbtq': '🏳️‍🌈',
        'anti_feminism': '👩',
        'call_to_action': '📢',
        'nationalism': '🇪🇸',
        'anti_government': '🏛️',
        'historical_revisionism': '📜',
        'political_general': '🗳️',
        'general': '✅'
    }
    return emoji_map.get(category, '📄')

def process_tweet_row(row) -> Dict[str, Any]:
    """Process tweet row with multiple categories."""
    tweet_dict = dict(row)
    
    # Parse categories_detected JSON
    if tweet_dict.get('categories_detected'):
        try:
            tweet_dict['categories_detected'] = json.loads(tweet_dict['categories_detected'])
        except:
            tweet_dict['categories_detected'] = [tweet_dict.get('category', 'general')]
    else:
        # Fallback to single category
        tweet_dict['categories_detected'] = [tweet_dict.get('category', 'general')]
    
    return tweet_dict
```

#### 6.4 Update Filters

Add multi-select category filter:

```python
@app.route('/user/<username>')
def user_page(username):
    # ... existing code ...
    
    # Parse multiple category filters
    category_filters = request.args.getlist('category')  # Changed from get() to getlist()
    
    if category_filters:
        # Filter tweets that have ANY of the selected categories
        # Use JSON_EXTRACT or similar for SQLite JSON queries
        pass
```

#### 6.5 Files to Modify
- `web/templates/user.html` - Update category display
- `web/templates/index.html` - Update category display
- `web/templates/admin/edit_analysis.html` - Multi-select categories
- `web/utils/helpers.py` - Add category helper functions
- `web/routes/main.py` - Update filtering logic
- `web/static/css/` - Add multi-category styles

**Estimated time:** 2 days

---

### Phase 7: Multi-Model Analysis Updates

**Goal:** Update multi-model system to handle multiple categories per model.

#### 7.1 Update Model Analysis Storage

In `scripts/analyze_multi_model.py`:

```python
def save_model_analysis(post_id: str, model_name: str, categories: List[str], 
                       primary_category: str, explanation: str):
    """Save multi-category analysis from a single model."""
    categories_json = json.dumps(categories)
    
    # Update model_analyses table schema to include categories_detected
```

#### 7.2 Update Consensus Calculation

```python
def calculate_category_consensus(model_analyses: List[Dict]) -> Dict:
    """
    Calculate consensus for multiple categories.
    Returns categories that appear in >= 50% of models.
    
    Returns:
        {
            'consensus_categories': ['cat1', 'cat2'],
            'primary_category': 'cat1',
            'category_votes': {'cat1': 3, 'cat2': 2, 'cat3': 1},
            'agreement_score': 0.75
        }
    """
    from collections import Counter
    
    category_votes = Counter()
    total_models = len(model_analyses)
    
    # Count votes for each category across all models
    for analysis in model_analyses:
        categories = analysis.get('categories', [analysis.get('category')])
        for category in categories:
            category_votes[category] += 1
    
    # Categories that appear in >= 50% of models
    threshold = total_models / 2
    consensus_categories = [
        cat for cat, votes in category_votes.items()
        if votes >= threshold
    ]
    
    # Select primary from consensus categories
    primary_category = select_primary_category(consensus_categories)
    
    # Calculate agreement score (avg votes per consensus category)
    if consensus_categories:
        avg_votes = sum(category_votes[cat] for cat in consensus_categories) / len(consensus_categories)
        agreement_score = avg_votes / total_models
    else:
        agreement_score = 0.0
    
    return {
        'consensus_categories': consensus_categories,
        'primary_category': primary_category,
        'category_votes': dict(category_votes),
        'agreement_score': agreement_score,
        'total_models': total_models
    }
```

#### 7.3 Update Dashboard Display

In `web/templates/models_dashboard.html`:

```html
<!-- Show consensus categories -->
<div class="consensus-categories mb-3">
    <strong>Consenso:</strong>
    {% for category in analysis.consensus_categories %}
    <span class="badge category-{{ category }} 
                 {% if category == analysis.primary_category %}primary-badge{% else %}secondary-badge{% endif %}">
        {{ category }}
        <small>({{ analysis.category_votes[category] }}/{{ analysis.total_models }} modelos)</small>
    </span>
    {% endfor %}
</div>

<!-- Show individual model categories -->
{% for model_analysis in analysis.model_analyses %}
<div class="model-categories">
    <strong>{{ model_analysis.model_name }}:</strong>
    {% for category in model_analysis.categories %}
    <span class="badge category-{{ category }}">{{ category }}</span>
    {% endfor %}
</div>
{% endfor %}
```

#### 7.4 Files to Modify
- `scripts/analyze_multi_model.py` - Update to handle multi-categories
- `database/database_multi_model.py` - Update consensus logic
- `web/templates/models_dashboard.html` - Update display
- `web/routes/models.py` - Update data processing

**Estimated time:** 1.5 days

---

### Phase 8: Admin Interface Updates

**Goal:** Allow admins to manage multiple categories.

#### 8.1 Update Edit Analysis Form

Replace single dropdown with multi-select:

```html
<div class="mb-3">
    <label class="form-label">
        <i class="fas fa-tags me-2"></i>
        Categorías (selecciona todas las que apliquen)
    </label>
    
    <!-- Multi-select checkboxes -->
    <div class="category-checkboxes">
        {% for category_key, category_name in categories.items() %}
        <div class="form-check">
            <input class="form-check-input" type="checkbox" 
                   name="categories[]" 
                   value="{{ category_key }}"
                   id="cat_{{ category_key }}"
                   {% if category_key in tweet.categories_detected %}checked{% endif %}>
            <label class="form-check-label" for="cat_{{ category_key }}">
                {{ get_category_emoji(category_key) }} {{ category_name }}
            </label>
        </div>
        {% endfor %}
    </div>
</div>

<div class="mb-3">
    <label class="form-label">
        <i class="fas fa-star me-2"></i>
        Categoría Principal
    </label>
    <select class="form-select" name="primary_category" required>
        {% for category_key in tweet.categories_detected %}
        <option value="{{ category_key }}" 
                {% if category_key == tweet.analysis_category %}selected{% endif %}>
            {{ get_category_display_name(category_key) }}
        </option>
        {% endfor %}
    </select>
    <small class="form-text text-muted">
        La categoría principal se usa para filtrado y priorización
    </small>
</div>
```

#### 8.2 Update Save Handler

In `web/routes/admin.py`:

```python
@admin_bp.route('/edit-analysis/<tweet_id>', methods=['POST'])
@admin_required
def admin_edit_analysis(tweet_id: str):
    # Get selected categories
    categories = request.form.getlist('categories[]')
    primary_category = request.form.get('primary_category')
    
    # Validate primary is in categories list
    if primary_category not in categories:
        flash('La categoría principal debe estar en la lista de categorías seleccionadas', 'error')
        return redirect(request.referrer)
    
    # Update database
    content_analysis_repo = get_content_analysis_repository()
    content_analysis_repo.update_categories(
        post_id=tweet_id,
        categories=categories,
        primary_category=primary_category
    )
    
    flash('Categorías actualizadas correctamente', 'success')
    return redirect(request.referrer)
```

#### 8.3 Add Bulk Operations

```python
@admin_bp.route('/bulk-add-category', methods=['POST'])
@admin_required
def bulk_add_category():
    """Add a category to multiple posts matching criteria."""
    target_category = request.form.get('target_category')
    add_category = request.form.get('add_category')
    
    # Example: Add 'call_to_action' to all 'anti_immigration' posts with external links
    # Implementation here
    pass
```

#### 8.4 Files to Modify
- `web/templates/admin/edit_analysis.html` - Multi-select interface
- `web/routes/admin.py` - Update handlers
- `database/repositories/sqlite_impl.py` - Add update_categories method

**Estimated time:** 1 day

---

### Phase 9: Testing & Validation

**Goal:** Comprehensive testing of multi-category system.

#### 9.1 Unit Tests

Create/update test files:

```python
# tests/test_category_priority.py
def test_select_primary_category():
    """Test primary category selection from multiple categories."""
    assert select_primary_category(['call_to_action', 'hate_speech']) == 'hate_speech'
    assert select_primary_category(['general', 'disinformation']) == 'disinformation'
    assert select_primary_category(['nationalism', 'call_to_action']) == 'call_to_action'

# tests/test_pattern_analyzer_multi.py
def test_multi_category_detection():
    """Test detection of multiple categories in single post."""
    text = "¡FIRMA LA PETICIÓN contra la invasión islámica! [link]"
    result = pattern_analyzer.analyze_content(text)
    assert 'anti_immigration' in result.categories
    assert 'call_to_action' in result.categories
    assert result.primary_category == 'anti_immigration'

# tests/test_llm_multi_response.py
def test_parse_multi_category_response():
    """Test parsing LLM responses with multiple categories."""
    response = """
    CATEGORÍAS: anti_immigration, call_to_action, disinformation
    EXPLICACIÓN: Este contenido combina retórica anti-inmigración...
    """
    categories, explanation = parse_ollama_response(response)
    assert len(categories) == 3
    assert 'anti_immigration' in categories
```

#### 9.2 Integration Tests

```python
# tests/test_multi_category_flow.py
async def test_full_analysis_flow_multi_category():
    """Test complete analysis flow with multi-category post."""
    content = "VOX petition against Arabic classes"
    
    result = await flow_manager.analyze_local(content)
    
    assert len(result.categories) > 1
    assert result.primary_category in result.categories
    assert 'call_to_action' in result.categories
    assert 'anti_immigration' in result.categories
```

#### 9.3 Web Interface Tests

```python
# web/tests/test_multi_category_display.py
def test_multiple_badges_displayed(client):
    """Test that multiple category badges are displayed."""
    response = client.get('/user/testuser')
    assert b'primary-badge' in response.data
    assert b'secondary-badge' in response.data
```

#### 9.4 Validation Dataset

Create test cases in `tests/fixtures/multi_category_test_cases.json`:

```json
[
  {
    "post_id": "test_001",
    "content": "FIRMA contra inmigración ilegal",
    "expected_categories": ["anti_immigration", "call_to_action"],
    "expected_primary": "anti_immigration"
  },
  {
    "post_id": "test_002",
    "content": "Gobierno oculta datos sobre sustitución demográfica",
    "expected_categories": ["conspiracy_theory", "disinformation", "anti_immigration"],
    "expected_primary": "disinformation"
  }
]
```

#### 9.5 Files to Create/Modify
- `tests/test_category_priority.py` - New
- `analyzer/tests/test_pattern_analyzer.py` - Update with multi-category tests
- `analyzer/tests/test_response_parser.py` - Add multi-category parsing tests
- `web/tests/test_multi_category_display.py` - New
- `tests/fixtures/multi_category_test_cases.json` - New

**Estimated time:** 2 days

---

### Phase 10: Data Migration & Rollout

**Goal:** Migrate existing data and deploy new system.

#### 10.1 Create Migration Script

Create `scripts/migrate_to_multi_category.py`:

```python
"""
Migrate existing single-category analyses to multi-category format.
Re-analyzes recent posts to populate categories_detected field.
"""

import sqlite3
import json
from analyzer.pattern_analyzer import PatternAnalyzer
from analyzer.categories import select_primary_category

def migrate_existing_analyses():
    """Populate categories_detected for existing analyses."""
    conn = sqlite3.connect('accounts.db')
    cursor = conn.cursor()
    
    # Get all analyses without categories_detected
    cursor.execute("""
        SELECT post_id, category, pattern_matches 
        FROM content_analyses 
        WHERE categories_detected IS NULL OR categories_detected = ''
    """)
    
    pattern_analyzer = PatternAnalyzer()
    
    for row in cursor.fetchall():
        post_id, category, pattern_matches = row
        
        # Parse pattern matches to extract all detected categories
        categories = [category]  # Start with current category
        
        if pattern_matches:
            try:
                matches = json.loads(pattern_matches)
                # Extract unique categories from pattern matches
                pattern_cats = list(set(m['category'] for m in matches))
                categories.extend(pattern_cats)
                categories = list(set(categories))  # Deduplicate
            except:
                pass
        
        # Select primary category
        primary = select_primary_category(categories)
        
        # Update database
        conn.execute("""
            UPDATE content_analyses 
            SET categories_detected = ?, category = ?
            WHERE post_id = ?
        """, (json.dumps(categories), primary, post_id))
    
    conn.commit()
    print(f"✅ Migrated {cursor.rowcount} analyses")

def reanalyze_recent_posts(limit=1000):
    """Re-analyze recent posts with new multi-category system."""
    # Implementation here
    pass

if __name__ == '__main__':
    print("🔄 Starting multi-category migration...")
    migrate_existing_analyses()
    print("✅ Migration complete!")
```

#### 10.2 Validation Script

Create `scripts/validate_multi_category.py`:

```python
"""Validate multi-category implementation."""

def validate_database_consistency():
    """Ensure categories_detected and category fields are consistent."""
    pass

def compare_old_vs_new_categorizations():
    """Compare single vs multi-category results."""
    pass

def generate_migration_report():
    """Generate report on migration results."""
    pass
```

#### 10.3 Rollout Plan

1. **Backup Database**
   ```bash
   ./run_in_venv.sh backup-db
   ```

2. **Deploy Backend Changes**
   - Merge pattern analyzer, flow manager, repository updates
   - Deploy to staging environment
   - Run migration script

3. **Validate Migration**
   ```bash
   python scripts/validate_multi_category.py
   ```

4. **Deploy Web Interface**
   - Deploy template updates
   - Deploy CSS/JS changes
   - Test in staging

5. **Monitor & Fine-tune**
   - Monitor category distribution
   - Collect user feedback
   - Adjust priority hierarchy if needed

#### 10.4 Rollback Plan

If issues occur:
```sql
-- Rollback: Use single category field only
UPDATE content_analyses 
SET categories_detected = json_array(category)
WHERE categories_detected IS NULL;
```

#### 10.5 Files to Create
- `scripts/migrate_to_multi_category.py` - Migration script
- `scripts/validate_multi_category.py` - Validation script
- `scripts/rollback_multi_category.py` - Rollback script

**Estimated time:** 1 day

---

## 📈 Expected Outcomes

### Example: Post 1979108828265250874

**Before (Single Category):**
```
Category: disinformation
Explanation: El texto presenta información selectiva y alarmista...
```

**After (Multi-Category):**
```
Categories: anti_immigration (primary), call_to_action, disinformation
Primary: anti_immigration

Explanation: Este contenido combina tres elementos problemáticos:
1. Anti-inmigración: Presenta las clases de árabe como amenaza cultural
2. Llamada a la acción: Solicita explícitamente firmar petición
3. Desinformación: Presenta información selectiva y alarmista sin contexto

Display: [🚫 Anti-Inmigración] [📢 Llamada a Acción] [❌ Desinformación]
         ^primary badge      ^secondary badge      ^secondary badge
```

### System-wide Benefits

1. **More Accurate Analysis**
   - Captures full complexity of content
   - No forced choice between categories
   - Better reflects reality of extremist content

2. **Better Searchability**
   - Find all posts with `call_to_action`, regardless of primary category
   - Query: "Show me all anti_immigration posts that also contain disinformation"
   - Advanced filtering: "Posts with 3+ categories"

3. **Richer Insights**
   - Understand category combinations: "80% of anti_immigration posts also contain call_to_action"
   - Track evolution: "Increase in hate_speech + call_to_action combinations"
   - Pattern recognition: "VOX posts combine nationalism + anti_immigration 75% of the time"

4. **Improved UX**
   - Users see complete categorization at a glance
   - Visual hierarchy (primary vs secondary badges)
   - Tooltips explain why multiple categories apply

5. **Multi-Model Robustness**
   - Consensus across categories, not just single winner
   - Better confidence metrics
   - Identify model biases per category

---

## ⚠️ Important Considerations

### Backwards Compatibility

- Keep `category` field for legacy queries and old code
- `categories_detected` is additive, doesn't break existing functionality
- Gradual migration allows testing at each step

### Performance Impact

- JSON parsing adds ~1ms per query (negligible)
- Index on `categories_detected` if using JSON queries frequently
- Consider materialized views for complex multi-category queries

### UI Complexity

- Balance between showing all categories and visual clutter
- Primary/secondary badge distinction helps prioritize
- Tooltips and expandable sections prevent information overload

### Category Inflation Prevention

- Clear guidelines in prompts about when to use multiple categories
- Priority system ensures meaningful primary category
- Regular audits of category combinations

### Priority Hierarchy Tuning

- Initial hierarchy based on severity/importance
- May need adjustment based on:
  - User feedback
  - Use cases (research vs moderation)
  - Category distribution in real data
- Easy to update `CATEGORY_PRIORITY` dict

---

## 📊 Success Metrics

### Quantitative Metrics

1. **Category Coverage**
   - % of posts with 2+ categories: Target >30%
   - % of posts with 3+ categories: Target >10%
   - Avg categories per post: Target ~1.5-2.0

2. **Accuracy Improvements**
   - User feedback: Reduction in "incorrect category" reports
   - Multi-model agreement: Increase in consensus scores
   - Manual validation: >85% accuracy on multi-category test set

3. **System Performance**
   - Analysis time: No significant increase (<5%)
   - Database queries: <10% slowdown with JSON parsing
   - Web page load: No noticeable impact

### Qualitative Metrics

1. **User Satisfaction**
   - Feedback on more accurate categorization
   - Reduced confusion about borderline content
   - Better understanding of content complexity

2. **Research Value**
   - New insights from category combination analysis
   - Better identification of content patterns
   - More nuanced reporting

---

## 🚀 Implementation Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Category Priority System | 0.5 days | None |
| 2 | Pattern Analyzer Updates | 0.5 days | Phase 1 |
| 3 | LLM Prompt Updates | 1 day | None |
| 4 | Flow Manager Updates | 1 day | Phases 1, 2, 3 |
| 5 | Database Repository | 1 day | Phase 1 |
| 6 | Web Interface | 2 days | Phases 4, 5 |
| 7 | Multi-Model Updates | 1.5 days | Phases 4, 5 |
| 8 | Admin Interface | 1 day | Phase 6 |
| 9 | Testing & Validation | 2 days | All phases |
| 10 | Migration & Rollout | 1 day | All phases |

**Total estimated time:** ~11 days of focused development

---

## 📝 Next Steps

1. **Review & Approve Plan** - Team review of approach and priorities
2. **Create Feature Branch** - `feature/multi-category-support`
3. **Start with Phase 1** - Low-risk foundational changes
4. **Incremental Development** - Complete phases sequentially
5. **Testing at Each Phase** - Ensure quality before proceeding
6. **Staging Deployment** - Test complete system before production
7. **Production Rollout** - Phased deployment with monitoring
8. **Iterate & Improve** - Adjust based on real-world usage

---

## 📚 References

- [Pattern Analyzer Documentation](./FETCHER_SYSTEM_ARCHITECTURE.md)
- [Category Definitions](../analyzer/categories.py)
- [Database Schema](../scripts/init_database.py)
- [Multi-Model Analysis](./MULTI_MODEL_ANALYSIS.md)

---

**Document Version:** 1.0  
**Last Updated:** October 31, 2025  
**Status:** Proposed - Awaiting Implementation
