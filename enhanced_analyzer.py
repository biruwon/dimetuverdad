"""
Enhanced analyzer: far-right activism detection with comprehensive coverage.
Integrates specialized components for content analysis workflow.
"""

import json
import os
import re
import sqlite3
import time
import warnings
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import our enhanced components
from pattern_analyzer import PatternAnalyzer, AnalysisResult, PatternMatch
from utils.text_utils import normalize_text
from llm_models import EnhancedLLMPipeline
from categories import Categories

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# DB path (same as other scripts)
DB_PATH = os.path.join(os.path.dirname(__file__), 'accounts.db')
@dataclass
class ContentAnalysis:
    """Content analysis result structure with multi-category support."""
    # Tweet metadata
    tweet_id: str
    tweet_url: str
    username: str
    tweet_content: str
    analysis_timestamp: str
    
    # Content categories (consolidated and multi-category support)
    category: str  # Primary category (backward compatibility)
    categories_detected: List[str] = None  # All detected categories
    
    # Analysis results
    llm_explanation: str = ""
    analysis_method: str = "pattern"  # "pattern" or "llm"
    
    # Technical data
    pattern_matches: List[Dict] = None
    topic_classification: Dict = None
    analysis_json: str = ""
    
    def __post_init__(self):
        # Initialize lists to avoid None values
        if self.categories_detected is None:
            self.categories_detected = []
        if self.pattern_matches is None:
            self.pattern_matches = []
    
    @property
    def has_multiple_categories(self) -> bool:
        """Check if content was classified with multiple categories."""
        return len(self.categories_detected) > 1
    
    def get_secondary_categories(self) -> List[str]:
        """Get all categories except the primary one."""
        return [cat for cat in self.categories_detected if cat != self.category]



class EnhancedAnalyzer:
    """
    Enhanced analyzer with improved LLM integration for content analysis workflows.
    """
    
    def __init__(self, use_llm: bool = True, model_priority: str = "balanced", verbose: bool = False):
        self.pattern_analyzer = PatternAnalyzer()
        self.use_llm = use_llm
        self.model_priority = model_priority
        self.verbose = verbose  # Control debug output
        self.llm_pipeline = None
        
        if self.verbose:
            print("🚀 Iniciando Enhanced Analyzer...")
            print("Componentes cargados:")
            print("- ✓ Analizador unificado de patrones (extrema derecha + temas políticos)")
            print("- ✓ Detector de afirmaciones verificables")
            print("- ✓ Sistema de recuperación de evidencia")
            print("- ✓ Modo de análisis de contenido activado")
        
        # Always try to load LLM pipeline as it's needed for fallback when no patterns are detected
        # The use_llm flag only controls whether we use it for enhancing pattern-based results
        if self.verbose:
            print("- ⏳ Cargando modelo LLM para análisis de contenido sin patrones...")
        try:
            # Use recommended model (now defaults to original gpt-oss-20b for best performance)
            self.llm_pipeline = EnhancedLLMPipeline(model_priority=model_priority)
            if self.verbose:
                print("- ✓ Modelo LLM cargado correctamente")
        except Exception as e:
            if self.verbose:
                print(f"- ⚠️ Error cargando LLM: {e}")
                print("- 🔄 Intentando con modelo de respaldo...")
            try:
                # Fallback to flan-t5-small if Ollama is not available
                self.llm_pipeline = EnhancedLLMPipeline(model_priority=model_priority)
                print("- ✓ Modelo de respaldo cargado correctamente")
            except Exception as e2:
                print(f"- ❌ Error cargando modelo de respaldo: {e2}")
                self.llm_pipeline = None
                print("- ⚠️ Sistema funcionará solo con análisis de patrones")
        
        if use_llm:
            print("- ✓ Modo LLM habilitado para mejora de explicaciones")
        else:
            print("- ℹ️ Modo LLM deshabilitado para explicaciones (solo usado como fallback sin patrones)")
    
    def analyze_content(self, 
                             tweet_id: str,
                             tweet_url: str, 
                             username: str,
                             content: str) -> ContentAnalysis:
        """
        Main content analysis pipeline with consolidated pattern analysis.
        """
        if not content or len(content.strip()) < 5:
            return ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category=Categories.GENERAL,
                llm_explanation="Content too short for analysis"
            )
        
        if self.verbose:
            print(f"\n🔍 Content analysis: @{username}")
            print(f"📝 Contenido: {content[:80]}...")

        # Normalize content for pattern matching and LLM prompts while
        # preserving the original content for storage/display.
        content_normalized = normalize_text(content)

        # Pipeline Step 1: Pattern analysis (all analyzers run once) using normalized text
        pattern_results = self._run_pattern_analysis(content_normalized)

        # Pipeline Step 2: Content categorization (using pattern results + LLM fallback)
        if self.verbose:
            print(f"🔍 Step 2: Categorization starting...")
        # Use normalized content for categorization and LLM explanation
        category, analysis_method = self._categorize_content(content_normalized, pattern_results)
        if self.verbose:
            print(f"🔍 Step 2: Category determined: {category}")

        # Pipeline Step 3: Smart LLM integration for uncertain cases
        llm_explanation = self._generate_explanation_with_smart_llm(content_normalized, category, pattern_results, analysis_method)

        # Pipeline Step 4: Create final analysis structure with multi-category support
        analysis_data = self._build_analysis_data(pattern_results)

        # Extract multi-category information from pattern results
        pattern_result = pattern_results.get('pattern_result', None)
        if pattern_result:
            categories_detected = pattern_result.categories
        else:
            # Fallback for LLM-only analysis
            categories_detected = [category]

        return ContentAnalysis(
            tweet_id=tweet_id,
            tweet_url=tweet_url,
            username=username,
            tweet_content=content,
            analysis_timestamp=datetime.now().isoformat(),
            category=category,
            categories_detected=categories_detected,
            llm_explanation=llm_explanation,
            analysis_method=analysis_method,
            pattern_matches=[{'matched_text': pm.matched_text, 'category': pm.category, 'description': pm.description} for pm in (pattern_result.pattern_matches if pattern_result else [])],
            topic_classification=analysis_data['topic_classification'],
            analysis_json=json.dumps(analysis_data, ensure_ascii=False, default=str)
        )
    
    def _run_pattern_analysis(self, content: str) -> Dict:
        """
        Pipeline Step 1: Pattern analysis using the consolidated pattern analyzer.
        
        Simplified Approach:
        - PatternAnalyzer: Combines all pattern detection including disinformation claims in one pass
        """
        # Single phase: Comprehensive pattern analysis
        pattern_result = self.pattern_analyzer.analyze_content(content)
        
        return {
            'pattern_result': pattern_result
        }
    
    def _categorize_content(self, content: str, pattern_results: Dict) -> Tuple[str, str]:
        """
        Pipeline Step 2: Determine content category using pattern results + LLM fallback.
        Returns: (category, analysis_method)
        
        Simplified approach:
        1. If patterns detected any category -> return first detected category (pattern)
        2. If no patterns detected -> use LLM fallback (llm)
        3. If LLM returns general -> general (llm)
        """
        pattern_result = pattern_results['pattern_result']
        detected_categories = pattern_result.categories
        
        if self.verbose:
            print(f"🔍 Detected categories: {detected_categories}")
        
        # Step 1: If patterns found any category, return the primary one
        if detected_categories:
            primary_category = detected_categories[0]  # Use first detected category
            if self.verbose:
                print(f"🎯 Pattern detected: {primary_category}")
            return primary_category, "pattern"
        
        # Step 2: No patterns detected - use LLM fallback
        if self.verbose:
            print("🧠 No patterns detected - using LLM for analysis")
        llm_category = self._get_llm_category(content, pattern_results)
        if self.verbose:
            print(f"🔍 LLM category result: {llm_category}")
        return llm_category, "llm"

    def _get_llm_category(self, content: str, pattern_results: Dict) -> str:
        """Use LLM to categorize content when patterns are insufficient."""
        # When no patterns are detected, we MUST use LLM to avoid defaulting to 'general'
        # This overrides the use_llm flag because we need categorization
        if not self.llm_pipeline:
            if self.verbose:
                print("🔍 LLM pipeline not available, returning general")
            return Categories.GENERAL

        try:
            if self.verbose:
                print(f"🔍 _get_llm_category called with content: {content[:50]}...")
                print("🔍 Calling llm_pipeline.get_category...")            # Use FAST category detection instead of full analysis
            llm_category = self.llm_pipeline.get_category(content)
            
            # No hardcoded fallback patterns - let the LLM handle all edge cases
            # This makes the system truly scalable without keyword maintenance
            return llm_category
            
        except Exception as e:
            print(f"⚠️ Error en categorización LLM: {e}")
            return Categories.GENERAL
    
    def _generate_explanation_with_smart_llm(self, content: str, category: str, 
                                           pattern_results: Dict, analysis_method: str) -> str:
        """
        Pipeline Step 3: Always use LLM for explanation generation.
        
        Simplified Strategy:
        - Patterns are only used for faster category detection
        - LLM is always used for explanation generation regardless of analysis_method
        - This ensures consistent, high-quality explanations across all content
        """
        # Always use LLM for explanations - patterns are only for category detection
        print("🧠 Using LLM for explanation generation (patterns used only for category detection)")
        return self._generate_llm_explanation(content, category, pattern_results)
    
    def _generate_llm_explanation(self, content: str, category: str, pattern_results: Dict) -> str:
        """Generate explanation using LLM - surface actual errors instead of hiding them."""
        if not self.llm_pipeline:
            # Surface the actual issue instead of hiding it
            return "ERROR: LLM pipeline not available - explanation generation impossible"
        
        try:
            # Extract pattern result for context
            pattern_result = pattern_results.get('pattern_result', None)
            
            # Create analysis context with comprehensive information
            analysis_context = {
                'category': category,
                'analysis_mode': 'explanation',
                'detected_categories': pattern_result.categories if pattern_result else [],
                'has_patterns': len(pattern_result.categories) > 0 if pattern_result else False
            }
            
            if self.verbose:
                print(f"🔍 Calling get_explanation with context: {analysis_context}")
            
            # Get LLM explanation using the category directly
            llm_explanation = self.llm_pipeline.get_explanation(content, category, analysis_context)
            
            if llm_explanation and len(llm_explanation.strip()) > 10:
                return llm_explanation
            else:
                # Surface the actual issue: LLM returned empty/insufficient response
                empty_response_info = f"LLM returned: '{llm_explanation}'" if llm_explanation else "LLM returned None/empty"
                return f"ERROR: LLM explanation generation failed - {empty_response_info} (length: {len(llm_explanation.strip()) if llm_explanation else 0})"
                
        except Exception as e:
            # Surface the actual error with full traceback information
            if self.verbose:
                print(f"❌ Error en explicación LLM: {e}")
                import traceback
                traceback.print_exc()
            # Return the actual error instead of hiding it
            return f"ERROR: LLM explanation generation exception - {type(e).__name__}: {str(e)}"
    
    def _build_analysis_data(self, pattern_results: Dict) -> Dict:
        """
        Pipeline Step 4: Build the final analysis data structure.
        """
        pattern_result = pattern_results.get('pattern_result', None)
        
        if pattern_result:
            pattern_matches = pattern_result.pattern_matches
            categories = pattern_result.categories
            political_context = pattern_result.political_context
        else:
            pattern_matches = []
            categories = []
            political_context = []
        
        return {
            'category': None,  # Will be set by caller
            'pattern_matches': [{'matched_text': pm.matched_text, 'category': pm.category, 'description': pm.description} for pm in pattern_matches],
            'topic_classification': {
                'primary_topic': categories[0] if categories else Categories.GENERAL,
                'all_topics': [{'category': cat} for cat in categories],
                'political_context': political_context
            },
            'unified_categories': categories
        }
    

    

    def cleanup_resources(self):
        """Clean up any resources used by the analyzer."""
        try:
            if hasattr(self, 'llm_pipeline') and self.llm_pipeline:
                if hasattr(self.llm_pipeline, 'cleanup'):
                    self.llm_pipeline.cleanup()
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")
    
    def print_system_status(self):
        """Print current system status for debugging."""
        print("🔧 ENHANCED ANALYZER SYSTEM STATUS")
        print("=" * 50)
        print(f"🤖 LLM Enabled: {self.use_llm}")
        if self.llm_pipeline:
            print(f"🔧 Model Priority: {self.model_priority}")
            print(f"🧠 LLM Pipeline: Loaded")
        else:
            print("⚠️ LLM Pipeline: Not available")
        
        print(f"🔍 Pattern analyzer: {'✓' if self.pattern_analyzer else '❌'}")
        
        # Check database
        try:
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM content_analyses")
            count = c.fetchone()[0]
            conn.close()
            print(f"📊 Database: {count} analyses stored")
        except Exception as e:
            print(f"❌ Database: Error - {e}")


# Database functions for content analysis workflow
def migrate_database_schema():
    """Migrate existing database to add missing columns and handle unified analysis."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    
    # Check if content_analyses table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='content_analyses'")
    if not c.fetchone():
        print("📋 Creating content_analyses table...")
        init_content_analysis_table()
        conn.close()
        return
    
    # Get current columns
    c.execute("PRAGMA table_info(content_analyses)")
    columns = [col[1] for col in c.fetchall()]
    
    # Add missing columns for unified analysis and multi-category support
    missing_columns = []
    expected_columns = {
        'analysis_method': 'TEXT DEFAULT "pattern"',  # "pattern" or "llm"
        'evidence_sources': 'TEXT',
        'verification_status': 'TEXT DEFAULT "pending"',
        'misinformation_risk': 'TEXT',
        'categories_detected': 'TEXT'  # JSON array of all detected categories
    }
    
    for col_name, col_type in expected_columns.items():
        if col_name not in columns:
            missing_columns.append((col_name, col_type))
    
    if missing_columns:
        print(f"🔧 Adding {len(missing_columns)} missing columns to database...")
        for col_name, col_type in missing_columns:
            try:
                c.execute(f"ALTER TABLE content_analyses ADD COLUMN {col_name} {col_type}")
                print(f"  ✓ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    print(f"  ⚠️ Error adding {col_name}: {e}")
    else:
        print("✅ Database schema is up to date")
    
    conn.commit()
    conn.close()

def init_content_analysis_table():
    """Initialize content analyses table with proper schema including multi-category support."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS content_analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_id TEXT UNIQUE,
        tweet_url TEXT,
        username TEXT,
        tweet_content TEXT,
        category TEXT,                -- Primary category (backward compatibility)
        subcategory TEXT,
        llm_explanation TEXT,
        calls_to_action BOOLEAN,
        analysis_json TEXT,
        analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        evidence_sources TEXT,
        verification_status TEXT DEFAULT "pending",
        misinformation_risk TEXT,
        analysis_method TEXT DEFAULT "pattern",
        categories_detected TEXT     -- JSON array of all detected categories  
    )
    ''')
    conn.commit()
    conn.close()

def save_content_analysis(analysis: ContentAnalysis):
    """Save content analysis to database with retry logic."""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            c = conn.cursor()
            
            c.execute('''
            INSERT OR REPLACE INTO content_analyses 
            (tweet_id, tweet_url, username, tweet_content, category, 
             llm_explanation, analysis_method, analysis_json, analysis_timestamp,
             categories_detected) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.tweet_id, analysis.tweet_url, analysis.username, analysis.tweet_content,
                analysis.category, analysis.llm_explanation, analysis.analysis_method,
                analysis.analysis_json, analysis.analysis_timestamp,
                json.dumps(analysis.categories_detected, ensure_ascii=False)
            ))
            
            conn.commit()
            conn.close()
            return  # Success
            
        except sqlite3.OperationalError as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Database locked, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"❌ Database remains locked after {max_retries} attempts: {e}")
                raise


if __name__ == '__main__':
    print("❌ Este módulo no debe ejecutarse directamente")
    print("💡 Usa scripts/test_suite.py para ejecutar tests")
    print("💡 O importa EnhancedAnalyzer para usar la funcionalidad de análisis")
