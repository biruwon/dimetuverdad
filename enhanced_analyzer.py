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
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import our enhanced components
from pattern_analyzer import PatternAnalyzer, AnalysisResult, PatternMatch
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
        
        # Pipeline Step 1: Pattern analysis (all analyzers run once)
        pattern_results = self._run_pattern_analysis(content)

        # Pipeline Step 2: Content categorization (using pattern results + LLM fallback)
        if self.verbose:
            print(f"🔍 Step 2: Categorization starting...")
        category, analysis_method = self._categorize_content(content, pattern_results)
        if self.verbose:
            print(f"🔍 Step 2: Category determined: {category}")

        # Pipeline Step 3: Smart LLM integration for uncertain cases  
        llm_explanation = self._generate_explanation_with_smart_llm(content, category, pattern_results, analysis_method)
        
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
        Pipeline Step 3: Smart LLM integration - use LLM only when patterns are ambiguous.
        
        LLM Strategy:
        - If analysis_method is "llm": ALWAYS use LLM for explanation (forced LLM usage)
        - Clear patterns detected: Use pattern-based explanation
        - Multiple conflicting patterns: Use LLM for disambiguation  
        - No clear patterns: Use LLM for analysis
        """
        # Extract pattern result from dictionary
        pattern_result = pattern_results['pattern_result']
        
        # Build base explanation from patterns
        base_explanation = self._generate_pattern_based_explanation(category, pattern_result)
        
        # CRITICAL FIX: If categorization used LLM, explanation must also use LLM
        if analysis_method == "llm":
            print("🧠 Analysis method is LLM - forcing LLM explanation generation")
            return self._primary_llm_analysis(content, category, pattern_results, base_explanation)
        
        # Check if we have clear, unambiguous results
        detected_categories = pattern_result.categories
        has_patterns = len(detected_categories) > 0
        has_multiple_patterns = len(detected_categories) > 1
        
        # Decision: When to use LLM (without any scoring)
        if not has_patterns and category == Categories.GENERAL:
            # Check content length and complexity to decide if LLM analysis is needed
            content_length = len(content.split())
            has_complex_language = any(word in content.lower() for word in [
                'características', 'correlaciones', 'interpretaciones', 'fundamentalmente',
                'transformando', 'instituciones', 'problemáticas', 'específicas',
                'tendencias', 'beneficiar', 'actores', 'internacionales'
            ])
            
            # Only skip LLM for very simple, clearly innocent content
            if content_length < 15 and not has_complex_language:
                print("✅ Clear simple content - using pattern-based analysis")
                return base_explanation
            else:
                # Complex content without patterns might need LLM analysis
                print("🧠 Complex content without patterns - using LLM for analysis")
                return self._primary_llm_analysis(content, category, pattern_results, base_explanation)
        
        elif has_patterns and not has_multiple_patterns:
            # Single clear pattern: Pattern analysis is sufficient
            print("🎯 Clear single pattern - using pattern-based analysis")
            return base_explanation
        
        elif has_multiple_patterns:
            # Multiple patterns: Use LLM for enhancement
            print("🤖 Multiple patterns detected - using LLM for enhancement")
            return self._enhance_explanation_with_llm(content, category, pattern_result, base_explanation)
        
        else:
            # Ambiguous cases: Use LLM for primary analysis
            print("🧠 Ambiguous content - using LLM for analysis")
            return self._primary_llm_analysis(content, category, pattern_results, base_explanation)
    
    def _generate_pattern_based_explanation(self, category: str, pattern_results: AnalysisResult) -> str:
        """Generate natural language explanation based purely on pattern analysis."""
        detected_categories = pattern_results.categories if pattern_results else []
        
        # Generate natural language explanations based on category
        if category == Categories.HATE_SPEECH:
            base_explanation = "Este contenido presenta características de discurso de odio, utilizando lenguaje discriminatorio y deshumanizante"
        
        elif category == Categories.DISINFORMATION:
            base_explanation = "Este contenido contiene afirmaciones que presentan características de desinformación"
            if 'health_disinformation' in detected_categories:
                base_explanation = "Este contenido presenta afirmaciones médicas sin respaldo científico verificable"
        
        elif category == Categories.CONSPIRACY_THEORY:
            base_explanation = "Este contenido promueve teorías conspiratorias sin base empírica"
            if 'conspiracy' in detected_categories:
                base_explanation += ", utilizando narrativas que fomentan desconfianza en instituciones oficiales"
        
        elif category == Categories.FAR_RIGHT_BIAS:
            base_explanation = "Este contenido muestra marcos interpretativos de extrema derecha"
            if detected_categories:
                base_explanation += " con elementos de retórica extremista"
        
        elif category == Categories.CALL_TO_ACTION:
            base_explanation = "Este contenido incluye llamadas a la acción o movilización."
        
        else:
            return "Contenido analizado sin patrones específicos detectados."
        
        # Add contextual information without redundancy
        context_parts = []
        if len(detected_categories) > 1:
            context_parts.append("presenta múltiples indicadores problemáticos")
        
        if context_parts:
            base_explanation += f" y {', '.join(context_parts)}"
        
        return base_explanation + "."
    
    def _enhance_explanation_with_llm(self, content: str, category: str, 
                                    pattern_results: AnalysisResult, base_explanation: str) -> str:
        """Use LLM to enhance and validate pattern-based analysis."""
        if not self.use_llm or not self.llm_pipeline:
            return base_explanation
        
        try:
            # Check if llm_pipeline has the required method
            if not hasattr(self.llm_pipeline, 'get_explanation'):
                print("⚠️ LLM pipeline missing get_explanation method")
                return base_explanation
            
            # Create analysis context
            analysis_context = {
                'category': category,
                'detected_categories': pattern_results.categories if pattern_results else []
            }
            
            # Get LLM enhancement using the category directly (no AnalysisType mapping)
            llm_enhancement = self.llm_pipeline.get_explanation(content, category, analysis_context)
            
            if llm_enhancement and len(llm_enhancement.strip()) > 10:
                # Combine natural language explanations
                return f"{base_explanation} {llm_enhancement}"
            else:
                return base_explanation
                
        except Exception as e:
            print(f"⚠️ Error en mejora LLM: {e}")
            return base_explanation
    
    def _primary_llm_analysis(self, content: str, category: str, 
                            pattern_results: Dict, base_explanation: str) -> str:
        """Use LLM as primary analysis method for ambiguous cases."""
        if not self.use_llm or not self.llm_pipeline:
            # Don't reveal LLM limitations to user - just return base explanation
            return base_explanation
        
        try:
            # Extract pattern result
            pattern_result = pattern_results.get('pattern_result', None)
            
            # Create analysis context with comprehensive information
            analysis_context = {
                'category': category,
                'analysis_mode': 'primary',
                'detected_categories': pattern_result.categories if pattern_result else [],
                'has_patterns': len(pattern_result.categories) > 0 if pattern_result else False
            }
            
            # Get LLM explanation using the category directly (no AnalysisType mapping)
            llm_explanation = self.llm_pipeline.get_explanation(content, category, analysis_context)
            
            if llm_explanation and len(llm_explanation.strip()) > 10:
                return llm_explanation
            else:
                # LLM failed to provide explanation - generate a comprehensive pattern-based one
                print("⚠️ LLM explanation was empty, using enhanced pattern analysis")
                return self._generate_enhanced_pattern_explanation(content, category, pattern_results)
                
        except Exception as e:
            print(f"❌ Error en análisis primario LLM: {e}")
            # Don't reveal LLM errors to user - return enhanced pattern explanation
            return self._generate_enhanced_pattern_explanation(content, category, pattern_results)
    
    def _generate_enhanced_pattern_explanation(self, content: str, category: str, pattern_results: Dict) -> str:
        """Generate a comprehensive explanation when LLM is not available or fails."""
        pattern_result = pattern_results.get('pattern_result', None)
        detected_categories = pattern_result.categories if pattern_result else []
        
        # Generate detailed explanations based on what was actually detected
        if category == Categories.FAR_RIGHT_BIAS:
            explanation_parts = []
            
            # Analyze the specific type of bias detected
            content_lower = content.lower()
            
            if any('sustitución' in content_lower for word in ['sustitución', 'sustitu']):
                explanation_parts.append("presenta narrativas sobre sustitución poblacional")
            
            if any(word in content_lower for word in ['inmigr', 'extranjero', 'moro']):
                explanation_parts.append("contiene referencias a inmigración con posible sesgo")
            
            if any(word in content_lower for word in ['muslim', 'islám', 'tradiciones culturales']):
                explanation_parts.append("incluye generalizaciones sobre grupos culturales o religiosos")
            
            if any(word in content_lower for word in ['efecto llamada', 'política', 'brussels', 'comisión europea']):
                explanation_parts.append("enmarca políticas oficiales desde perspectivas potencialmente sesgadas")
            
            if explanation_parts:
                base = "Este contenido muestra características de sesgo político de extrema derecha ya que "
                return base + ", ".join(explanation_parts) + ". El análisis detecta marcos interpretativos que pueden reforzar narrativas problemáticas sobre inmigración y diversidad cultural."
            else:
                return "Este contenido presenta marcos interpretativos característicos del sesgo de extrema derecha, utilizando lenguaje que puede reforzar narrativas discriminatorias."
        
        elif category == Categories.CONSPIRACY_THEORY:
            return "Este contenido promueve teorías conspiratorias presentando afirmaciones sin base empírica verificable y fomentando desconfianza en instituciones oficiales."
        
        elif category == Categories.HATE_SPEECH:
            return "Este contenido presenta características de discurso de odio, utilizando lenguaje discriminatorio que puede fomentar hostilidad hacia grupos específicos."
        
        elif category == Categories.DISINFORMATION:
            return "Este contenido presenta características de desinformación, incluyendo afirmaciones que requieren verificación y que pueden difundir información inexacta."
        
        elif category == Categories.CALL_TO_ACTION:
            return "Este contenido incluye llamadas explícitas a la acción o movilización que pueden promover activismo de extrema derecha."
        
        else:
            return "Contenido categorizado mediante análisis de patrones sin características específicas de extremismo detectadas."
    
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
    print("💡 Usa comprehensive_test_suite.py para ejecutar tests")
    print("💡 O importa EnhancedAnalyzer para usar la funcionalidad de análisis")
