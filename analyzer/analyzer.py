"""
Analyzer: far-right activism detection with comprehensive coverage.
Integrates specialized components for content analysis workflow.
"""

import json
import os
import re
import sqlite3
import time
import warnings
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our components
from .pattern_analyzer import PatternAnalyzer, AnalysisResult, PatternMatch
from utils.text_utils import normalize_text
from .llm_models import EnhancedLLMPipeline
from .categories import Categories

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# DB path (same as other scripts)
from utils.paths import get_db_path
DB_PATH = get_db_path()
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
    analysis_method: str = "pattern"  # "pattern", "llm", or "gemini"
    
    # Media analysis fields
    media_urls: List[str] = None  # List of media URLs
    media_analysis: str = ""      # Gemini multimodal analysis result
    media_type: str = ""          # "image", "video", or ""
    multimodal_analysis: bool = False  # Whether media was analyzed
    
    # Technical data
    pattern_matches: List[Dict] = None
    topic_classification: Dict = None
    analysis_json: str = ""
    
    # Performance metrics
    analysis_time_seconds: float = 0.0  # Total analysis time
    model_used: str = ""                # Which model was used (gpt-oss:20b, gemini-2.5-flash, etc.)
    tokens_used: int = 0                # Approximate tokens used (if available)
    
    def __post_init__(self):
        # Initialize lists to avoid None values
        if self.categories_detected is None:
            self.categories_detected = []
        if self.pattern_matches is None:
            self.pattern_matches = []
        if self.media_urls is None:
            self.media_urls = []
    
    @property
    def has_multiple_categories(self) -> bool:
        """Check if content was classified with multiple categories."""
        return len(self.categories_detected) > 1
    
    def get_secondary_categories(self) -> List[str]:
        """Get all categories except the primary one."""
        return [cat for cat in self.categories_detected if cat != self.category]



class Analyzer:
    """
    Analyzer with improved LLM integration for content analysis workflows.
    """
    
    def __init__(self, use_llm: bool = True, model_priority: str = "balanced", verbose: bool = False):
        self.pattern_analyzer = PatternAnalyzer()
        self.use_llm = use_llm
        self.model_priority = model_priority
        self.verbose = verbose  # Control debug output
        self.llm_pipeline = None
        
        # Metrics tracking (always enabled)
        self.metrics = {
            'total_analyses': 0,
            'method_counts': {'pattern': 0, 'llm': 0},
            'multimodal_count': 0,
            'category_counts': {},
            'total_time': 0.0,
            'avg_time_per_analysis': 0.0,
            'model_usage': {},
            'start_time': time.time()
        }
        
        if self.verbose:
            print("🚀 Iniciando Analyzer...")
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
                             content: str,
                             media_urls: List[str] = None) -> ContentAnalysis:
        """
        Main content analysis pipeline with conditional multimodal analysis.
        
        Routes to Gemini multimodal analysis if media is present, otherwise uses
        the traditional pattern + LLM pipeline.
        """
        analysis_start_time = time.time()
        
        # Conditional routing: Use Gemini for media, traditional pipeline for text-only
        if media_urls and len(media_urls) > 0:
            result = self.analyze_multi_modal(tweet_id, tweet_url, username, content, media_urls)
        else:
            result = self._analyze_text_only(tweet_id, tweet_url, username, content)
        
        # Track metrics (always enabled)
        analysis_time = time.time() - analysis_start_time
        self._update_metrics(result, analysis_time)
        
        # Add metrics to result
        result.analysis_time_seconds = analysis_time
        result.model_used = self._get_model_name(result.analysis_method, result.multimodal_analysis)
        
        return result
    
    def _update_metrics(self, result: ContentAnalysis, analysis_time: float):
        """Update internal metrics with analysis results."""
        self.metrics['total_analyses'] += 1
        self.metrics['method_counts'][result.analysis_method] += 1
        
        # Track multimodal usage separately
        if result.multimodal_analysis:
            self.metrics['multimodal_count'] += 1
        
        self.metrics['total_time'] += analysis_time
        
        # Update category counts
        if result.category not in self.metrics['category_counts']:
            self.metrics['category_counts'][result.category] = 0
        self.metrics['category_counts'][result.category] += 1
        
        # Update model usage
        model_name = self._get_model_name(result.analysis_method, result.multimodal_analysis)
        if model_name not in self.metrics['model_usage']:
            self.metrics['model_usage'][model_name] = 0
        self.metrics['model_usage'][model_name] += 1
        
        # Update average time
        self.metrics['avg_time_per_analysis'] = self.metrics['total_time'] / self.metrics['total_analyses']
    
    def _get_model_name(self, analysis_method: str, is_multimodal: bool = False) -> str:
        """Get the model name based on analysis method and multimodal flag."""
        if is_multimodal:
            return "gemini-2.5-flash"
        elif analysis_method == "llm":
            return f"ollama-{self.model_priority}"
        else:
            return "pattern-matching"
    
    def get_metrics_report(self) -> str:
        """Generate a comprehensive metrics report."""
        total_time = time.time() - self.metrics['start_time']
        
        report = []
        report.append("📊 ANALYSIS METRICS REPORT")
        report.append("=" * 50)
        report.append(f"⏱️  Total runtime: {total_time:.2f}s")
        report.append(f"📈 Total analyses: {self.metrics['total_analyses']}")
        report.append(f"⚡ Average time per analysis: {self.metrics['avg_time_per_analysis']:.2f}s")
        report.append(f"💰 Total analysis time: {self.metrics['total_time']:.2f}s")
        report.append("")
        
        # Method breakdown
        report.append("🔧 Analysis Methods Used:")
        for method, count in self.metrics['method_counts'].items():
            if count > 0:
                percentage = (count / self.metrics['total_analyses']) * 100
                report.append(f"  {method.upper()}: {count} ({percentage:.1f}%)")
        
        # Multimodal breakdown
        if self.metrics['multimodal_count'] > 0:
            multimodal_percentage = (self.metrics['multimodal_count'] / self.metrics['total_analyses']) * 100
            report.append(f"  🎥 Multimodal: {self.metrics['multimodal_count']} ({multimodal_percentage:.1f}%)")
        report.append("")
        
        # Model usage
        report.append("🤖 Models Used:")
        for model, count in self.metrics['model_usage'].items():
            percentage = (count / self.metrics['total_analyses']) * 100
            report.append(f"  {model}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Category breakdown
        report.append("🏷️  Categories Detected:")
        sorted_categories = sorted(self.metrics['category_counts'].items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            percentage = (count / self.metrics['total_analyses']) * 100
            report.append(f"  {category}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Performance insights
        if self.metrics['total_analyses'] > 0:
            report.append("⚡ Performance Insights:")
            if self.metrics['multimodal_count'] > 0:
                avg_multimodal_time = self.metrics['total_time'] / self.metrics['multimodal_count']
                report.append(f"  🎥 Average Multimodal analysis time: {avg_multimodal_time:.2f}s")
            if self.metrics['method_counts']['llm'] > 0:
                avg_llm_time = self.metrics['total_time'] / self.metrics['method_counts']['llm']
                report.append(f"  🧠 Average LLM analysis time: {avg_llm_time:.2f}s")
            if self.metrics['method_counts']['pattern'] > 0:
                avg_pattern_time = self.metrics['total_time'] / self.metrics['method_counts']['pattern']
                report.append(f"  🔍 Average Pattern analysis time: {avg_pattern_time:.2f}s")
        
        return "\n".join(report)
    def _analyze_text_only(self, 
                             tweet_id: str,
                             tweet_url: str, 
                             username: str,
                             content: str) -> ContentAnalysis:
        
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
        llm_explanation = self._generate_llm_explanation(content_normalized, category, pattern_results)

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
    
    def analyze_multi_modal(self,
                           tweet_id: str,
                           tweet_url: str,
                           username: str,
                           content: str,
                           media_urls: List[str]) -> ContentAnalysis:
        """
        Analyze multimodal content (text + media) using Gemini 2.5 Flash.
        
        This method handles tweets with images/videos by using Gemini's multimodal
        capabilities instead of the traditional pattern + LLM pipeline.
        """
        if self.verbose:
            print(f"\n🎥 Multimodal analysis: @{username}")
            print(f"📝 Content: {content[:80]}...")
            print(f"🖼️ Media URLs: {len(media_urls)} found")
        
        # Import the multimodal analyzer
        from .gemini_multimodal import analyze_multimodal_content, extract_media_type
        
        # Perform multimodal analysis
        media_analysis, analysis_time = analyze_multimodal_content(media_urls, content)
        
        if media_analysis:
            # Extract category from Gemini analysis (simplified approach)
            # For now, we'll use a basic heuristic - in production, you might want to parse the Gemini response
            category = self._extract_category_from_gemini(media_analysis)
            media_type = extract_media_type(media_urls)
            
            if self.verbose:
                print(f"✅ Gemini analysis completed in {analysis_time:.2f}s")
                print(f"🏷️ Category: {category}")
                print(f"📊 Media type: {media_type}")
            
            return ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category=category,
                categories_detected=[category],  # Single category for now
                llm_explanation="",  # Gemini analysis is in media_analysis field
                analysis_method="llm",  # Use "llm" method for multimodal analysis
                media_urls=media_urls,
                media_analysis=media_analysis,
                media_type=media_type,
                multimodal_analysis=True,
                pattern_matches=[],  # No pattern analysis for multimodal
                topic_classification={},
                analysis_json=json.dumps({
                    'multimodal_analysis': True,
                    'media_type': media_type,
                    'analysis_time': analysis_time
                }, ensure_ascii=False)
            )
        else:
            # Fallback to text-only analysis if multimodal fails
            if self.verbose:
                print("❌ Multimodal analysis failed, falling back to text-only")
            return self._analyze_text_only(tweet_id, tweet_url, username, content)
    
    def _extract_category_from_gemini(self, gemini_analysis: str) -> str:
        """
        Extract category from Gemini multimodal analysis.
        
        This is a simplified implementation. In production, you might want to:
        - Parse the structured Gemini response
        - Use more sophisticated category extraction
        - Map Gemini categories to your existing category system
        """
        analysis_lower = gemini_analysis.lower()
        
        # Map Gemini analysis to our categories (check for Spanish keywords)
        if any(keyword in analysis_lower for keyword in ['hate_speech', 'odio', 'racismo', 'discriminación']):
            return Categories.HATE_SPEECH
        elif any(keyword in analysis_lower for keyword in ['disinformation', 'desinformación', 'fake news', 'mentira']):
            return Categories.DISINFORMATION
        elif any(keyword in analysis_lower for keyword in ['conspiracy', 'conspiración', 'teoría conspirativa']):
            return Categories.CONSPIRACY_THEORY
        elif any(keyword in analysis_lower for keyword in ['far_right', 'extrema derecha', 'ultraderecha']):
            return Categories.FAR_RIGHT_BIAS
        elif any(keyword in analysis_lower for keyword in ['call_to_action', 'llamado a la acción', 'llamados a la acción']):
            return Categories.CALL_TO_ACTION
        else:
            return Categories.GENERAL  # Default fallback
    
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
        print("🔧 ANALYZER SYSTEM STATUS")
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
             categories_detected, media_urls, media_analysis, media_type, multimodal_analysis) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.tweet_id, analysis.tweet_url, analysis.username, analysis.tweet_content,
                analysis.category, analysis.llm_explanation, analysis.analysis_method,
                analysis.analysis_json, analysis.analysis_timestamp,
                json.dumps(analysis.categories_detected, ensure_ascii=False),
                json.dumps(analysis.media_urls, ensure_ascii=False),
                analysis.media_analysis, analysis.media_type, analysis.multimodal_analysis
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
    print("💡 O importa Analyzer para usar la funcionalidad de análisis")


# Utility functions for analyzer operations
def create_analyzer(use_llm: bool = True, model_priority: str = "balanced", verbose: bool = False) -> Analyzer:
    """Create and return an Analyzer instance with specified parameters."""
    return Analyzer(use_llm=use_llm, model_priority=model_priority, verbose=verbose)


def reanalyze_tweet(tweet_id: str, analyzer: Optional[Analyzer] = None) -> Optional[ContentAnalysis]:
    """Reanalyze a single tweet and return the result."""
    from utils.database import get_tweet_data, delete_existing_analysis

    # Get tweet data
    tweet_data = get_tweet_data(tweet_id)
    if not tweet_data:
        return None

    # Delete existing analysis
    delete_existing_analysis(tweet_id)

    # Use provided analyzer or create default one
    if analyzer is None:
        analyzer = create_analyzer()

    # Reanalyze
    analysis_result = analyzer.analyze_content(
        tweet_id=tweet_data['tweet_id'],
        tweet_url=f"https://twitter.com/placeholder/status/{tweet_data['tweet_id']}",
        username=tweet_data['username'],
        content=tweet_data['content'],
        media_urls=tweet_data.get('media_urls', [])
    )

    # Save result
    save_content_analysis(analysis_result)

    return analysis_result
