"""
Analyzer: far-right activism detection with comprehensive coverage.
Integrates specialized components for content analysis workflow.
"""

import json
import sqlite3
import time
import warnings
import sys
import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our components
from .pattern_analyzer import PatternAnalyzer
from utils.text_utils import normalize_text
from .llm_models import EnhancedLLMPipeline
from .categories import Categories

# Import new modular components
from .config import AnalyzerConfig
from .metrics import MetricsCollector
from .repository import ContentAnalysisRepository
from .text_analyzer import TextAnalyzer
from .multimodal_analyzer import MultimodalAnalyzer
from .models import ContentAnalysis
from .constants import AnalysisMethods, ErrorMessages

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging for multimodal analysis debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DB path (same as other scripts)
from utils.paths import get_db_path
DB_PATH = get_db_path()


class Analyzer:
    """
    Refactored analyzer with improved modularity and separation of concerns.

    Uses specialized components for different analysis types:
    - TextAnalyzer: Text-only content analysis
    - MultimodalAnalyzer: Media content analysis
    - MetricsCollector: Performance tracking
    - ContentAnalysisRepository: Database operations
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None, verbose: bool = False):
        """
        Initialize analyzer with configuration.

        Args:
            config: Analyzer configuration (created with defaults if None)
            verbose: Whether to enable verbose logging
        """
        self.config = config or AnalyzerConfig()
        self.verbose = verbose

        # Initialize components
        self.metrics = MetricsCollector()
        self.repository = ContentAnalysisRepository(DB_PATH)

        # Initialize analysis components
        self.text_analyzer = TextAnalyzer(config=self.config, verbose=verbose)
        self.multimodal_analyzer = MultimodalAnalyzer(verbose=verbose)

        if self.verbose:
            print("🚀 Iniciando Analyzer con componentes modulares...")
            print("Componentes cargados:")
            print("- ✓ Analizador de texto especializado")
            print("- ✓ Analizador multimodal para medios")
            print("- ✓ Recolector de métricas de rendimiento")
            print("- ✓ Repositorio de análisis de contenido")
            print(f"- ✓ Configuración: {self.config.model_priority} priority")

    def analyze_content(self,
                       tweet_id: str,
                       tweet_url: str,
                       username: str,
                       content: str,
                       media_urls: List[str] = None) -> ContentAnalysis:
        """
        Main content analysis pipeline with conditional multimodal analysis.

        Routes to appropriate analyzer based on content type.
        """
        analysis_start_time = time.time()

        try:
            # Route to appropriate analyzer
            if media_urls and len(media_urls) > 0:
                result = self.multimodal_analyzer.analyze_with_media(
                    tweet_id, tweet_url, username, content, media_urls
                )
            else:
                result = self.text_analyzer.analyze(
                    tweet_id, tweet_url, username, content
                )

            # Track metrics
            analysis_time = time.time() - analysis_start_time
            self.metrics.record_analysis(
                method=result.analysis_method,
                duration=analysis_time,
                category=result.category,
                model_used=self._get_model_name(result),
                is_multimodal=result.analysis_method == AnalysisMethods.MULTIMODAL.value
            )

            # Add performance data to result
            result.analysis_time_seconds = analysis_time
            result.model_used = self._get_model_name(result)

            return result

        except Exception as e:
            # Handle analysis errors gracefully
            if self.verbose:
                print(f"❌ Error en análisis: {e}")
                import traceback
                traceback.print_exc()

            # Return error result
            return ContentAnalysis(
                tweet_id=tweet_id,
                tweet_url=tweet_url,
                username=username,
                tweet_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category=Categories.GENERAL,
                categories_detected=[Categories.GENERAL],
                llm_explanation=ErrorMessages.ANALYSIS_FAILED.format(error=str(e)),
                analysis_method=AnalysisMethods.ERROR.value,
                pattern_matches=[],
                topic_classification={},
                analysis_json=json.dumps({'error': str(e)}, ensure_ascii=False)
            )

    def _get_model_name(self, result: ContentAnalysis) -> str:
        """Get the model name based on analysis result."""
        if result.analysis_method == AnalysisMethods.MULTIMODAL.value:
            return "gemini-2.5-flash"
        elif result.analysis_method == AnalysisMethods.LLM.value:
            return f"ollama-{self.config.model_priority}"
        else:
            return "pattern-matching"

    def get_metrics_report(self) -> str:
        """Generate comprehensive metrics report."""
        return self.metrics.generate_report()

    def save_analysis(self, analysis: ContentAnalysis) -> bool:
        """
        Save analysis result to database.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.repository.save(analysis)
            return True
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving analysis: {e}")
            return False

    def get_analysis(self, tweet_id: str) -> Optional[ContentAnalysis]:
        """
        Retrieve analysis result from database.

        Returns:
            ContentAnalysis if found, None otherwise
        """
        try:
            return self.repository.get_by_tweet_id(tweet_id)
        except Exception as e:
            if self.verbose:
                print(f"❌ Error retrieving analysis: {e}")
            return None

    def cleanup_resources(self):
        """Clean up any resources used by the analyzer."""
        try:
            if hasattr(self.text_analyzer, 'llm_pipeline') and self.text_analyzer.llm_pipeline:
                if hasattr(self.text_analyzer.llm_pipeline, 'cleanup'):
                    self.text_analyzer.llm_pipeline.cleanup()
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")

    def print_system_status(self):
        """Print current system status for debugging."""
        print("🔧 ANALYZER SYSTEM STATUS")
        print("=" * 50)
        print(f"🤖 LLM Enabled: {self.config.use_llm}")
        print(f"🔧 Model Priority: {self.config.model_priority}")
        print(f"🧠 LLM Pipeline: {'✓' if self.text_analyzer.llm_pipeline else '❌'}")
        print(f"🔍 Pattern analyzer: {'✓' if self.text_analyzer.pattern_analyzer else '❌'}")
        print(f"🎥 Multimodal analyzer: {'✓' if self.multimodal_analyzer.gemini_analyzer else '❌'}")

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





if __name__ == '__main__':
    print("❌ Este módulo no debe ejecutarse directamente")
    print("💡 Usa scripts/test_suite.py para ejecutar tests")
    print("💡 O importa Analyzer para usar la funcionalidad de análisis")


# Utility functions for analyzer operations
def create_analyzer(config: Optional[AnalyzerConfig] = None, verbose: bool = False) -> Analyzer:
    """Create and return an Analyzer instance with specified configuration."""
    return Analyzer(config=config, verbose=verbose)


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
    analyzer.save_analysis(analysis_result)

    return analysis_result
