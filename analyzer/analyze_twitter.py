"""
Analyze Twitter: Core Twitter content analysis system.

Contains the main Analyzer class for programmatic content analysis.
CLI functionality has been moved to analyzer.cli module.
"""

import time
import warnings
import logging
import traceback
from typing import List, Optional
from datetime import datetime
from .categories import Categories
from .config import AnalyzerConfig
from .metrics import MetricsCollector
from .repository import ContentAnalysisRepository
from .flow_manager import AnalysisFlowManager
from .models import ContentAnalysis
from database import get_db_connection_context

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging for multimodal analysis debugging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Analyzer:
    """
    Dual-flow content analysis system:
    - Local flow: Pattern detection → Local LLM
    - External flow: Gemini multimodal (for non-general/political_general categories)
    - Evidence retrieval: For disinformation/conspiracy/numerical/temporal claims
    """

    def __init__(self, config: Optional[AnalyzerConfig] = None, verbose: bool = False, fast_mode: bool = False):
        """
        Initialize analyzer with dual-flow architecture.

        Args:
            config: Analyzer configuration (created with defaults if None)
            verbose: Whether to enable verbose logging
            fast_mode: Use simplified prompts for faster bulk processing
        """
        self.config = config or AnalyzerConfig()
        self.verbose = verbose
        self.fast_mode = fast_mode

        # Initialize components
        self.metrics = MetricsCollector()
        self.repository = ContentAnalysisRepository()

        # Initialize analysis flow manager (handles all 3 stages)
        self.flow_manager = AnalysisFlowManager(verbose=verbose, fast_mode=fast_mode)

        if self.verbose:
            print("🚀 Iniciando Analyzer con arquitectura dual-flow...")
            print("Componentes cargados:")
            print("- ✓ Flow Manager (pattern → local LLM → external)")
            print("- ✓ Local LLM: gemma3 (Ollama)")
            print("- ✓ External LLM: Gemini 2.5 Flash (multimodal)")
            print("- ✓ Recolector de métricas de rendimiento")
            print("- ✓ Repositorio de análisis de contenido")
            print("- ✓ Integración de recuperación de evidencia")
            print(f"- ✓ External analysis: {self.config.enable_external_analysis}")

    async def analyze_content(self,
                       tweet_id: str,
                       tweet_url: str,
                       username: str,
                       content: str,
                       media_urls: List[str] = None,
                       admin_override: bool = False) -> ContentAnalysis:
        """
        Main content analysis pipeline with dual-flow architecture.

        Flow:
        1. Local analysis (pattern → local LLM with gpt-oss:20b)
        2. External analysis (Gemini) for non-general/political_general categories OR admin override
        3. Evidence retrieval for disinformation/conspiracy/numerical/temporal claims

        Args:
            tweet_id: Tweet identifier
            tweet_url: Tweet URL
            username: Author username
            content: Tweet content text
            media_urls: Optional list of media URLs
            admin_override: Force external analysis regardless of category

        Returns:
            ContentAnalysis with both local and (optionally) external explanations
        """
        analysis_start_time = time.time()

        try:
            # Run dual-flow analysis
            analysis_result = await self.flow_manager.analyze_full(
                content=content,
                media_urls=media_urls,
                admin_override=admin_override,
                force_disable_external=not self.config.enable_external_analysis
            )

            # Create initial analysis result
            result = ContentAnalysis(
                post_id=tweet_id,
                post_url=tweet_url,
                author_username=username,
                post_content=content,
                analysis_timestamp=datetime.now().isoformat(),
                category=analysis_result.category,
                categories_detected=[analysis_result.category],  # Flow manager returns single primary category
                local_explanation=analysis_result.local_explanation,
                external_explanation=analysis_result.external_explanation or '',
                analysis_stages=analysis_result.stages.to_string(),
                external_analysis_used=analysis_result.stages.external,
                media_urls=media_urls or [],
                media_type=self._detect_media_type(media_urls) if media_urls else '',
                pattern_matches=analysis_result.pattern_data.get('pattern_matches', []),
                topic_classification=analysis_result.pattern_data.get('topic_classification', {}),
                analysis_json=f'{{"stages": "{analysis_result.stages.to_string()}", "has_media": {bool(media_urls)}}}',
                verification_data=analysis_result.verification_data if analysis_result.verification_data else None
            )

            # Track metrics
            analysis_time = time.time() - analysis_start_time
            self.metrics.record_analysis(
                method="dual_flow" if analysis_result.stages.external else "local_only",
                duration=analysis_time,
                category=result.category,
                model_used=self._get_model_name(result),
                is_multimodal=result.multimodal_analysis
            )

            # Add performance data to result
            result.analysis_time_seconds = analysis_time
            result.model_used = self._get_model_name(result)

            return result

        except Exception as e:
            # Re-raise all errors to stop the analysis pipeline
            if self.verbose:
                print(f"💥 Critical error in analyze_content: {e}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"Analysis failed: {str(e)}") from e
    def _detect_media_type(self, media_urls: List[str]) -> str:
        """
        Detect the primary media type from URLs.
        
        Args:
            media_urls: List of media URLs
            
        Returns:
            "image", "video", or "" if no media or mixed/unknown
        """
        if not media_urls:
            return ""
            
        # Check for video extensions and formats
        video_extensions = ['.mp4', '.m3u8', '.mov', '.avi', '.webm', '.m4v', '.flv', '.wmv']
        video_formats = ['format=mp4', 'format=m3u8', 'format=mov', 'format=avi', 'format=webm', 'format=m4v', 'format=flv', 'format=wmv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg']
        image_formats = ['format=jpg', 'format=jpeg', 'format=png', 'format=gif', 'format=webp', 'format=bmp', 'format=tiff', 'format=svg']
        
        has_videos = False
        has_images = False
        
        for url in media_urls:
            url_lower = url.lower()
            # Check if it's a video (extensions, format parameters, or 'video' in URL)
            if (any(ext in url_lower for ext in video_extensions) or 
                any(fmt in url_lower for fmt in video_formats) or 
                'video' in url_lower):
                has_videos = True
            # Check if it's an image (extensions, format parameters, or 'image' in URL)
            elif (any(ext in url_lower for ext in image_extensions) or 
                  any(fmt in url_lower for fmt in image_formats) or 
                  'image' in url_lower):
                has_images = True
        
        if has_videos and not has_images:
            return "video"
        elif has_images and not has_videos:
            return "image"
        elif has_images and has_videos:
            return "mixed"
        else:
            return ""
    
    def _get_model_name(self, result: ContentAnalysis) -> str:
        """Get the model name based on analysis result."""
        if result.external_analysis_used:
            if result.multimodal_analysis:
                return "gemma3+gemini-2.5-flash (multimodal)"
            return "gemma3+gemini-2.5-flash"
        return "gemma3"

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
            return self.repository.get_by_post_id(tweet_id)
        except Exception as e:
            if self.verbose:
                print(f"❌ Error retrieving analysis: {e}")
            return None

    def cleanup_resources(self):
        """Clean up any resources used by the analyzer."""
        try:
            # Cleanup flow manager resources (LLM pipelines, etc.)
            if hasattr(self.flow_manager, 'cleanup'):
                self.flow_manager.cleanup()
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")

    def print_system_status(self):
        """Print current system status for debugging."""
        print("🔧 ANALYZER SYSTEM STATUS")
        print("=" * 50)
        print(f"🤖 External Analysis Enabled: {self.config.enable_external_analysis}")
        print(f"⚡ Flow Manager: {'✓' if self.flow_manager else '❌'}")
        print(f"🔍 Pattern Analyzer: ✓")
        print(f"🧠 Local LLM (gemma3:4b): ✓")
        print(f"🎥 External Gemini (2.5 Flash): ✓")

        # Check database
        try:
            with get_db_connection_context() as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM content_analyses")
                count = c.fetchone()[0]
            print(f"📊 Database: {count} analyses stored")
        except Exception as e:
            print(f"❌ Database: Error - {e}")


# Utility functions for analyzer operations
def create_analyzer(config: Optional[AnalyzerConfig] = None, verbose: bool = False, fast_mode: bool = False) -> Analyzer:
    """Create and return an Analyzer instance with specified configuration."""
    return Analyzer(config=config, verbose=verbose, fast_mode=fast_mode)


async def reanalyze_tweet(tweet_id: str, verbose: bool = False, analyzer: Optional[Analyzer] = None) -> Optional[ContentAnalysis]:
    """Reanalyze a single tweet and return the result."""

    # Use provided analyzer or create default one
    if analyzer is None:
        analyzer = create_analyzer(verbose=verbose)

    # Get tweet data
    tweet_data = analyzer.repository.get_tweet_data(tweet_id)
    if not tweet_data:
        return None

    # Parse media URLs from media_links string
    media_urls = []
    if tweet_data.get('media_links'):
        media_urls = [url.strip() for url in tweet_data['media_links'].split(',') if url.strip()]

    # Combine main content with quoted/original context when available
    analysis_content = tweet_data['content']
    original_content = tweet_data.get('original_content')
    if original_content and original_content.strip():
        analysis_content = f"{analysis_content}\n\n[Contenido citado]: {original_content}"

    # Debug print
    if media_urls:
        print(f"    🖼️ Found {len(media_urls)} media files for analysis")
        for i, url in enumerate(media_urls):
            print(f"      {i+1}. {url[:50]}...")

    # Reanalyze FIRST (before deleting old analysis)
    analysis_result = await analyzer.analyze_content(
        tweet_id=tweet_data['tweet_id'],
        tweet_url=f"https://twitter.com/placeholder/status/{tweet_data['tweet_id']}",
        username=tweet_data['username'],
        content=analysis_content,
        media_urls=media_urls
    )

    # Preserve original tweet text in stored analysis for consistency with batch pipeline
    analysis_result.post_content = tweet_data['content']

    # Only delete existing analysis AFTER successful reanalysis
    analyzer.repository.delete_existing_analysis(tweet_id)

    # Save the new analysis result
    analyzer.save_analysis(analysis_result)

    return analysis_result