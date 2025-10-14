"""
Text-only content analysis functionality.
"""

import json
import traceback
from typing import Dict, Tuple, Optional
from datetime import datetime

from .pattern_analyzer import PatternAnalyzer
from .llm_models import EnhancedLLMPipeline
from .models import ContentAnalysis
from .categories import Categories
from .config import AnalyzerConfig
from .prompts import EnhancedPromptGenerator
from utils.text_utils import normalize_text
from .constants import AnalysisMethods, ErrorMessages


class TextAnalyzer:
    """
    Handles text-only content analysis using pattern matching and LLM fallback.

    This class encapsulates all text analysis logic, making it easier to test
    and maintain separate from multimodal analysis.
    """

    def __init__(self, pattern_analyzer: Optional[PatternAnalyzer] = None,
                 llm_pipeline: Optional[EnhancedLLMPipeline] = None,
                 config: Optional['AnalyzerConfig'] = None,
                 verbose: bool = False):
        """
        Initialize text analyzer.

        Args:
            pattern_analyzer: Pattern analyzer instance (created if None)
            llm_pipeline: LLM pipeline instance (optional)
            config: Analyzer configuration for LLM setup
            verbose: Whether to enable verbose logging
        """
        self.pattern_analyzer = pattern_analyzer or PatternAnalyzer()
        self.llm_pipeline = llm_pipeline
        self.config = config
        self.verbose = verbose
        
        # Create LLM pipeline if not provided and config allows
        if self.llm_pipeline is None and self.config and self.config.use_llm:
            self.llm_pipeline = EnhancedLLMPipeline(model_priority=self.config.model_priority)

    def analyze(self, tweet_id: str, tweet_url: str, username: str, content: str) -> ContentAnalysis:
        """
        Perform text-only content analysis.

        Args:
            tweet_id: Twitter tweet ID
            tweet_url: Twitter tweet URL
            username: Twitter username
            content: Tweet content to analyze

        Returns:
            ContentAnalysis result
        """
        if self.verbose:
            print(f"\n🔍 Content analysis: @{username}")
            print(f"📝 Contenido: {content[:80]}...")

        # Normalize content for pattern matching and LLM prompts
        content_normalized = normalize_text(content)

        # Step 1: Pattern analysis
        pattern_results = self._run_pattern_analysis(content_normalized)

        # Step 2: Content categorization
        if self.verbose:
            print("🔍 Step 2: Categorization starting...")
        category, analysis_method = self._categorize_content(content_normalized, pattern_results)
        if self.verbose:
            print(f"🔍 Step 2: Category determined: {category}")

        # Step 3: LLM explanation generation
        llm_explanation = self._generate_llm_explanation(content_normalized, category, pattern_results)

        # Step 4: Build final analysis structure
        analysis_data = self._build_analysis_data(pattern_results)

        # Extract multi-category information
        pattern_result = pattern_results.get('pattern_result', None)
        categories_detected = pattern_result.categories if pattern_result else [category]

        return ContentAnalysis(
            post_id=tweet_id,
            post_url=tweet_url,
            author_username=username,
            post_content=content,
            analysis_timestamp=datetime.now().isoformat(),
            category=category,
            categories_detected=categories_detected,
            llm_explanation=llm_explanation,
            analysis_method=analysis_method,
            pattern_matches=[
                {'matched_text': pm.matched_text, 'category': pm.category, 'description': pm.description}
                for pm in (pattern_result.pattern_matches if pattern_result else [])
            ],
            topic_classification=analysis_data['topic_classification'],
            analysis_json=json.dumps(analysis_data, ensure_ascii=False, default=str)
        )

    def _run_pattern_analysis(self, content: str) -> Dict:
        """
        Run pattern analysis on normalized content.

        Args:
            content: Normalized content to analyze

        Returns:
            Dictionary containing pattern analysis results
        """
        pattern_result = self.pattern_analyzer.analyze_content(content)
        return {'pattern_result': pattern_result}

    def _categorize_content(self, content: str, pattern_results: Dict) -> Tuple[str, str]:
        """
        Determine content category using pattern results + LLM fallback.

        Args:
            content: Normalized content
            pattern_results: Results from pattern analysis

        Returns:
            Tuple of (category, analysis_method)
        """
        pattern_result = pattern_results['pattern_result']
        detected_categories = pattern_result.categories

        if self.verbose:
            print(f"🔍 Detected categories: {detected_categories}")

        # If patterns found any category, return the primary one
        if detected_categories:
            primary_category = detected_categories[0]
            if self.verbose:
                print(f"🎯 Pattern detected: {primary_category}")
            return primary_category, AnalysisMethods.PATTERN.value

        # No patterns detected - use LLM fallback if available
        if self.llm_pipeline:
            if self.verbose:
                print("🧠 No patterns detected - using LLM for analysis")
            llm_category = self._get_llm_category(content, pattern_results)
            if self.verbose:
                print(f"🔍 LLM category result: {llm_category}")
            return llm_category, AnalysisMethods.LLM.value
        else:
            # No patterns and no LLM - return general category with pattern method
            if self.verbose:
                print("🔍 No patterns detected and no LLM available - using general category")
            return Categories.GENERAL, AnalysisMethods.PATTERN.value

    def _get_llm_category(self, content: str, pattern_results: Dict) -> str:
        """
        Use LLM to categorize content when patterns are insufficient.

        Args:
            content: Content to categorize
            pattern_results: Pattern analysis results (for context)

        Returns:
            Predicted category
        """
        if not self.llm_pipeline:
            if self.verbose:
                print("🔍 LLM pipeline not available, returning general")
            return Categories.GENERAL

        try:
            if self.verbose:
                print(f"🔍 _get_llm_category called with content: {content[:50]}...")
                print("🔍 Calling llm_pipeline.get_category...")

            llm_category = self.llm_pipeline.get_category(content)
            return llm_category

        except Exception as e:
            print(ErrorMessages.LLM_CATEGORY_ERROR.format(error=e))
            return Categories.GENERAL

    def _generate_llm_explanation(self, content: str, category: str, pattern_results: Dict) -> str:
        """
        Generate detailed explanation using LLM with category-specific prompts.

        Args:
            content: Content to explain
            category: Detected category
            pattern_results: Pattern analysis results

        Returns:
            LLM-generated explanation
        """
        if not self.llm_pipeline:
            return ErrorMessages.LLM_PIPELINE_NOT_AVAILABLE

        try:
            # Import here to avoid circular imports
            # from .prompts import EnhancedPromptGenerator

            # Extract pattern result for context
            pattern_result = pattern_results.get('pattern_result', None)
            analysis_context = {
                'category': category,
                'analysis_mode': 'explanation',
                'detected_categories': pattern_result.categories if pattern_result else [],
                'has_patterns': len(pattern_result.categories) > 0 if pattern_result else False
            }

            if self.verbose:
                print(f"🔍 Generating category-specific explanation for: {category}")

            # Generate category-specific prompt
            prompt_generator = EnhancedPromptGenerator()
            custom_explanation_prompt = prompt_generator.generate_explanation_prompt(content, category)

            if self.verbose:
                print(f"🔍 Using category-specific prompt for {category}")

            # Temporarily replace prompt generator for custom prompt
            original_prompt_generator = self.llm_pipeline.prompt_generator

            class CustomPromptGenerator:
                def generate_explanation_prompt(self, text, category, model_type="ollama"):
                    return custom_explanation_prompt

            self.llm_pipeline.prompt_generator = CustomPromptGenerator()

            try:
                llm_explanation = self.llm_pipeline.get_explanation(content, category, analysis_context)

                if llm_explanation and len(llm_explanation.strip()) > 10:
                    if self.verbose:
                        print(f"✅ Generated explanation: {llm_explanation[:100]}...")
                    return llm_explanation
                else:
                    empty_response_info = f"LLM returned: '{llm_explanation}'" if llm_explanation else "LLM returned None/empty"
                    return ErrorMessages.LLM_EXPLANATION_FAILED.format(
                        details=empty_response_info,
                        length=len(llm_explanation.strip()) if llm_explanation else 0
                    )

            finally:
                # Always restore original prompt generator
                self.llm_pipeline.prompt_generator = original_prompt_generator

        except Exception as e:
            if self.verbose:
                print(f"❌ Error en explicación LLM: {e}")
                # import traceback
                traceback.print_exc()
            return ErrorMessages.LLM_EXPLANATION_EXCEPTION.format(
                error_type=type(e).__name__,
                error_message=str(e)
            )

    def _build_analysis_data(self, pattern_results: Dict) -> Dict:
        """
        Build the final analysis data structure.

        Args:
            pattern_results: Results from pattern analysis

        Returns:
            Analysis data dictionary
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
            'pattern_matches': [
                {'matched_text': pm.matched_text, 'category': pm.category, 'description': pm.description}
                for pm in pattern_matches
            ],
            'topic_classification': {
                'primary_topic': categories[0] if categories else Categories.GENERAL,
                'all_topics': [{'category': cat} for cat in categories],
                'political_context': political_context
            },
            'unified_categories': categories
        }