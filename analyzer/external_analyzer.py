"""
External Analyzer wrapper for Gemini multimodal analysis.
Provides independent analysis without context from local analyzer.
"""

from typing import List, Optional, Tuple
import asyncio
import os
from dataclasses import dataclass

from .gemini_multimodal import GeminiMultimodal
from .categories import Categories


@dataclass
class ExternalAnalysisResult:
    """Result from external analysis with category and explanation"""
    category: Optional[str]
    explanation: str


class ExternalAnalyzer:
    """
    Wrapper for external analysis services (currently Gemini).
    Analyzes both text and media content independently.
    No context passed from local analysis - Gemini makes its own determination.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize external analyzer with Gemini.
        
        Args:
            verbose: Enable detailed logging
        """
        self.gemini = GeminiMultimodal()
        self.verbose = verbose
        
        if self.verbose:
            print("🌐 ExternalAnalyzer initialized with Gemini 2.5 Flash")
    
    async def analyze(self, content: str, media_urls: Optional[List[str]] = None) -> ExternalAnalysisResult:
        """
        Perform independent external analysis using Gemini.
        
        Args:
            content: Text content to analyze
            media_urls: List of media URLs (if any)
        
        Returns:
            ExternalAnalysisResult with category and explanation (Spanish, 2-3 sentences)
        """
        if self.verbose:
            print(f"🌐 Running external analysis (Gemini)")
            print(f"📝 Content: {content[:100]}...")
            if media_urls:
                print(f"🖼️  Media: {len(media_urls)} URLs")
        
        try:
            # Determine if this is multimodal analysis
            if media_urls and len(media_urls) > 0:
                result = await self._analyze_multimodal(content, media_urls)
            else:
                result = await self._analyze_text_only(content)
            
            if self.verbose:
                print(f"✅ External analysis complete: {result.category} - {result.explanation[:100]}...")
            
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"❌ External analysis error: {e}")
            return ExternalAnalysisResult(
                category=None,
                explanation=f"Error en análisis externo: {str(e)}"
            )
    
    async def _analyze_multimodal(self, content: str, media_urls: List[str]) -> ExternalAnalysisResult:
        """
        Analyze content with media using Gemini multimodal capabilities.
        
        Args:
            content: Text content
            media_urls: List of media URLs
        
        Returns:
            ExternalAnalysisResult with category and explanation
        """
        # Select best media URL for analysis
        selected_media_url = self.gemini._select_media_url(media_urls)
        
        # Run Gemini multimodal analysis
        # Returns tuple: (analysis_text, analysis_time)
        analysis_result, analysis_time = await asyncio.to_thread(
            self.gemini.analyze_multimodal_content,
            [selected_media_url],
            content
        )
        
        if analysis_result is None:
            return ExternalAnalysisResult(
                category=None,
                explanation="No se pudo completar el análisis multimodal externo."
            )
        
        # Parse Gemini's structured response
        return self._parse_gemini_response(analysis_result)
    
    async def _analyze_text_only(self, content: str) -> ExternalAnalysisResult:
        """
        Analyze text-only content using Gemini multimodal infrastructure.
        
        Args:
            content: Text content to analyze
        
        Returns:
            ExternalAnalysisResult with category and explanation
        """
        # Use the same robust infrastructure as multimodal analysis
        analysis_result, analysis_time = await asyncio.to_thread(
            self.gemini.analyze_multimodal_content,
            [],  # Empty media list for text-only
            content
        )
        
        if analysis_result is None:
            return ExternalAnalysisResult(
                category=None,
                explanation="No se pudo completar el análisis de texto externo."
            )
        
        # Parse Gemini's structured response
        return self._parse_gemini_response(analysis_result)
    
    def _parse_gemini_response(self, response: str) -> ExternalAnalysisResult:
        """
        Parse Gemini's structured response to extract category and explanation.
        
        Expected format:
        CATEGORÍA: category_name
        EXPLICACIÓN: explanation text
        
        Args:
            response: Raw Gemini response
        
        Returns:
            ExternalAnalysisResult with parsed category and explanation
        """
        if not response or len(response.strip()) == 0:
            return ExternalAnalysisResult(
                category=None,
                explanation="Análisis externo no disponible."
            )
        
        lines = response.strip().split('\n')
        category = None
        explanation = response  # Fallback to full response
        
        # Extract category and explanation from structured response
        for line in lines:
            line = line.strip()
            if line.upper().startswith("CATEGORÍA:") or line.upper().startswith("CATEGORY:"):
                category_part = line.split(":", 1)[1].strip().lower()
                category = self._map_category_name(category_part)
            elif line.upper().startswith("EXPLICACIÓN:"):
                explanation = line.split(":", 1)[1].strip()
                break
        
        # If no structured explanation found, use full response
        # but clean up any category prefix
        if explanation == response:
            # Remove category line if present
            cleaned_lines = []
            for line in lines:
                if not (line.upper().startswith("CATEGORÍA:") or line.upper().startswith("CATEGORY:")):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                explanation = " ".join(cleaned_lines).strip()
        
        # Validate explanation is meaningful
        if not explanation or len(explanation.strip()) < 10:
            explanation = "Análisis externo completado sin detalles adicionales."
        
        return ExternalAnalysisResult(category=category, explanation=explanation)
    
    def _map_category_name(self, category_name: str) -> Optional[str]:
        """
        Map category name from Gemini response to our Categories enum.
        
        Args:
            category_name: Raw category name from Gemini
            
        Returns:
            Mapped category string or None if not recognized
        """
        # Map common category names to our Categories enum
        category_mapping = {
            'hate_speech': Categories.HATE_SPEECH,
            'disinformation': Categories.DISINFORMATION,
            'conspiracy_theory': Categories.CONSPIRACY_THEORY,
            'anti_immigration': Categories.ANTI_IMMIGRATION,
            'anti_lgbtq': Categories.ANTI_LGBTQ,
            'anti_feminism': Categories.ANTI_FEMINISM,
            'nationalism': Categories.NATIONALISM,
            'anti_government': Categories.ANTI_GOVERNMENT,
            'historical_revisionism': Categories.HISTORICAL_REVISIONISM,
            'call_to_action': Categories.CALL_TO_ACTION,
            'political_general': Categories.POLITICAL_GENERAL,
            'general': Categories.GENERAL,
            # Spanish variations
            'discurso_odio': Categories.HATE_SPEECH,
            'desinformación': Categories.DISINFORMATION,
            'teoría_conspiración': Categories.CONSPIRACY_THEORY,
            'anti_inmigración': Categories.ANTI_IMMIGRATION,
            'anti_lgbt': Categories.ANTI_LGBTQ,
            'anti_feminista': Categories.ANTI_FEMINISM,
            'nacionalismo': Categories.NATIONALISM,
            'anti_gobierno': Categories.ANTI_GOVERNMENT,
            'revisionismo_histórico': Categories.HISTORICAL_REVISIONISM,
            'llamada_accion': Categories.CALL_TO_ACTION,
            'política_general': Categories.POLITICAL_GENERAL
        }
        
        return category_mapping.get(category_name.lower())
