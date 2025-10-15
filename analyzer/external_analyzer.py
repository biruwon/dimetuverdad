"""
External Analyzer wrapper for Gemini multimodal analysis.
Provides independent analysis without context from local analyzer.
"""

from typing import List, Optional
import asyncio
import os

from .gemini_multimodal import GeminiMultimodal
from .categories import Categories


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
    
    async def analyze(self, content: str, media_urls: Optional[List[str]] = None) -> str:
        """
        Perform independent external analysis using Gemini.
        
        Args:
            content: Text content to analyze
            media_urls: List of media URLs (if any)
        
        Returns:
            External explanation (Spanish, 2-3 sentences)
        """
        if self.verbose:
            print(f"🌐 Running external analysis (Gemini)")
            print(f"📝 Content: {content[:100]}...")
            if media_urls:
                print(f"🖼️  Media: {len(media_urls)} URLs")
        
        try:
            # Determine if this is multimodal analysis
            if media_urls and len(media_urls) > 0:
                explanation = await self._analyze_multimodal(content, media_urls)
            else:
                explanation = await self._analyze_text_only(content)
            
            if self.verbose:
                print(f"✅ External analysis complete: {explanation[:100]}...")
            
            return explanation
            
        except Exception as e:
            if self.verbose:
                print(f"❌ External analysis error: {e}")
            return f"Error en análisis externo: {str(e)}"
    
    async def _analyze_multimodal(self, content: str, media_urls: List[str]) -> str:
        """
        Analyze content with media using Gemini multimodal capabilities.
        
        Args:
            content: Text content
            media_urls: List of media URLs
        
        Returns:
            Multimodal explanation combining text and visual analysis
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
            return "No se pudo completar el análisis multimodal externo."
        
        # Parse Gemini's structured response
        explanation = self._parse_gemini_response(analysis_result)
        
        return explanation
    
    async def _analyze_text_only(self, content: str) -> str:
        """
        Analyze text-only content using Gemini multimodal infrastructure.
        
        Args:
            content: Text content to analyze
        
        Returns:
            Text-based explanation from Gemini
        """
        # Use the same robust infrastructure as multimodal analysis
        analysis_result, analysis_time = await asyncio.to_thread(
            self.gemini.analyze_multimodal_content,
            [],  # Empty media list for text-only
            content
        )
        
        if analysis_result is None:
            return "No se pudo completar el análisis de texto externo."
        
        # Parse Gemini's structured response
        explanation = self._parse_gemini_response(analysis_result)
        
        return explanation
    
    def _parse_gemini_response(self, response: str) -> str:
        """
        Parse Gemini's structured response to extract explanation.
        
        Expected format:
        CATEGORÍA: category_name
        EXPLICACIÓN: explanation text
        
        Args:
            response: Raw Gemini response
        
        Returns:
            Parsed explanation text
        """
        if not response or len(response.strip()) == 0:
            return "Análisis externo no disponible."
        
        lines = response.strip().split('\n')
        explanation = response  # Fallback to full response
        
        # Extract explanation part if structured response
        for line in lines:
            line = line.strip()
            if line.upper().startswith("EXPLICACIÓN:"):
                explanation = line.split(":", 1)[1].strip()
                break
        
        # If no structured explanation found, use full response
        # but clean up any category prefix
        if explanation == response:
            # Remove category line if present
            cleaned_lines = []
            for line in lines:
                if not line.upper().startswith("CATEGORÍA:"):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                explanation = " ".join(cleaned_lines).strip()
        
        # Validate explanation is meaningful
        if not explanation or len(explanation.strip()) < 10:
            return "Análisis externo completado sin detalles adicionales."
        
        return explanation
