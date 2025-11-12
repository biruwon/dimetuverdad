"""
Multi-Model Analyzer for comparing analysis results across different models.
Orchestrates parallel/sequential analysis and aggregates results using multi-stage approach.
"""

import time
from typing import Dict, Tuple, Optional, List
from .ollama_analyzer import OllamaAnalyzer
from .categories import Categories


class MultiModelAnalyzer:
    """
    Orchestrates multi-model analysis for comparison and consensus building.
    Uses multi-stage analysis approach: category detection → media description → explanation generation.
    Manages model selection, sequential execution, and result aggregation.
    """
    
    # Available models for multi-model analysis
    AVAILABLE_MODELS = {
        "gemma3:27b-it-q4_K_M": {
            "type": "accurate",
            "multimodal": True,
            "description": "Large 27B parameter model with quantization"
        },
        "gemma3:4b": {
            "type": "fast",
            "multimodal": True,
            "description": "Fast 4B parameter model"
        },
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize multi-model analyzer.
        
        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.analyzers = {}  # Cache of OllamaAnalyzer instances per model
        
        if self.verbose:
            print("🔬 MultiModelAnalyzer initialized")
    
    def _get_analyzer(self, model: str) -> OllamaAnalyzer:
        """
        Get or create analyzer instance for a specific model.
        
        Args:
            model: Model name
        
        Returns:
            OllamaAnalyzer instance for the model
        """
        if model not in self.analyzers:
            if self.verbose:
                print(f"🤖 Creating OllamaAnalyzer for model: {model}")
            self.analyzers[model] = OllamaAnalyzer(model=model, verbose=self.verbose)
        return self.analyzers[model]
    
    async def analyze_with_multiple_models(
        self,
        content: str,
        media_urls: Optional[List[str]] = None,
        models: Optional[List[str]] = None
    ) -> Dict[str, Tuple[str, str, float]]:
        """
        Analyze content with multiple models sequentially for comparison.
        
        Args:
            content: Text content to analyze
            media_urls: Optional list of media URLs
            models: List of model names to use (defaults to all available models)
        
        Returns:
            Dictionary mapping model_name -> (category, explanation, processing_time)
            Example: {
                "gemma3:4b": ("hate_speech", "Explanation...", 25.3),
                "gpt-oss:20b": ("hate_speech", "Different explanation...", 45.2)
            }
        
        Raises:
            RuntimeError: If no valid models are available
        """
        # Use all models if not specified
        if models is None:
            models = list(self.AVAILABLE_MODELS.keys())
        
        if self.verbose:
            print(f"🔍 Running multi-model analysis with {len(models)} models")
            print(f"📝 Models: {', '.join(models)}")
        
        # Check if we have media and if it's video-only
        has_media = media_urls is not None and len(media_urls) > 0
        if has_media:
            # Use first available analyzer to check video-only status
            temp_analyzer = self._get_analyzer(models[0])
            if temp_analyzer._has_only_videos(media_urls):
                if self.verbose:
                    print("🎥 Content contains only videos - analyzing text only")
                media_urls = None
                has_media = False
        
        # Prepare media content once for all multimodal models
        prepared_media_content = None
        if has_media:
            if self.verbose:
                print("📥 Preparing media content for multimodal analysis...")
            # Use first available analyzer to prepare media
            temp_analyzer = self._get_analyzer(models[0])
            prepared_media_content = await temp_analyzer._prepare_media_content(media_urls)
            if prepared_media_content and self.verbose:
                print(f"📦 Prepared {len(prepared_media_content)} media items")
            elif not prepared_media_content and self.verbose:
                print("⚠️  No valid media content prepared, falling back to text-only")
                has_media = False
        
        # Execute analyses sequentially
        analysis_results = {}
        
        for i, model in enumerate(models, 1):
            # Skip unknown models
            model_info = self.AVAILABLE_MODELS.get(model)
            if not model_info:
                if self.verbose:
                    print(f"⚠️  Unknown model: {model}, skipping")
                continue
            
            # Determine if this model should use multimodal analysis
            use_multimodal = has_media and model_info.get("multimodal", False) and prepared_media_content
            
            if self.verbose:
                print(f"⚡ Analyzing with model {i}/{len(models)}: {model}")
            
            try:
                category, explanation, processing_time = await self._analyze_with_specific_model(
                    content=content,
                    prepared_media_content=prepared_media_content if use_multimodal else None,
                    original_media_urls=media_urls if use_multimodal else None,
                    model=model
                )
                
                analysis_results[model] = (category, explanation, processing_time)
                
                if self.verbose:
                    print(f"✅ Model {model}: {category} ({processing_time:.1f}s)")
                    
            except Exception as e:
                if self.verbose:
                    print(f"❌ Model {model} failed: {str(e)}")
                # Store error result
                analysis_results[model] = (
                    Categories.GENERAL,
                    f"Error durante el análisis: {str(e)[:100]}",
                    0.0
                )
        
        if not analysis_results:
            raise RuntimeError("No valid models available for analysis")
        
        return analysis_results
    
    async def _analyze_with_specific_model(
        self,
        content: str,
        prepared_media_content: Optional[List[dict]],
        original_media_urls: Optional[List[str]],
        model: str
    ) -> Tuple[str, str, float]:
        """
        Analyze content with a specific model using multi-stage approach.
        
        Args:
            content: Text content to analyze
            prepared_media_content: Optional pre-prepared media content (base64 encoded)
            model: Model name to use
        
        Returns:
            Tuple of (category, explanation, processing_time_seconds)
        
        Raises:
            RuntimeError: If model analysis fails
        """
        start_time = time.time()
        
        try:
            # Get analyzer for this model
            analyzer = self._get_analyzer(model)
            
            # Determine if this is multimodal
            model_info = self.AVAILABLE_MODELS.get(model, {})
            is_multimodal = model_info.get("multimodal", False) and prepared_media_content
            
            if self.verbose:
                mode = "MULTIMODAL" if is_multimodal else "TEXT-ONLY"
                media_status = f"with {len(prepared_media_content)} media items" if prepared_media_content else "no media"
                print(f"    🔬 Running {mode} multi-stage analysis ({media_status})")
            
            # Stage 1: Category detection
            if self.verbose:
                print("    📋 Stage 1: Category detection")
            category = await analyzer.detect_category_only(content)
            
            # Stage 2: Media description (if multimodal)
            media_description = None
            if is_multimodal:
                if self.verbose:
                    print("    🖼️  Stage 2: Media description")
                if original_media_urls:
                    media_description = await analyzer.describe_media(original_media_urls)
            
            # Stage 3: Explanation generation
            if self.verbose:
                print("    💭 Stage 3: Explanation generation")
            explanation = await analyzer.generate_explanation_with_context(
                content=content,
                category=category,
                media_description=media_description
            )
            
            processing_time = time.time() - start_time
            
            return category, explanation, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise RuntimeError(f"Model {model} analysis failed: {str(e)}") from e
