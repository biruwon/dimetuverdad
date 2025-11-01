"""
Multi-Model Analyzer for comparing analysis results across different models.
Orchestrates parallel/sequential analysis and aggregates results.
"""

import time
from typing import Dict, Tuple, Optional, List
from .ollama_analyzer import OllamaAnalyzer
from .categories import Categories


class MultiModelAnalyzer:
    """
    Orchestrates multi-model analysis for comparison and consensus building.
    Manages model selection, sequential execution, and result aggregation.
    """
    
    # Available models for multi-model analysis
    AVAILABLE_MODELS = {
        "gemma3:4b": {
            "type": "fast",
            "multimodal": True,
            "description": "Fast 4B parameter model"
        },
        "gemma3:12b": {
            "type": "fast",
            "multimodal": True,
            "description": "Fast 12B parameter model"
        },
        "gemma3:27b-it-qat": {
            "type": "accurate",
            "multimodal": True,
            "description": "Large 27B parameter model with quantization"
        },
        "gpt-oss:20b": {
            "type": "balanced",
            "multimodal": False,
            "description": "Balanced 20B parameter text-only model"
        }
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
        model: str
    ) -> Tuple[str, str, float]:
        """
        Analyze content with a specific model and track processing time.
        
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
                print(f"    🔬 Running {mode} analysis ({media_status})")
            
            # Run analysis
            if is_multimodal:
                # For multimodal analysis with pre-prepared media, we need to pass the prepared images
                # Extract base64 images from prepared content
                images = [media["image_url"]["url"] for media in prepared_media_content 
                         if media.get("type") == "image_url" and media.get("image_url", {}).get("url")]
                
                # Build prompts
                prompt = analyzer.prompt_generator.build_multimodal_categorization_prompt(content)
                system_prompt = analyzer.prompt_generator.build_ollama_multimodal_system_prompt()
                
                # Generate with pre-prepared images
                response = await analyzer.client.generate_multimodal(
                    prompt=prompt,
                    images=images,
                    model=model,
                    system_prompt=system_prompt,
                    options={
                        "temperature": analyzer.DEFAULT_TEMPERATURE_MULTIMODAL,
                        "top_p": analyzer.DEFAULT_TOP_P_MULTIMODAL,
                        "num_predict": analyzer.DEFAULT_NUM_PREDICT_MULTIMODAL,
                    },
                    keep_alive=analyzer.DEFAULT_KEEP_ALIVE
                )
            else:
                # Text-only analysis
                prompt = analyzer.prompt_generator.build_ollama_categorization_prompt(content)
                system_prompt = analyzer.prompt_generator.build_ollama_text_analysis_system_prompt()
                response = await analyzer.client.generate_text(
                    prompt=prompt,
                    model=model,
                    system_prompt=system_prompt,
                    options={
                        "temperature": analyzer.DEFAULT_TEMPERATURE_TEXT,
                        "num_predict": analyzer.DEFAULT_MAX_TOKENS,
                    }
                )
            
            # Parse response
            category, explanation = analyzer._parse_category_and_explanation(response)
            
            processing_time = time.time() - start_time
            
            return category, explanation, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise RuntimeError(f"Model {model} analysis failed: {str(e)}") from e
