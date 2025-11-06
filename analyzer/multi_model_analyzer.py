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
    
    # Available prompt variants for single-model comparison
    PROMPT_VARIANTS = {
        "normal": {
            "description": "Standard detailed prompts for comprehensive analysis",
            "fast_mode": False,
            "multimodal": True,
            "timeout": 300  # 5 minutes for complex prompts
        },
        "fast": {
            "description": "Simplified prompts optimized for speed (text-only)",
            "fast_mode": True,
            "multimodal": False,  # Fast mode skips multimodal to be actually fast
            "timeout": 120  # 2 minutes for simple prompts
        }
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize multi-model analyzer.
        
        Args:
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.analyzers = {}  # Cache of OllamaAnalyzer instances per model+variant
        
        if self.verbose:
            print("🔬 MultiModelAnalyzer initialized")
    
    def _get_analyzer(self, model: str, fast_mode: bool = False) -> OllamaAnalyzer:
        """
        Get or create analyzer instance for a specific model and prompt variant.
        
        Args:
            model: Model name
            fast_mode: Whether to use fast mode prompts
        
        Returns:
            OllamaAnalyzer instance for the model+variant combination
        """
        key = f"{model}_{'fast' if fast_mode else 'normal'}"
        if key not in self.analyzers:
            if self.verbose:
                print(f"🤖 Creating OllamaAnalyzer for model: {model}, fast_mode: {fast_mode}")
            self.analyzers[key] = OllamaAnalyzer(model=model, verbose=self.verbose, fast_mode=fast_mode)
        return self.analyzers[key]
    
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
            temp_analyzer = self._get_analyzer(models[0], fast_mode=False)
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
            temp_analyzer = self._get_analyzer(models[0], fast_mode=False)
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
    
    async def analyze_with_prompt_variants(
        self,
        content: str,
        media_urls: Optional[List[str]] = None,
        model: str = "gemma3:27b-it-q4_K_M",
        variants: Optional[List[str]] = None
    ) -> Dict[str, Tuple[str, str, float]]:
        """
        Analyze content with different prompt variants using the same model.
        
        Args:
            content: Text content to analyze
            media_urls: Optional list of media URLs
            model: Model to use for all variants
            variants: List of prompt variants to use (defaults to all available)
        
        Returns:
            Dictionary mapping variant_name -> (category, explanation, processing_time)
            Example: {
                "normal": ("hate_speech", "Explanation...", 25.3),
                "fast": ("hate_speech", "Different explanation...", 15.2)
            }
        
        Raises:
            RuntimeError: If no valid variants are available
        """
        # Use all variants if not specified
        if variants is None:
            variants = list(self.PROMPT_VARIANTS.keys())
        
        if self.verbose:
            print(f"🔍 Running prompt variant analysis with model {model}")
            print(f"📝 Variants: {', '.join(variants)}")
        
        # Check if we have media and if it's video-only
        has_media = media_urls is not None and len(media_urls) > 0
        if has_media:
            # Use first variant to check video-only status
            temp_analyzer = self._get_analyzer(model, fast_mode=self.PROMPT_VARIANTS[variants[0]]["fast_mode"])
            if temp_analyzer._has_only_videos(media_urls):
                if self.verbose:
                    print("🎥 Content contains only videos - analyzing text only")
                media_urls = None
                has_media = False
        
        # Execute analyses sequentially with different prompt variants
        analysis_results = {}
        
        for i, variant in enumerate(variants, 1):
            # Skip unknown variants
            variant_info = self.PROMPT_VARIANTS.get(variant)
            if not variant_info:
                if self.verbose:
                    print(f"⚠️  Unknown variant: {variant}, skipping")
                continue
            
            fast_mode = variant_info["fast_mode"]
            use_multimodal = has_media and variant_info.get("multimodal", True)
            
            if self.verbose:
                mode_desc = "MULTIMODAL" if use_multimodal else "TEXT-ONLY"
                print(f"⚡ Analyzing with variant {i}/{len(variants)}: {variant} ({mode_desc}, fast_mode: {fast_mode})")
            
            try:
                timeout = variant_info.get("timeout", 120)  # Use variant-specific timeout, fallback to 120s
                category, explanation, processing_time = await self._analyze_with_specific_variant(
                    content=content,
                    media_urls=media_urls if use_multimodal else None,  # Pass original URLs for multimodal
                    model=model,
                    fast_mode=fast_mode,
                    timeout=timeout
                )
                
                analysis_results[variant] = (category, explanation, processing_time)
                
                if self.verbose:
                    print(f"✅ Variant {variant}: {category} ({processing_time:.1f}s)")
                    
            except Exception as e:
                if self.verbose:
                    print(f"❌ Variant {variant} failed: {str(e)}")
                # Store error result
                analysis_results[variant] = (
                    Categories.GENERAL,
                    f"Error durante el análisis: {str(e)[:100]}",
                    0.0
                )
        
        if not analysis_results:
            raise RuntimeError("No valid prompt variants available for analysis")
        
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
                        "top_p": analyzer.DEFAULT_TOP_P,
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
                        "temperature": analyzer.DEFAULT_TEMPERATURE_MULTIMODAL,
                        "top_p": analyzer.DEFAULT_TOP_P,
                        "num_predict": analyzer.DEFAULT_NUM_PREDICT_MULTIMODAL,
                    }
                )
            
            # Parse response
            category, explanation = analyzer._parse_category_and_explanation(response)
            
            processing_time = time.time() - start_time
            
            return category, explanation, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise RuntimeError(f"Model {model} analysis failed: {str(e)}") from e
    
    async def _analyze_with_specific_variant(
        self,
        content: str,
        media_urls: Optional[List[str]],
        model: str,
        fast_mode: bool,
        timeout: int = 120
    ) -> Tuple[str, str, float]:
        """
        Analyze content with a specific model and prompt variant.
        
        Args:
            content: Text content to analyze
            media_urls: Optional list of media URLs
            model: Model name to use
            fast_mode: Whether to use fast mode prompts
        
        Returns:
            Tuple of (category, explanation, processing_time_seconds)
        
        Raises:
            RuntimeError: If variant analysis fails
        """
        start_time = time.time()
        
        try:
            # Get analyzer for this model+variant combination
            analyzer = self._get_analyzer(model, fast_mode)
            
            if self.verbose:
                variant_desc = "fast" if fast_mode else "normal"
                media_status = f"with {len(media_urls)} media URLs" if media_urls else "no media"
                print(f"    🔬 Running analysis ({variant_desc} prompts, {media_status})")
            
            # Run analysis using the analyzer's categorize_and_explain method
            # This will automatically use fast prompts when fast_mode=True
            category, explanation = await analyzer.categorize_and_explain(
                content=content,
                media_urls=media_urls,
                timeout=timeout
            )
            
            processing_time = time.time() - start_time
            
            return category, explanation, processing_time
            
        except Exception as e:
            processing_time = time.time() - start_time
            raise RuntimeError(f"Variant {fast_mode} analysis failed: {str(e)}") from e
