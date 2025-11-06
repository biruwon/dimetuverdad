"""
Ollama Analyzer for single-model content analysis.
Handles media preparation, response parsing, and analysis workflows.
"""

import base64
import re
import requests
from typing import Tuple, Optional, List
from .ollama_client import OllamaClient, OllamaRetryError
from .categories import Categories
from .prompts import EnhancedPromptGenerator

class OllamaAnalyzer:
    """
    High-level analyzer for Ollama models supporting text and multimodal analysis.
    Handles media preparation, response parsing, and analysis workflows.
    """
    
    # Default generation parameters
    DEFAULT_TEMPERATURE_TEXT = 0.1 # Lower for faster convergence
    DEFAULT_MAX_TOKENS = 60 # Reduced for speed (categorization doesn't need long responses)
    DEFAULT_TEMPERATURE_MULTIMODAL = 0.1
    DEFAULT_TOP_P = 0.8 # Slightly more focused for speed
    DEFAULT_NUM_PREDICT_MULTIMODAL = 60 # Reduced for speed
    DEFAULT_KEEP_ALIVE = "72h"
    DEFAULT_NUM_CTX = 8192  # Limit context window to prevent unbounded growth
    #DETAULT_SEED = 42 # just a fixed number to force determinist responses
    
    # Media handling settings
    DEFAULT_MEDIA_TIMEOUT = 5.0
    DEFAULT_MAX_MEDIA_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_MEDIA_ITEMS = 3  # Process up to 3 media files
    
    def __init__(self, model: str = "gemma3:27b-it-q4_K_M", verbose: bool = False, fast_mode: bool = False):
        """
        Initialize Ollama analyzer.
        
        Args:
            model: Default model to use
            verbose: Enable detailed logging
            fast_mode: Use simplified prompts for faster bulk processing
        """
        self.model = model
        self.verbose = verbose
        self.fast_mode = fast_mode
        self.client = OllamaClient(verbose=verbose)
        self.prompt_generator = EnhancedPromptGenerator()
        
        if self.verbose:
            print(f"🤖 OllamaAnalyzer initialized with model: {self.model} (fast_mode: {fast_mode})")
    
    async def reset_model_context(self, keep_alive: str = "72h"):
        """
        Reset model context without unloading the model.
        Sends a minimal request to clear conversation history while keeping model hot.
        
        Args:
            keep_alive: How long to keep model loaded after reset
        """
        await self.client.reset_model_context(self.model, keep_alive)
    
    def _get_fast_system_prompt(self) -> str:
        """Get simplified system prompt for fast mode."""
        return self.prompt_generator.build_fast_system_prompt()

    def _get_fast_categorization_prompt(self, content: str) -> str:
        """Get simplified categorization prompt for fast mode."""
        return self.prompt_generator.build_fast_categorization_prompt(content)

    def _get_fast_explanation_prompt(self, content: str, category: str) -> str:
        """Get simplified explanation prompt for fast mode."""
        return self.prompt_generator.build_fast_explanation_prompt(content, category)
    
    async def categorize_and_explain(
        self,
        content: str,
        media_urls: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> Tuple[str, str]:
        """
        Analyze content and return both category and explanation.
        Uses multimodal analysis if media is provided, text-only otherwise.
        
        Args:
            content: Text content to analyze
            media_urls: Optional list of media URLs (images)
            timeout: Optional timeout in seconds for LLM calls
        
        Returns:
            Tuple of (category, explanation)
        
        Raises:
            RuntimeError: If analysis fails
        """
        import time
        total_start = time.time()
        
        if self.verbose:
            print(f"🔍 Running {'multimodal' if media_urls else 'text'} categorization + explanation")
            print(f"📝 Content: {content[:100]}...")
            if media_urls:
                print(f"🖼️  Media URLs: {len(media_urls)} items")
        
        # Check if content contains only videos (can't process videos)
        if media_urls and self._has_only_videos(media_urls):
            if self.verbose:
                print("🎥 Content contains only videos - analyzing text only")
            media_urls = None
        
        try:
            # Phase 1: Build prompts
            prompt_start = time.time()
            if self.fast_mode:
                if media_urls:
                    prompt = self.prompt_generator.build_fast_multimodal_categorization_prompt(content)
                else:
                    prompt = self._get_fast_categorization_prompt(content)
                system_prompt = self._get_fast_system_prompt()
            elif media_urls:
                prompt = self.prompt_generator.build_multimodal_categorization_prompt(content)
                system_prompt = self.prompt_generator.build_ollama_multimodal_system_prompt()
            else:
                prompt = self.prompt_generator.build_ollama_categorization_prompt(content)
                system_prompt = self.prompt_generator.build_ollama_text_analysis_system_prompt()
            prompt_time = time.time() - prompt_start
            
            # Phase 2: Prepare media (if needed)
            media_prep_start = time.time()
            media_content = None
            if media_urls:
                media_content = await self._prepare_media_content(media_urls)
                if not media_content:
                    if self.verbose:
                        print("⚠️  No valid media found, falling back to text-only")
            media_prep_time = time.time() - media_prep_start
            
            # Phase 3: Generate LLM response
            llm_start = time.time()
            if media_urls and media_content:
                response = await self._generate_multimodal(prompt, media_content, system_prompt, timeout)
            else:
                response = await self._generate_text(prompt, system_prompt, timeout)
            llm_time = time.time() - llm_start
            
            # Phase 4: Parse response
            parse_start = time.time()
            category, explanation = self._parse_category_and_explanation(response)
            parse_time = time.time() - parse_start
            
            total_time = time.time() - total_start
            
            if self.verbose:
                print(f"✅ Category: {category}")
                print(f"💭 Explanation: {explanation[:100]}...")
                print(f"⏱️  Timing breakdown:")
                print(f"   📝 Prompt building: {prompt_time:.3f}s")
                print(f"   🖼️  Media preparation: {media_prep_time:.3f}s")
                print(f"   🤖 LLM generation: {llm_time:.3f}s")
                print(f"   🔍 Response parsing: {parse_time:.3f}s")
                print(f"   📊 Total analysis: {total_time:.3f}s")
            
            return category, explanation
            
        except Exception as e:
            total_time = time.time() - total_start
            if self.verbose:
                print(f"❌ Analysis failed after {total_time:.3f}s: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Analysis failed: {str(e)}") from e
    
    async def explain_only(
        self,
        content: str,
        category: str,
        media_urls: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        Generate explanation for already-known category.
        
        Args:
            content: Content to explain
            category: Already-detected category
            media_urls: Optional list of media URLs
            timeout: Optional timeout in seconds for LLM calls
        
        Returns:
            Explanation (Spanish, 2-3 sentences)
        
        Raises:
            RuntimeError: If explanation generation fails
        """
        import time
        total_start = time.time()
        
        if self.verbose:
            print(f"🔍 Generating {'multimodal' if media_urls else 'text'} explanation for: {category}")
        
        # Check if content contains only videos
        if media_urls and self._has_only_videos(media_urls):
            if self.verbose:
                print("🎥 Content contains only videos - analyzing text only")
            media_urls = None
        
        try:
            # Phase 1: Build prompts
            prompt_start = time.time()
            if self.fast_mode:
                if media_urls:
                    prompt = self.prompt_generator.build_fast_multimodal_explanation_prompt(content, category)
                else:
                    prompt = self._get_fast_explanation_prompt(content, category)
                system_prompt = self._get_fast_system_prompt()
            elif media_urls:
                prompt = self.prompt_generator.build_multimodal_explanation_prompt(content, category)
                system_prompt = self.prompt_generator.build_ollama_multimodal_system_prompt()
            else:
                prompt = self.prompt_generator.build_ollama_text_explanation_prompt(content, category, model_type="ollama")
                system_prompt = self.prompt_generator.build_ollama_text_analysis_system_prompt()
            prompt_time = time.time() - prompt_start
            
            # Phase 2: Prepare media (if needed)
            media_prep_start = time.time()
            media_content = None
            if media_urls:
                media_content = await self._prepare_media_content(media_urls)
                if not media_content:
                    if self.verbose:
                        print("⚠️  No valid media found, falling back to text-only")
            media_prep_time = time.time() - media_prep_start
            
            # Phase 3: Generate LLM response
            llm_start = time.time()
            if media_urls and media_content:
                response = await self._generate_multimodal(prompt, media_content, system_prompt, timeout)
            else:
                response = await self._generate_text(prompt, system_prompt, timeout)
            llm_time = time.time() - llm_start
            
            # Phase 4: Parse response
            parse_start = time.time()
            explanation = response.strip()
            
            # Clean up any format prefixes that the LLM might have added
            explanation = re.sub(r'^CATEGORÍA:\s*[^\n]*\n?', '', explanation, flags=re.IGNORECASE | re.MULTILINE)
            explanation = re.sub(r'^EXPLICACIÓN:\s*', '', explanation, flags=re.IGNORECASE | re.MULTILINE)
            
            if not explanation or len(explanation.strip()) < 10:
                explanation = f"Contenido clasificado como {category}."
            parse_time = time.time() - parse_start
            
            total_time = time.time() - total_start
            
            if self.verbose:
                print(f"💭 Explanation: {explanation[:100]}...")
                print(f"⏱️  Timing breakdown:")
                print(f"   📝 Prompt building: {prompt_time:.3f}s")
                print(f"   🖼️  Media preparation: {media_prep_time:.3f}s")
                print(f"   🤖 LLM generation: {llm_time:.3f}s")
                print(f"   🔍 Response parsing: {parse_time:.3f}s")
                print(f"   📊 Total analysis: {total_time:.3f}s")
            
            return explanation
            
        except Exception as e:
            total_time = time.time() - total_start
            if self.verbose:
                print(f"❌ Explanation generation failed after {total_time:.3f}s: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Explanation generation failed: {str(e)}") from e
    
    async def _generate_text(self, prompt: str, system_prompt: str, timeout: Optional[float] = None) -> str:
        """Generate text-only response."""
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"📝 Starting text generation at {time.strftime('%H:%M:%S')}")
        
        try:
            result = await self.client.generate_text(
                prompt=prompt,
                model=self.model,
                system_prompt=system_prompt,
                options={
                    "temperature": self.DEFAULT_TEMPERATURE_TEXT,
                    "num_predict": self.DEFAULT_MAX_TOKENS,
                    "top_p": self.DEFAULT_TOP_P,
                    "num_ctx": self.DEFAULT_NUM_CTX  # Limit context window
                },
                keep_alive=self.DEFAULT_KEEP_ALIVE,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            if self.verbose:
                print(f"✅ Text generation completed in {duration:.2f}s")
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            if self.verbose:
                print(f"❌ Text generation failed after {duration:.2f}s: {type(e).__name__}: {str(e)}")
            raise
    
    async def _generate_multimodal(
        self,
        prompt: str,
        media_content: List[dict],
        system_prompt: str,
        timeout: Optional[float] = None
    ) -> str:
        """Generate multimodal response with images."""
        try:
            # Extract base64 images
            images = [media["image_url"]["url"] for media in media_content 
                     if media.get("type") == "image_url" and media.get("image_url", {}).get("url")]
            
            if not images:
                raise RuntimeError("No valid images found in media content")
            
            return await self.client.generate_multimodal(
                prompt=prompt,
                images=images,
                model=self.model,
                system_prompt=system_prompt,
                options={
                    "temperature": self.DEFAULT_TEMPERATURE_TEXT,
                    "num_predict": self.DEFAULT_MAX_TOKENS,
                    "top_p": self.DEFAULT_TOP_P,
                    "num_ctx": self.DEFAULT_NUM_CTX  # Limit context window
                },
                keep_alive=self.DEFAULT_KEEP_ALIVE,
                timeout=timeout
            )
        except OllamaRetryError as e:
            raise RuntimeError(f"Multimodal generation failed: {str(e)}") from e
    
    async def _prepare_media_content(self, media_urls: List[str]) -> List[dict]:
        """
        Download and prepare media content for analysis.
        Only processes images (skips videos).
        
        Args:
            media_urls: List of media URLs
        
        Returns:
            List of prepared media dicts with base64 data
        """
        media_content = []
        
        # Define supported formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg']
        image_formats = ['format=jpg', 'format=jpeg', 'format=png', 'format=gif', 
                        'format=webp', 'format=bmp', 'format=tiff', 'format=svg']
        video_extensions = ['.mp4', '.m3u8', '.mov', '.avi', '.webm', '.m4v', '.flv', '.wmv']
        video_formats = ['format=mp4', 'format=m3u8', 'format=mov', 'format=avi', 
                        'format=webm', 'format=m4v', 'format=flv', 'format=wmv']
        
        for url in media_urls[:self.MAX_MEDIA_ITEMS]:
            try:
                url_lower = url.lower()
                
                # Check file extension first
                has_video_ext = any(url_lower.endswith(ext) for ext in video_extensions)
                has_image_ext = any(url_lower.endswith(ext) for ext in image_extensions)
                has_video_format = any(fmt in url_lower for fmt in video_formats)
                has_image_format = any(fmt in url_lower for fmt in image_formats)
                
                # Skip videos - Ollama only supports images
                if has_video_ext or has_video_format or ('/vid/' in url_lower and not has_image_ext):
                    if self.verbose:
                        print(f"⚠️  Skipping video: {url}")
                    continue
                
                # Accept images (including video thumbnails with image extensions)
                is_image = (has_image_ext or has_image_format or 
                           '/img/' in url_lower or 'image' in url_lower or 'thumb' in url_lower)
                
                if not is_image:
                    if self.verbose:
                        print(f"⚠️  Skipping unsupported format: {url}")
                    continue
                
                if self.verbose:
                    print(f"📥 Downloading image: {url}")
                
                # Download with timeout
                response = requests.get(url, timeout=self.DEFAULT_MEDIA_TIMEOUT, stream=True)
                response.raise_for_status()
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.DEFAULT_MAX_MEDIA_SIZE:
                    if self.verbose:
                        print(f"⚠️  Skipping large file ({content_length} bytes)")
                    continue
                
                # Convert to base64
                image_data = base64.b64encode(response.content).decode('utf-8')
                
                media_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_data}
                })
                
                if self.verbose:
                    print(f"✅ Downloaded {len(response.content)} bytes")
                
            except requests.exceptions.Timeout:
                if self.verbose:
                    print(f"⏰ Timeout downloading {url}")
                continue
            except requests.exceptions.RequestException as e:
                if self.verbose:
                    print(f"⚠️  Failed to download {url}: {e}")
                continue
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Unexpected error with {url}: {e}")
                continue
        
        if self.verbose and media_content:
            print(f"📦 Prepared {len(media_content)} media items")
        
        return media_content
    
    def _has_only_videos(self, media_urls: List[str]) -> bool:
        """
        Check if media URLs contain only videos (no images).
        
        Args:
            media_urls: List of media URLs
        
        Returns:
            True if all URLs are videos, False if any images found
        """
        if not media_urls:
            return False
        
        video_extensions = ['.mp4', '.m3u8', '.mov', '.avi', '.webm', '.m4v', '.flv', '.wmv']
        video_formats = ['format=mp4', 'format=m3u8', 'format=mov', 'format=avi', 
                        'format=webm', 'format=m4v', 'format=flv', 'format=wmv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg']
        image_formats = ['format=jpg', 'format=jpeg', 'format=png', 'format=gif', 
                        'format=webp', 'format=bmp', 'format=tiff', 'format=svg']
        
        has_videos = False
        has_images = False
        
        for url in media_urls:
            url_lower = url.lower()
            
            # Check for images (including video thumbnails with image extensions)
            if (any(ext in url_lower for ext in image_extensions) or 
                any(fmt in url_lower for fmt in image_formats)):
                has_images = True
            # Check for videos
            elif (any(ext in url_lower for ext in video_extensions) or 
                  any(fmt in url_lower for fmt in video_formats)):
                has_videos = True
        
        return has_videos and not has_images
    
    def _parse_category_and_explanation(self, response: str) -> Tuple[str, str]:
        """
        Parse structured response to extract category and explanation.
        
        Expected format:
        CATEGORÍA: category_name
        EXPLICACIÓN: explanation text
        
        Args:
            response: Raw model response
        
        Returns:
            Tuple of (category, explanation)
        """
        if not response or not response.strip():
            return Categories.GENERAL, "Error: Model returned empty response"
        
        # Extract category
        category_match = re.search(r'CATEGORÍA:\s*([^\n]+)', response, re.IGNORECASE)
        explanation_match = re.search(
            r'EXPLICACIÓN:\s*([^\n]+(?:\n(?!\s*CATEGORÍA:)[^\n]*)*)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        
        # Find matching category
        category = Categories.GENERAL
        if category_match:
            category_text = category_match.group(1).strip().lower()
            for cat in Categories.get_all_categories():
                if cat.lower() == category_text:
                    category = cat
                    break
        
        # Extract explanation
        explanation = explanation_match.group(1).strip() if explanation_match else response.strip()
        
        # Validate explanation
        if not explanation or len(explanation.strip()) < 10:
            explanation = f"Contenido clasificado como {category}."
        
        return category, explanation
