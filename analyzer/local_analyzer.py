"""
Local Multimodal Analyzer supporting multiple Ollama models for text and media analysis.
Supports both text-only and multimodal analysis with Gemma models and gpt-oss.
"""

import base64
import requests
from typing import Tuple, Optional, List
import ollama
import asyncio
from .categories import Categories
from .prompts import EnhancedPromptGenerator

class OllamaEmptyResponseError(Exception):
    """Raised when Ollama returns an empty response."""
    pass

class OllamaRetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass

class LocalMultimodalAnalyzer:
    """
    Unified local multimodal analyzer using Gemma3:4b for all analysis.
    Supports both text-only and multimodal analysis with the same model.
    
    Uses gemma3:4b for all content types (text, images, videos fallback to text).
    """
    
    # Model capabilities
    TEXT_MODELS = "gemma3:4b"
    MULTIMODAL_MODELS = ["gemma3:4b", "gemma3:12b", "gemma3:27b-it-qat"]
    
    # Default generation parameters
    DEFAULT_TEMPERATURE_TEXT = 0.3
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TIMEOUT_TEXT = 30.0
    DEFAULT_TEMPERATURE_MULTIMODAL = 0.2
    DEFAULT_TOP_P_MULTIMODAL = 0.7
    DEFAULT_NUM_PREDICT_MULTIMODAL = 250
    DEFAULT_KEEP_ALIVE = "24h"
    DEFAULT_MEDIA_TIMEOUT = 5.0
    DEFAULT_MAX_MEDIA_SIZE = 10 * 1024 * 1024
    
    def __init__(self, verbose: bool = False):
        """
        Initialize local multimodal analyzer with Ollama.
        
        Args:
            verbose: Enable detailed logging
        """
        self.primary_model = self.TEXT_MODELS
        self.verbose = verbose
        
        self.ollama_client = ollama.AsyncClient()
        self.prompt_generator = EnhancedPromptGenerator()
        
        if self.verbose:
            print(f"🤖 LocalMultimodalAnalyzer initialized with primary model: {self.primary_model}")
    
    def _handle_error(self, operation: str, error: Exception, fallback_value=None) -> any:
        """
        Centralized error handling with consistent logging and fallback behavior.
        
        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            fallback_value: Value to return if operation fails (None = raise exception)
        
        Returns:
            Fallback value if provided, otherwise raises RuntimeError
        """
        error_msg = f"{operation} failed: {str(error)}"
        
        if self.verbose:
            print(f"❌ {error_msg}")
        
        if fallback_value is not None:
            return fallback_value
        else:
            raise RuntimeError(error_msg)
    
    async def _retry_ollama_call(self, operation: str, call_func, *args, **kwargs) -> str:
        """
        Retry Ollama API calls with exponential backoff.
        
        Args:
            operation: Description of the operation for logging
            call_func: Async function to call (ollama_client.generate)
            *args, **kwargs: Arguments to pass to call_func
        
        Returns:
            Generated text response
        
        Raises:
            OllamaRetryError: When all retry attempts are exhausted
        """
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                if self.verbose and attempt > 0:
                    print(f"   🔄 Retry attempt {attempt + 1}/{max_retries} for {operation}")
                
                # Make the Ollama API call
                response = await call_func(*args, **kwargs)
                
                # Extract and validate response
                generated_text = response.get("response", "").strip()
                if not generated_text:
                    raise OllamaEmptyResponseError("Model returned empty response")
                
                return generated_text
                
            except OllamaEmptyResponseError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    if self.verbose:
                        print(f"   ⚠️  {operation} returned empty response, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise OllamaRetryError(f"{operation} returned empty response after {max_retries} attempts")
                    
            except Exception as e:
                # Check if it's a retryable Ollama error (network issues, server errors, etc.)
                if self._is_retryable_ollama_error(e):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        if self.verbose:
                            print(f"   ⚠️  {operation} failed with retryable error: {str(e)}, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise OllamaRetryError(f"{operation} failed after {max_retries} attempts: {str(e)}")
                else:
                    # Non-retryable error (model not found, invalid parameters, etc.)
                    raise e
    
    def _is_retryable_ollama_error(self, error: Exception) -> bool:
        """
        Determine if an Ollama error is retryable.
        
        Args:
            error: The exception that occurred
        
        Returns:
            True if the error is retryable (network, server, or temporary issues)
        """
        if isinstance(error, OllamaEmptyResponseError):
            return True
            
        # Check for Ollama ResponseError with retryable status codes
        if hasattr(error, 'status_code'):
            # 5xx server errors are retryable
            if 500 <= error.status_code < 600:
                return True
            # 429 Too Many Requests is retryable
            if error.status_code == 429:
                return True
            # 408 Request Timeout is retryable
            if error.status_code == 408:
                return True
        
        # Network-related errors are retryable
        error_msg = str(error).lower()
        retryable_patterns = [
            'connection', 'timeout', 'network', 'server', 'temporary',
            'unavailable', 'overload', 'busy', 'rate limit'
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    async def categorize_and_explain(self, content: str, media_urls: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Single LLM call for both category detection and explanation.
        Uses multimodal analysis if media is provided, text-only otherwise.
        
        Args:
            content: Text content to analyze
            media_urls: Optional list of media URLs (images/videos)
        
        Returns:
            Tuple of (category, explanation)
        """
        if self.verbose:
            print(f"🔍 Running local {'multimodal' if media_urls else 'text'} categorization + explanation")
            print(f"📝 Content: {content[:100]}...")
            if media_urls:
                print(f"🖼️  Media URLs: {len(media_urls)} items")
        
        # Check if content contains only videos (local analyzer can't process videos)
        if media_urls and self._has_only_videos(media_urls):
            if self.verbose:
                print("🎥 Content contains only videos - analyzing text only")
            # For video-only content, analyze text without media
            media_urls = None
        
        try:
            # Use unified analysis method
            response = await self._analyze_content(content, media_urls)
            
            # Parse structured response
            category, explanation = self._parse_category_and_explanation(response)
            
            if self.verbose:
                print(f"✅ Category detected: {category}")
                print(f"💭 Explanation: {explanation[:100]}...")
            
            return category, explanation
            
        except Exception as e:
            # Re-raise all errors to stop the analysis pipeline
            raise RuntimeError(f"Analysis failed: {str(e)}") from e
    
    async def explain_only(self, content: str, category: str, media_urls: Optional[List[str]] = None) -> str:
        """
        Generate explanation for known category.
        Uses multimodal analysis if media is provided.
        
        Args:
            content: Content to explain
            category: Already-detected category
            media_urls: Optional list of media URLs
        
        Returns:
            Explanation (Spanish, 2-3 sentences)
        """
        if self.verbose:
            print(f"🔍 Generating local {'multimodal' if media_urls else 'text'} explanation for category: {category}")
            print(f"📝 Content: {content[:100]}...")
            if media_urls:
                print(f"🖼️  Media URLs: {len(media_urls)} items")
        
        # Check if content contains only videos (local analyzer can't process videos)
        if media_urls and self._has_only_videos(media_urls):
            if self.verbose:
                print("🎥 Content contains only videos - analyzing text only")
            # For video-only content, analyze text without media
            media_urls = None
        
        try:
            # Use unified analysis method for explanation
            explanation = await self._analyze_content(content, media_urls, category)
            
            if self.verbose:
                print(f"💭 Explanation generated: {explanation[:100]}...")
            
            return explanation
            
        except Exception as e:
            # Re-raise all errors to stop the analysis pipeline
            raise RuntimeError(f"Explanation generation failed: {str(e)}") from e
    
    async def _analyze_content(self, content: str, media_urls: Optional[List[str]] = None, category: Optional[str] = None) -> str:
        """
        Unified method for analyzing content (text-only or multimodal, categorization or explanation).

        Args:
            content: Text content to analyze
            media_urls: Optional media URLs for multimodal analysis
            category: Optional known category for explanation-only mode

        Returns:
            Analysis response (category + explanation or explanation only)
        """
        # Determine if this is multimodal analysis
        is_multimodal = media_urls is not None and len(media_urls) > 0
        is_explanation_only = category is not None

        # Build appropriate prompt
        if is_explanation_only:
            if is_multimodal:
                prompt = self.prompt_generator.build_multimodal_explanation_prompt(content, category)
            else:
                prompt = self._build_explanation_prompt(content, category)
        else:
            if is_multimodal:
                prompt = self.prompt_generator.build_multimodal_categorization_prompt(content)
            else:
                prompt = self.prompt_generator.build_ollama_categorization_prompt(content)

        # Select model and analysis type
        if is_multimodal:
            media_content = await self._prepare_media_content(media_urls)
            if media_content:  # Only do multimodal if we have valid media content
                model_to_use = self._select_multimodal_model()
                return await self._generate_multimodal_with_ollama(prompt, media_content, model_to_use)
            else:
                # Fall back to text-only if no valid media content
                if self.verbose:
                    print("⚠️  No valid media content found, falling back to text-only analysis")
                is_multimodal = False
        
        # Text-only analysis
        model_to_use = self.TEXT_MODELS
        return await self._generate_with_ollama(prompt, model_to_use)
    
    def _select_multimodal_model(self) -> str:
        """
        Select the multimodal model to use.
        Now uses the same TEXT_MODELS for consistency.
        """
        return self.TEXT_MODELS  # gemma3:4b
    
    async def _prepare_media_content(self, media_urls: List[str]) -> List[dict]:
        """
        Prepare media content for multimodal analysis.
        Downloads images only (skips videos) and converts to base64 for Ollama.
        Ollama multimodal models only support images, not videos.
        """
        media_content = []
        
        for url in media_urls[:3]:  # Limit to 3 media files for performance
            try:
                # Skip videos - Ollama multimodal models only support images
                if any(ext in url.lower() for ext in ['.mp4', '.m3u8', '.mov', '.avi', '.webm', 'video']):
                    if self.verbose:
                        print(f"⚠️  Skipping video file (Ollama only supports images): {url}")
                    continue
                
                # Only process images
                if not any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                    if self.verbose:
                        print(f"⚠️  Skipping unsupported media format: {url}")
                    continue
                
                if self.verbose:
                    print(f"📥 Downloading image: {url}")
                
                # Download image with timeout and retry
                response = requests.get(url, timeout=self.DEFAULT_MEDIA_TIMEOUT, stream=True)
                response.raise_for_status()
                
                # Check content length to avoid downloading very large files
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.DEFAULT_MAX_MEDIA_SIZE:
                    if self.verbose:
                        print(f"⚠️  Skipping large file ({content_length} bytes): {url}")
                    continue
                
                # Convert to base64
                image_data = base64.b64encode(response.content).decode('utf-8')
                
                media_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data  # Just the base64 string, not data URL
                    }
                })
                
                if self.verbose:
                    print(f"✅ Downloaded {len(image_data)} bytes")
                
            except requests.exceptions.Timeout:
                if self.verbose:
                    print(f"⏰ Timeout downloading media {url}")
                continue
            except requests.exceptions.RequestException as e:
                if self.verbose:
                    print(f"⚠️  Failed to download media {url}: {e}")
                continue
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Unexpected error downloading {url}: {e}")
                continue
        
        if self.verbose and media_content:
            print(f"📦 Prepared {len(media_content)} media items for analysis")
        
        return media_content
    
    def _build_explanation_prompt(self, content: str, category: str) -> str:
        """
        Build prompt for text-only explanation.
        """
        # Use existing prompt generator for category-specific prompts
        return self.prompt_generator.generate_explanation_prompt(
            content,
            category,
            model_type="ollama"
        )
    
    async def _generate_with_ollama(self, prompt: str, model: str) -> str:
        """
        Generate response using Ollama API (text-only).
        
        Args:
            prompt: Prompt to send to the model
            model: Model to use
        
        Returns:
            Generated text response
        """
        try:
            # Use the retry utility for robust API calls
            return await self._retry_ollama_call(
                "Ollama text generation",
                self.ollama_client.generate,
                model=model,
                prompt=prompt,
                system=EnhancedPromptGenerator.build_ollama_text_analysis_system_prompt(),
                options={
                    "temperature": self.DEFAULT_TEMPERATURE_TEXT,
                    "num_predict": self.DEFAULT_MAX_TOKENS,
                }
            )
            
        except OllamaRetryError as e:
            return self._handle_error("Ollama text generation", e)
    
    async def _generate_multimodal_with_ollama(self, content: str, media_content: List[dict], model: str) -> str:
        """
        Generate multimodal response using native Ollama API.
        
        Args:
            content: Text content to analyze
            media_content: List of media content dicts with base64 data
            model: Multimodal model to use
        
        Returns:
            Generated response
        """
        try:
            # Extract base64 images from media_content
            images = []
            for media in media_content:
                if media.get("type") == "image_url" and media.get("image_url", {}).get("url"):
                    # The URL field now contains just the base64 string
                    images.append(media["image_url"]["url"])
            
            if not images:
                raise RuntimeError("No valid images found in media content")
            
            # Build combined prompt with system instructions
            system_prompt = self.prompt_generator.build_ollama_multimodal_system_prompt()
            
            # Build multimodal-specific prompt that instructs analysis of both text and images
            prompt = self.prompt_generator.build_multimodal_categorization_prompt(content)

            # Use the retry utility for robust multimodal API calls
            return await self._retry_ollama_call(
                "Ollama multimodal generation",
                self.ollama_client.generate,
                model=model,
                prompt=prompt,
                system=system_prompt,
                images=images,
                options={
                    "temperature": self.DEFAULT_TEMPERATURE_MULTIMODAL,
                    "top_p": self.DEFAULT_TOP_P_MULTIMODAL,
                    "num_predict": self.DEFAULT_NUM_PREDICT_MULTIMODAL,
                },
                keep_alive=self.DEFAULT_KEEP_ALIVE
            )
            
        except OllamaRetryError as e:
            return self._handle_error("Multimodal generation", e)
    
    def _has_only_videos(self, media_urls: List[str]) -> bool:
        """
        Check if media URLs contain only videos (no processable images).
        Local multimodal models only support images, not videos.
        
        Args:
            media_urls: List of media URLs
            
        Returns:
            True if all URLs are videos (no images found), False otherwise
        """
        if not media_urls:
            return False
            
        # Check for video extensions and formats
        video_extensions = ['.mp4', '.m3u8', '.mov', '.avi', '.webm', '.m4v', '.flv', '.wmv']
        video_formats = ['format=mp4', 'format=m3u8', 'format=mov', 'format=avi', 'format=webm', 'format=m4v', 'format=flv', 'format=wmv']
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg']
        image_formats = ['format=jpg', 'format=jpeg', 'format=png', 'format=gif', 'format=webp', 'format=bmp', 'format=tiff', 'format=svg']
        
        has_videos = False
        has_images = False
        
        for url in media_urls:
            url_lower = url.lower()
            # Check if it's a video (extensions or format parameters)
            if (any(ext in url_lower for ext in video_extensions) or 
                any(fmt in url_lower for fmt in video_formats) or 
                'video' in url_lower):
                has_videos = True
            # Check if it's an image (extensions or format parameters)
            elif (any(ext in url_lower for ext in image_extensions) or 
                  any(fmt in url_lower for fmt in image_formats) or 
                  'image' in url_lower):
                has_images = True
        
        # Return True only if we have videos but no images
        return has_videos and not has_images
    
    def _parse_category_and_explanation(self, response: str) -> Tuple[str, str]:
        """
        Parse structured response from categorization prompt.

        Expected format:
        CATEGORÍA: category_name
        EXPLICACIÓN: explanation text

        Returns:
            Tuple of (category, explanation)
        """
        import re

        if not response or not response.strip():
            return Categories.GENERAL, "Error: Model returned empty response"

        # Use regex to extract category and explanation
        category_match = re.search(r'CATEGORÍA:\s*([^\n]+)', response, re.IGNORECASE)
        explanation_match = re.search(r'EXPLICACIÓN:\s*([^\n]+(?:\n(?!\s*CATEGORÍA:)[^\n]*)*)', response, re.IGNORECASE | re.DOTALL)

        # Extract category
        category = Categories.GENERAL
        if category_match:
            category_text = category_match.group(1).strip().lower()
            # Find exact case match
            for cat in Categories.get_all_categories():
                if cat.lower() == category_text:
                    category = cat
                    break

        # Extract explanation
        explanation = explanation_match.group(1).strip() if explanation_match else response.strip()

        # Validate explanation is not empty
        if not explanation or len(explanation.strip()) < 10:
            explanation = f"Contenido clasificado como {category}."

        return category, explanation
