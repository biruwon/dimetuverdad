"""
Gemini API client for the dimetuverdad analyzer.

Provides pure Gemini API client functionality with proper error handling
and media preparation.
"""

import os
import logging
from typing import Optional, Any, Tuple

# Simple warning suppression for Google Cloud libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import google.generativeai as genai

from .error_handler import classify_error, ErrorCategory


class GeminiClient:
    """
    Pure Gemini API client with proper error handling and media preparation.
    """

    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        """
        Initialize Gemini client with API key.

        Args:
            api_key: Gemini API key
            logger: Optional logger instance
        """
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)

    def get_model(self, model_name: str) -> Tuple[Optional[genai.GenerativeModel], Optional[str]]:
        """
        Get initialized Gemini model.

        Args:
            model_name: Name of the model to initialize

        Returns:
            Tuple of (model, error_message) where model is None if initialization failed
        """
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(model_name)
            return model, None
        except Exception as e:
            error = classify_error(e, f"client initialization ({model_name})")
            return None, str(error)

    def prepare_media_for_analysis(self, media_path: str, media_url: str) -> Optional[Any]:
        """
        Prepare media file for Gemini analysis.

        For the direct Gemini API, we return the file path or PIL Image directly
        instead of uploading to Vertex AI.

        Args:
            media_path: Path to the downloaded media file
            media_url: Original media URL (for logging)

        Returns:
            Prepared media object (PIL Image or file path) or None if preparation failed
        """
        try:
            self.logger.info("📤 Preparing media for Gemini...")

            # Check if file exists and get its size
            if not os.path.exists(media_path):
                self.logger.error(f"❌ Media file does not exist: {media_path}")
                return None

            file_size = os.path.getsize(media_path)
            if file_size == 0:
                self.logger.error("❌ Media file is empty")
                return None

            # For images, we can return PIL Image or file path directly
            if media_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
                try:
                    from PIL import Image
                    # Try to load as PIL Image for better compatibility
                    image = Image.open(media_path)
                    # Verify image can be loaded
                    image.verify()
                    image.close()
                    # Reopen for use
                    image = Image.open(media_path)
                    self.logger.info("✅ Image prepared successfully")
                    return image
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load as PIL Image, using file path: {e}")
                    # Fallback to file path
                    self.logger.info("✅ Media file prepared successfully")
                    return media_path

            # For videos and other files, return the file path
            # Gemini API can handle file paths directly for generate_content
            self.logger.info("✅ Media file prepared successfully")
            return media_path

        except Exception as e:
            error = classify_error(e, "media preparation")
            self.logger.error(f"❌ {error}")
            return None

    def generate_content(self, model: genai.GenerativeModel, content: Any, timeout: float = 60.0) -> Optional[str]:
        """
        Generate content using the Gemini model with timeout.

        Args:
            model: Initialized Gemini model
            content: Content to generate from (text, image, etc.)
            timeout: Timeout in seconds

        Returns:
            Generated content or None if failed
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        try:
            # Use ThreadPoolExecutor with timeout instead of signal-based timeout
            # to avoid "signal only works in main thread" error
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.generate_content, content)
                try:
                    response = future.result(timeout=timeout)
                    if response and response.text:
                        return response.text
                    else:
                        self.logger.warning("⚠️ Model returned empty response")
                        return None
                except FutureTimeoutError:
                    self.logger.warning("⏰ Model generation timed out")
                    future.cancel()
                    return None

        except Exception as e:
            error = classify_error(e, "content generation")
            self.logger.error(f"❌ {error}")
            
            # Re-raise quota errors so they can be caught by rate limiting logic
            if error.category == ErrorCategory.QUOTA_ERROR:
                raise e
            
            return None