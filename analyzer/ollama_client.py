"""
Ollama API client with robust retry logic and timeout handling.
Low-level interface for interacting with Ollama models.
"""

import asyncio
from typing import List, Optional, Dict
import ollama


class OllamaEmptyResponseError(Exception):
    """Raised when Ollama returns an empty response."""
    pass


class OllamaRetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


class OllamaClient:
    """
    Low-level client for Ollama API with retry and timeout handling.
    Provides robust text and multimodal generation with automatic error recovery.
    """
    
    # Default timeout settings
    DEFAULT_TEXT_TIMEOUT = 120.0  # 2 minutes for text generation
    DEFAULT_MULTIMODAL_TIMEOUT = 240.0  # 4 minutes for multimodal generation
    
    # Retry settings
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Ollama client.
        
        Args:
            verbose: Enable detailed logging
        """
        self.client = ollama.AsyncClient()
        self.verbose = verbose
        
        if self.verbose:
            print("🔌 OllamaClient initialized")
    
    async def generate_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        Generate text response from Ollama model.
        
        Args:
            prompt: User prompt
            model: Model name
            system_prompt: Optional system prompt
            options: Optional generation parameters (temperature, num_predict, etc.)
            timeout: Maximum wait time in seconds (defaults to DEFAULT_TEXT_TIMEOUT)
        
        Returns:
            Generated text response
        
        Raises:
            OllamaRetryError: When all retry attempts are exhausted
        """
        if timeout is None:
            timeout = self.DEFAULT_TEXT_TIMEOUT
        
        return await self._retry_ollama_call(
            operation="Ollama text generation",
            call_func=self.client.generate,
            timeout=timeout,
            model=model,
            prompt=prompt,
            system=system_prompt,
            options=options or {}
        )
    
    async def generate_multimodal(
        self,
        prompt: str,
        images: List[str],
        model: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict] = None,
        keep_alive: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        Generate multimodal response from Ollama model with images.
        
        Args:
            prompt: User prompt
            images: List of base64-encoded images
            model: Model name (must support multimodal)
            system_prompt: Optional system prompt
            options: Optional generation parameters
            keep_alive: How long to keep model loaded (e.g., "24h")
            timeout: Maximum wait time in seconds (defaults to DEFAULT_MULTIMODAL_TIMEOUT)
        
        Returns:
            Generated text response
        
        Raises:
            OllamaRetryError: When all retry attempts are exhausted
        """
        if timeout is None:
            timeout = self.DEFAULT_MULTIMODAL_TIMEOUT
        
        if not images:
            raise ValueError("At least one image is required for multimodal generation")
        
        return await self._retry_ollama_call(
            operation="Ollama multimodal generation",
            call_func=self.client.generate,
            timeout=timeout,
            model=model,
            prompt=prompt,
            system=system_prompt,
            images=images,
            options=options or {},
            keep_alive=keep_alive
        )
    
    async def _retry_ollama_call(
        self,
        operation: str,
        call_func,
        timeout: float,
        *args,
        **kwargs
    ) -> str:
        """
        Execute Ollama API call with retry logic and timeout.
        
        Args:
            operation: Description of operation for logging
            call_func: Async function to call
            timeout: Maximum wait time
            *args, **kwargs: Arguments for call_func
        
        Returns:
            Generated text response
        
        Raises:
            OllamaRetryError: When all retries are exhausted or non-retryable error occurs
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.verbose and attempt > 0:
                    print(f"   🔄 Retry attempt {attempt + 1}/{self.MAX_RETRIES} for {operation}")
                
                # Make the Ollama API call with timeout
                response = await asyncio.wait_for(
                    call_func(*args, **kwargs),
                    timeout=timeout
                )
                
                # Extract and validate response
                generated_text = response.get("response", "").strip()
                if not generated_text:
                    raise OllamaEmptyResponseError("Model returned empty response")
                
                return generated_text
                
            except asyncio.TimeoutError:
                error_msg = f"{operation} timed out after {timeout}s"
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                    if self.verbose:
                        print(f"   ⏱️  {error_msg}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise OllamaRetryError(f"{error_msg} after {self.MAX_RETRIES} attempts")
                
            except OllamaEmptyResponseError as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                    if self.verbose:
                        print(f"   ⚠️  {operation} returned empty response, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise OllamaRetryError(f"{operation} returned empty response after {self.MAX_RETRIES} attempts")
                    
            except Exception as e:
                # Check if it's a retryable error
                if self._is_retryable_error(e):
                    if attempt < self.MAX_RETRIES - 1:
                        delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                        if self.verbose:
                            print(f"   ⚠️  {operation} failed with retryable error: {str(e)}, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise OllamaRetryError(f"{operation} failed after {self.MAX_RETRIES} attempts: {str(e)}")
                else:
                    # Non-retryable error - fail immediately
                    raise e
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: The exception that occurred
        
        Returns:
            True if the error is retryable (network, server, temporary issues)
        """
        if isinstance(error, OllamaEmptyResponseError):
            return True
        
        # Check for HTTP status codes indicating retryable errors
        if hasattr(error, 'status_code'):
            status_code = error.status_code
            # 5xx server errors are retryable
            if 500 <= status_code < 600:
                return True
            # 429 Too Many Requests is retryable
            if status_code == 429:
                return True
            # 408 Request Timeout is retryable
            if status_code == 408:
                return True
        
        # Check error message for retryable patterns
        error_msg = str(error).lower()
        retryable_patterns = [
            'connection', 'timeout', 'network', 'server', 'temporary',
            'unavailable', 'overload', 'busy', 'rate limit'
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
