"""
Ollama API client with robust retry logic and timeout handling.
Low-level interface for interacting with Ollama models.
"""

import asyncio
from typing import List, Optional, Dict
import ollama
from .constants import ConfigDefaults


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
    
    # Timeout settings (reasonable for production use)
    DEFAULT_TEXT_TIMEOUT = 120.0  # 120 seconds for text generation
    DEFAULT_MULTIMODAL_TIMEOUT = 300.0  # 300 seconds (5 minutes) for multimodal (allows for image processing)
    
    # Retry settings
    MAX_RETRIES = 2
    BASE_RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Ollama client with proper timeout configuration.
        
        Args:
            verbose: Enable detailed logging
        """
        # Configure httpx timeout for the Ollama client
        # This ensures HTTP requests don't hang indefinitely
        import httpx
        timeout_config = httpx.Timeout(
            connect=30.0,  # 30s to establish connection
            read=300.0,    # 5 minutes to read response (for slow LLM generation)
            write=30.0,    # 30s to write request
            pool=10.0      # 10s to get connection from pool
        )
        
        # Initialize Ollama clients with timeout configuration
        self.client = ollama.AsyncClient(timeout=timeout_config)
        self.sync_client = ollama.Client(timeout=timeout_config)
        self.verbose = verbose
        
        if self.verbose:
            print("🔌 OllamaClient initialized with timeout protection")
    
    async def generate_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        options: Optional[Dict] = None,
        keep_alive: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        Generate text response from Ollama model.
        
        Args:
            prompt: User prompt
            model: Model name
            system_prompt: Optional system prompt
            options: Optional generation parameters (temperature, num_predict, etc.)
            keep_alive: How long to keep model loaded (e.g., "24h")
            timeout: Maximum wait time in seconds (defaults to DEFAULT_TEXT_TIMEOUT)
        
        Returns:
            Generated text response
        
        Raises:
            OllamaRetryError: When all retry attempts are exhausted
        """
        if timeout is None:
            timeout = self.DEFAULT_TEXT_TIMEOUT
        
        if self.verbose:
            print(f"🔄 Starting Ollama text generation (timeout: {timeout}s)")
            print(f"📝 Prompt length: {len(prompt)} chars")
            if system_prompt:
                print(f"📋 System prompt length: {len(system_prompt)} chars")
        
        return await self._retry_ollama_call(
            operation="Ollama text generation",
            call_func=self.client.generate,
            timeout=timeout,
            model=model,
            prompt=prompt,
            system=system_prompt,
            options=options or {},
            keep_alive=keep_alive
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
        import time
        import psutil
        import os
        import sys
        
        # Only print debug info if verbose is enabled
        if self.verbose:
            print(f"🐛 DEBUG: OllamaClient._retry_ollama_call - verbose={self.verbose}, operation={operation}", flush=True)
            sys.stdout.flush()
        
        for attempt in range(self.MAX_RETRIES):
            attempt_start = time.time()
            
            # Debug: Log system resources before attempt
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            system_mem = psutil.virtual_memory()
            
            if self.verbose:
                print(f"🔄 Attempt {attempt + 1}/{self.MAX_RETRIES} for {operation}", flush=True)
                print(f"   💾 Process RAM: {mem_info.rss / 1024 / 1024:.1f} MB", flush=True)
                print(f"   🖥️  System RAM: {system_mem.percent:.1f}% used ({system_mem.available / 1024 / 1024 / 1024:.1f} GB available)", flush=True)
                print(f"   ⚡ CPU: {cpu_percent:.1f}%", flush=True)
                
                # Log prompt details
                if 'prompt' in kwargs:
                    prompt_text = kwargs['prompt']
                    print(f"   📏 Prompt length: {len(prompt_text)} chars, {len(prompt_text.split())} words")
                    # Check for problematic patterns
                    if len(prompt_text) > 4000:
                        print(f"   ⚠️  LONG PROMPT detected ({len(prompt_text)} chars)")
                    if prompt_text.count('\n') > 50:
                        print(f"   ⚠️  Many newlines detected ({prompt_text.count('\\n')})")
                    
                if 'system' in kwargs and kwargs['system']:
                    system_text = kwargs['system']
                    print(f"   📋 System prompt: {len(system_text)} chars")
                    
                if attempt > 0:
                    print(f"   ⏱️  Starting retry attempt {attempt + 1} at {time.strftime('%H:%M:%S')}")
            
            try:
                # ALWAYS log operation start time for tracking
                operation_start_wall_time = time.strftime('%H:%M:%S')
                
                if self.verbose:
                    print(f"   📡 Calling Ollama API with {timeout}s timeout...")
                else:
                    # Even in non-verbose mode, log the start time for slow operations tracking
                    print(f"⏱️  {operation} starting at {operation_start_wall_time}", flush=True)
                
                # Make the Ollama API call with timeout
                api_call_start = time.time()
                
                # Create task with explicit cancellation on timeout
                task = asyncio.create_task(call_func(*args, **kwargs))
                
                try:
                    response = await asyncio.wait_for(task, timeout=timeout)
                except asyncio.TimeoutError:
                    # Explicitly cancel the task on timeout
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise asyncio.TimeoutError(f"{operation} timed out after {timeout}s")
                
                api_call_duration = time.time() - api_call_start
                
                # ALWAYS log operation completion with timing
                if not self.verbose:
                    if api_call_duration > 120:
                        print(f"⚠️  {operation} completed in {api_call_duration:.1f}s (SLOW, >120s threshold)", flush=True)
                    else:
                        print(f"✅ {operation} completed in {api_call_duration:.1f}s", flush=True)
                
                # Log detailed info for slow calls
                if api_call_duration > 120:
                    print(f"   ⚠️  SLOW API CALL DETAILS:", flush=True)
                    print(f"      Model: {kwargs.get('model', 'unknown')}", flush=True)
                    if 'prompt' in kwargs:
                        print(f"      Prompt length: {len(kwargs['prompt'])} chars", flush=True)
                    if 'images' in kwargs:
                        print(f"      Images: {len(kwargs['images'])} items", flush=True)
                
                if self.verbose:
                    print(f"   ✅ Ollama API call completed in {api_call_duration:.2f}s")
                    
                    # Debug: Log system resources after completion
                    mem_info_after = process.memory_info()
                    cpu_percent_after = process.cpu_percent(interval=0.1)
                    print(f"   💾 RAM after: {mem_info_after.rss / 1024 / 1024:.1f} MB (Δ {(mem_info_after.rss - mem_info.rss) / 1024 / 1024:.1f} MB)")
                    print(f"   ⚡ CPU after: {cpu_percent_after:.1f}%")
                
                # Extract and validate response
                response_extraction_start = time.time()
                generated_text = response.get("response", "").strip()
                
                if self.verbose:
                    print(f"   📄 Response extracted in {time.time() - response_extraction_start:.3f}s")
                    print(f"   📏 Response length: {len(generated_text)} chars")
                    
                    # Debug: Check response characteristics
                    if len(generated_text) > 1000:
                        print(f"   ⚠️  LONG RESPONSE: {len(generated_text)} chars")
                    
                    # Log if API call was unusually slow
                    if api_call_duration > 30:
                        print(f"   ⚠️  SLOW API CALL: {api_call_duration:.1f}s (threshold: 30s)")
                        print(f"   🔍 Consider investigating this specific content pattern")
                
                if not generated_text:
                    raise OllamaEmptyResponseError("Model returned empty response")
                
                total_attempt_duration = time.time() - attempt_start
                if self.verbose:
                    print(f"   🎯 Attempt {attempt + 1} succeeded in {total_attempt_duration:.2f}s")
                
                return generated_text
                
            except asyncio.TimeoutError:
                attempt_duration = time.time() - attempt_start
                error_msg = f"{operation} timed out after {timeout}s (attempt took {attempt_duration:.2f}s)"
                if self.verbose:
                    print(f"⏰ {error_msg}")
                
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                    if self.verbose:
                        print(f"⏳ Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise OllamaRetryError(f"{error_msg} after {self.MAX_RETRIES} attempts")
                
            except OllamaEmptyResponseError as e:
                attempt_duration = time.time() - attempt_start
                if self.verbose:
                    print(f"📭 Empty response in {attempt_duration:.2f}s")
                
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                    if self.verbose:
                        print(f"⏳ Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise OllamaRetryError(f"{operation} returned empty response after {self.MAX_RETRIES} attempts")
                    
            except Exception as e:
                attempt_duration = time.time() - attempt_start
                if self.verbose:
                    print(f"💥 Exception in attempt {attempt + 1} after {attempt_duration:.2f}s: {type(e).__name__}: {str(e)}")
                
                # Check if it's a retryable error
                if self._is_retryable_error(e):
                    if attempt < self.MAX_RETRIES - 1:
                        delay = self.BASE_RETRY_DELAY * (2 ** attempt)
                        if self.verbose:
                            print(f"🔄 Retryable error, waiting {delay}s...")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise OllamaRetryError(f"{operation} failed after {self.MAX_RETRIES} attempts: {str(e)}")
                else:
                    # Non-retryable error - fail immediately
                    if self.verbose:
                        print(f"❌ Non-retryable error: {type(e).__name__}: {str(e)}")
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
    
    async def reset_model_context(self, model: str, keep_alive: str = "5m", timeout: float = ConfigDefaults.CONTEXT_RESET_TIMEOUT):
        """
        Reset model context by unloading and reloading the model.
        This fully clears accumulated context to prevent slowdowns.
        
        Args:
            model: Model name to reset
            keep_alive: How long to keep model loaded after reset (default: 5m for batch processing)
            timeout: Maximum time to wait for context reset (default: from ConfigDefaults.CONTEXT_RESET_TIMEOUT)
        """
        if self.verbose:
            print(f"🔄 Fully unloading and reloading model: {model}")
        
        try:
            # First, unload the model completely by setting keep_alive to 0
            await asyncio.wait_for(
                self.client.generate(
                    model=model,
                    prompt=".",
                    options={"num_predict": 1},
                    keep_alive="0"  # Unload immediately
                ),
                timeout=timeout
            )
            
            # Wait a moment for unload to complete
            await asyncio.sleep(1)
            
            # Then reload with a fresh context
            await asyncio.wait_for(
                self.client.generate(
                    model=model,
                    prompt="Reset",
                    options={"num_predict": 1},
                    keep_alive=keep_alive  # Keep loaded for next batch of analyses (5 minutes default)
                ),
                timeout=timeout
            )
            
            if self.verbose:
                print(f"✅ Model fully reset and reloaded (keep_alive: {keep_alive})")
        except asyncio.TimeoutError:
            if self.verbose:
                print(f"⚠️  Model reset timed out after {timeout}s - continuing anyway")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Model reset failed: {e}")
