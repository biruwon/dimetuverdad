"""
Tests for OllamaClient - low-level Ollama API client with retry and timeout logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from analyzer.ollama_client import OllamaClient, OllamaEmptyResponseError, OllamaRetryError


class TestOllamaClientInitialization:
    """Test OllamaClient initialization."""
    
    def test_default_initialization(self):
        """Test client initializes with default settings."""
        client = OllamaClient()
        assert client.client is not None
        assert client.verbose is False
    
    def test_verbose_initialization(self):
        """Test client initializes with verbose mode."""
        client = OllamaClient(verbose=True)
        assert client.verbose is True


class TestGenerateText:
    """Test text generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_text_success(self):
        """Test successful text generation."""
        client = OllamaClient()
        
        with patch.object(client.client, 'generate') as mock_generate:
            mock_generate.return_value = {"response": "Generated text response"}
            
            result = await client.generate_text(
                prompt="Test prompt",
                model="test-model"
            )
            
            assert result == "Generated text response"
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_text_with_options(self):
        """Test text generation with custom options."""
        client = OllamaClient()
        
        with patch.object(client.client, 'generate') as mock_generate:
            mock_generate.return_value = {"response": "Response with options"}
            
            result = await client.generate_text(
                prompt="Test prompt",
                model="test-model",
                system_prompt="System prompt",
                options={"temperature": 0.5},
                timeout=60.0
            )
            
            assert result == "Response with options"


class TestGenerateMultimodal:
    """Test multimodal generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_multimodal_success(self):
        """Test successful multimodal generation."""
        client = OllamaClient()
        
        with patch.object(client.client, 'generate') as mock_generate:
            mock_generate.return_value = {"response": "Multimodal response"}
            
            result = await client.generate_multimodal(
                prompt="Test prompt",
                images=["base64image"],
                model="test-model"
            )
            
            assert result == "Multimodal response"
            mock_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_multimodal_no_images_raises_error(self):
        """Test that multimodal generation requires images."""
        client = OllamaClient()
        
        with pytest.raises(ValueError, match="At least one image is required"):
            await client.generate_multimodal(
                prompt="Test prompt",
                images=[],
                model="test-model"
            )


class TestRetryLogic:
    """Test retry logic for Ollama API calls."""
    
    @pytest.mark.asyncio
    async def test_retry_on_empty_response(self):
        """Test retry when model returns empty response."""
        client = OllamaClient()
        
        with patch.object(client.client, 'generate') as mock_generate:
            # First call returns empty, second call succeeds
            mock_generate.side_effect = [
                {"response": ""},
                {"response": "Success on retry"}
            ]
            
            result = await client.generate_text(
                prompt="Test",
                model="test-model"
            )
            
            assert result == "Success on retry"
            assert mock_generate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_exhausted_on_empty_response(self):
        """Test that retries are exhausted after max attempts with empty responses."""
        client = OllamaClient()
        
        with patch.object(client.client, 'generate') as mock_generate:
            # Always return empty
            mock_generate.return_value = {"response": ""}
            
            with pytest.raises(OllamaRetryError, match="returned empty response after 2 attempts"):
                await client.generate_text(
                    prompt="Test",
                    model="test-model"
                )
            
            assert mock_generate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry when request times out."""
        client = OllamaClient()
        
        with patch.object(client.client, 'generate') as mock_generate:
            # First call times out (simulate by raising TimeoutError), second succeeds
            mock_generate.side_effect = [
                asyncio.TimeoutError("Simulated timeout"),
                {"response": "Success on retry"}
            ]
            
            result = await client.generate_text(
                prompt="Test",
                model="test-model",
                timeout=1.0  # Reasonable timeout
            )
            
            assert result == "Success on retry"
            assert mock_generate.call_count == 2
    
    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(self):
        """Test that non-retryable errors fail without retry."""
        client = OllamaClient()
        
        with patch.object(client.client, 'generate') as mock_generate:
            mock_generate.side_effect = ValueError("Invalid parameter")
            
            with pytest.raises(ValueError, match="Invalid parameter"):
                await client.generate_text(
                    prompt="Test",
                    model="test-model"
                )
            
            # Should not retry on non-retryable error
            assert mock_generate.call_count == 1


class TestErrorClassification:
    """Test error classification for retryable vs non-retryable errors."""
    
    def test_empty_response_error_is_retryable(self):
        """Test OllamaEmptyResponseError is classified as retryable."""
        client = OllamaClient()
        error = OllamaEmptyResponseError("Empty response")
        assert client._is_retryable_error(error) is True
    
    def test_network_errors_are_retryable(self):
        """Test network-related errors are retryable."""
        client = OllamaClient()
        
        retryable_errors = [
            Exception("connection timeout"),
            Exception("network unavailable"),
            Exception("server error"),
            Exception("temporary failure"),
        ]
        
        for error in retryable_errors:
            assert client._is_retryable_error(error) is True
    
    def test_http_5xx_errors_are_retryable(self):
        """Test HTTP 5xx errors are retryable."""
        client = OllamaClient()
        
        error = Mock()
        error.status_code = 503
        assert client._is_retryable_error(error) is True
        
        error.status_code = 500
        assert client._is_retryable_error(error) is True
    
    def test_rate_limit_errors_are_retryable(self):
        """Test rate limit errors are retryable."""
        client = OllamaClient()
        
        error = Mock()
        error.status_code = 429
        assert client._is_retryable_error(error) is True
    
    def test_non_retryable_errors(self):
        """Test that validation errors are not retryable."""
        client = OllamaClient()
        
        non_retryable_errors = [
            ValueError("Invalid input"),
            Exception("model not found"),
            Exception("invalid parameters"),
        ]
        
        for error in non_retryable_errors:
            assert client._is_retryable_error(error) is False
