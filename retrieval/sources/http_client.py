"""
Shared HTTP client utility for all retrieval sources.
Provides standardized HTTP request handling, error handling, and retry logic.
"""

import requests
import time
import logging
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class HttpClient:
    """
    Shared HTTP client with standardized configuration, error handling, and retry logic.
    Used across all source modules to eliminate HTTP duplication.
    """

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        user_agent: str = "Mozilla/5.0 (compatible; EvidenceVerifier/1.0)",
        language: str = "es"
    ):
        """
        Initialize HTTP client with retry strategy.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff multiplier between retries (0.5 = 0.5s, 1s, 2s, 4s...)
            user_agent: User agent string for requests
            language: Accept-Language header value
        """
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Create session with retry strategy
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]  # Retry on these methods
        )

        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': f'{language};q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> requests.Response:
        """
        Perform GET request with error handling.

        Args:
            url: URL to request
            params: Query parameters
            headers: Additional headers (merged with defaults)
            timeout: Override default timeout

        Returns:
            Response object

        Raises:
            requests.RequestException: On request failure after retries
        """
        try:
            response = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout or self.timeout
            )
            response.raise_for_status()
            return response

        except requests.Timeout as e:
            self.logger.error(f"Request timeout for {url}: {e}")
            raise

        except requests.ConnectionError as e:
            self.logger.error(f"Connection error for {url}: {e}")
            raise

        except requests.HTTPError as e:
            self.logger.error(f"HTTP error for {url}: {e.response.status_code}")
            raise

        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            raise

    def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> requests.Response:
        """
        Perform POST request with error handling.

        Args:
            url: URL to request
            data: Form data
            json: JSON data
            headers: Additional headers (merged with defaults)
            timeout: Override default timeout

        Returns:
            Response object

        Raises:
            requests.RequestException: On request failure after retries
        """
        try:
            response = self.session.post(
                url,
                data=data,
                json=json,
                headers=headers,
                timeout=timeout or self.timeout
            )
            response.raise_for_status()
            return response

        except requests.Timeout as e:
            self.logger.error(f"Request timeout for {url}: {e}")
            raise

        except requests.ConnectionError as e:
            self.logger.error(f"Connection error for {url}: {e}")
            raise

        except requests.HTTPError as e:
            self.logger.error(f"HTTP error for {url}: {e.response.status_code}")
            raise

        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            raise

    def get_with_fallback(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> Optional[requests.Response]:
        """
        Perform GET request with fallback (returns None on failure instead of raising).

        Args:
            url: URL to request
            params: Query parameters
            headers: Additional headers
            timeout: Override default timeout

        Returns:
            Response object or None on failure
        """
        try:
            return self.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            self.logger.warning(f"Request failed for {url}, returning None: {e}")
            return None

    def close(self):
        """Close the HTTP session."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_http_client(
    timeout: int = 10,
    max_retries: int = 3,
    language: str = "es"
) -> HttpClient:
    """
    Factory function to create HTTP client with standard configuration.

    Args:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        language: Accept-Language header value

    Returns:
        Configured HTTP client
    """
    return HttpClient(
        timeout=timeout,
        max_retries=max_retries,
        language=language
    )
