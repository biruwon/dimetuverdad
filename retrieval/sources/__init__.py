"""
Data sources for evidence retrieval.
"""

from .statistical_apis import StatisticalAPIManager
from .http_client import HttpClient, create_http_client

__all__ = [
    "StatisticalAPIManager",
    "HttpClient",
    "create_http_client",
]