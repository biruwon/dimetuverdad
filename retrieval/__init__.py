"""
Retrieval system for evidence-based content verification.
Provides comprehensive fact-checking and source validation capabilities.
"""

from .api import RetrievalAPI, RetrievalConfig, VerificationRequest, RetrievalResult, create_retrieval_api
from .api import verify_text_content, verify_single_claim

__version__ = "1.0.0"
__all__ = [
    # Main API
    "RetrievalAPI",
    "create_retrieval_api",
    "verify_text_content",
    "verify_single_claim",

    # Data models
    "RetrievalConfig",
    "VerificationRequest",
    "RetrievalResult",
]