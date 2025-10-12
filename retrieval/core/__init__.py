"""
Core components of the retrieval system.
"""

from .claim_extractor import ClaimExtractor, Claim
from .evidence_aggregator import EvidenceAggregator
from .query_builder import QueryBuilder
from .models import VerificationResult, EvidenceSource, VerificationVerdict

__all__ = [
    "ClaimExtractor",
    "Claim",
    "EvidenceAggregator",
    "QueryBuilder",
    "VerificationResult",
    "EvidenceSource",
    "VerificationVerdict",
]