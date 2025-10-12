"""
Data structures and models for evidence verification results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from enum import Enum


class VerificationVerdict(Enum):
    """Possible verdicts for claim verification."""
    VERIFIED = "verified"      # Claim is confirmed as true
    QUESTIONABLE = "questionable"  # Claim needs more verification
    DEBUNKED = "debunked"      # Claim is confirmed as false
    UNVERIFIED = "unverified"  # Could not verify the claim


@dataclass
class EvidenceSource:
    """A source that provides evidence for or against a claim."""
    source_name: str
    source_type: str  # 'fact_checker', 'statistical_agency', 'news_outlet', etc.
    url: str
    title: str
    credibility_score: float  # 0.0 to 1.0
    publication_date: Optional[datetime] = None
    content_snippet: str = ""
    verdict_contribution: VerificationVerdict = VerificationVerdict.UNVERIFIED
    confidence: float = 0.5  # How confident this source is about its verdict


@dataclass
class VerificationResult:
    """Result of verifying a specific claim."""
    claim: str
    verdict: VerificationVerdict
    confidence: float  # Overall confidence in the verdict (0.0-1.0)
    evidence_sources: List[EvidenceSource] = field(default_factory=list)
    explanation: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    processing_time_seconds: float = 0.0
    claim_type: str = "unknown"  # From ClaimType enum
    extracted_value: Optional[str] = None

    def add_evidence_source(self, source: EvidenceSource):
        """Add an evidence source to this result."""
        self.evidence_sources.append(source)
        self._update_verdict()

    def _update_verdict(self):
        """Update overall verdict based on evidence sources."""
        if not self.evidence_sources:
            self.verdict = VerificationVerdict.UNVERIFIED
            self.confidence = 0.0
            return

        # Weight sources by credibility and confidence
        weighted_verdicts = []
        total_weight = 0.0

        for source in self.evidence_sources:
            weight = source.credibility_score * source.confidence
            total_weight += weight
            weighted_verdicts.append((source.verdict_contribution, weight))

        if total_weight == 0:
            self.verdict = VerificationVerdict.UNVERIFIED
            self.confidence = 0.0
            return

        # Calculate weighted verdict
        verdict_scores = {
            VerificationVerdict.VERIFIED: 0.0,
            VerificationVerdict.DEBUNKED: 0.0,
            VerificationVerdict.QUESTIONABLE: 0.0,
            VerificationVerdict.UNVERIFIED: 0.0
        }

        for verdict, weight in weighted_verdicts:
            verdict_scores[verdict] += weight

        # Normalize scores
        for verdict in verdict_scores:
            verdict_scores[verdict] /= total_weight

        # Determine overall verdict
        max_verdict = max(verdict_scores.items(), key=lambda x: x[1])

        # Check for contradictory evidence
        verified_score = verdict_scores[VerificationVerdict.VERIFIED]
        debunked_score = verdict_scores[VerificationVerdict.DEBUNKED]

        if verified_score > 0.4 and debunked_score > 0.4:
            self.verdict = VerificationVerdict.QUESTIONABLE  # Changed from CONTRADICTORY
            self.confidence = min(verified_score, debunked_score)  # Confidence in the contradiction
        else:
            self.verdict = max_verdict[0]
            self.confidence = max_verdict[1]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'claim': self.claim,
            'verdict': self.verdict.value,
            'confidence': self.confidence,
            'evidence_sources': [
                {
                    'source_name': src.source_name,
                    'source_type': src.source_type,
                    'url': src.url,
                    'title': src.title,
                    'credibility_score': src.credibility_score,
                    'publication_date': src.publication_date.isoformat() if src.publication_date else None,
                    'content_snippet': src.content_snippet,
                    'verdict_contribution': src.verdict_contribution.value,
                    'confidence': src.confidence
                }
                for src in self.evidence_sources
            ],
            'explanation': self.explanation,
            'last_updated': self.last_updated.isoformat(),
            'processing_time_seconds': self.processing_time_seconds,
            'claim_type': self.claim_type,
            'extracted_value': self.extracted_value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VerificationResult':
        """Create from dictionary."""
        result = cls(
            claim=data['claim'],
            verdict=VerificationVerdict(data['verdict']),
            confidence=data['confidence'],
            explanation=data.get('explanation', ''),
            last_updated=datetime.fromisoformat(data['last_updated']),
            processing_time_seconds=data.get('processing_time_seconds', 0.0),
            claim_type=data.get('claim_type', 'unknown'),
            extracted_value=data.get('extracted_value')
        )

        for src_data in data.get('evidence_sources', []):
            source = EvidenceSource(
                source_name=src_data['source_name'],
                source_type=src_data['source_type'],
                url=src_data['url'],
                title=src_data['title'],
                credibility_score=src_data['credibility_score'],
                publication_date=datetime.fromisoformat(src_data['publication_date']) if src_data.get('publication_date') else None,
                content_snippet=src_data.get('content_snippet', ''),
                verdict_contribution=VerificationVerdict(src_data['verdict_contribution']),
                confidence=src_data.get('confidence', 0.5)
            )
            result.evidence_sources.append(source)

        return result


@dataclass
class VerificationRequest:
    """A request to verify multiple claims from a piece of content."""
    content_id: str  # ID of the content being verified
    content_text: str
    claims_to_verify: List[str] = field(default_factory=list)
    request_timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 5  # 1-10, higher = more urgent
    context_metadata: Dict = field(default_factory=dict)  # Additional context

    def add_claim(self, claim: str):
        """Add a claim to verify."""
        if claim not in self.claims_to_verify:
            self.claims_to_verify.append(claim)


@dataclass
class VerificationBatch:
    """A batch of verification requests for processing."""
    batch_id: str
    requests: List[VerificationRequest] = field(default_factory=list)
    created_timestamp: datetime = field(default_factory=datetime.now)
    status: Literal['pending', 'processing', 'completed', 'failed'] = 'pending'

    def add_request(self, request: VerificationRequest):
        """Add a verification request to the batch."""
        self.requests.append(request)

    def get_total_claims(self) -> int:
        """Get total number of claims across all requests."""
        return sum(len(req.claims_to_verify) for req in self.requests)