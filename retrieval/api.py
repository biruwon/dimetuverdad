"""
Main retrieval API providing unified access to all verification capabilities.
This is the primary interface for integrating evidence verification into the analyzer.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from retrieval.core.models import VerificationResult, EvidenceSource, VerificationVerdict
from retrieval.core.claim_extractor import ClaimExtractor, Claim
from retrieval.core.evidence_aggregator import EvidenceAggregator
from retrieval.core.query_builder import QueryBuilder
from retrieval.verification.claim_verifier import ClaimVerifier, VerificationContext, VerificationReport
from retrieval.verification.credibility_scorer import CredibilityScorer
from .sources.web_scrapers import WebScraperManager
from retrieval.sources.statistical_apis import StatisticalAPIManager
from retrieval.sources.web_scrapers import WebScraperManager
from retrieval.integration.analyzer_hooks import AnalyzerHooks, AnalysisResult

# Import performance tracking utility
from utils.performance import start_tracking, stop_tracking, print_performance_summary


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval system."""
    max_parallel_requests: int = 4
    verification_timeout: float = 30.0  # seconds
    enable_statistical_apis: bool = True
    enable_web_search: bool = True  # Enable web scraping for additional verification
    default_language: str = "es"
    log_level: str = "INFO"


@dataclass
class VerificationRequest:
    """Request for content verification."""
    content: str
    content_category: str = "general"
    language: str = "es"
    priority_level: str = "balanced"  # fast, balanced, quality
    user_context: Optional[str] = None
    enable_temporal_verification: bool = True
    enable_statistical_verification: bool = True


@dataclass
class RetrievalResult:
    """Comprehensive result from the retrieval system."""
    success: bool
    verification_report: Optional[VerificationReport]
    claims_extracted: List[Claim]
    processing_time: float
    error_message: Optional[str] = None


class RetrievalAPI:
    """
    Main API for the evidence retrieval and verification system.
    Provides unified access to all verification capabilities.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level))

        # Initialize core components
        self.claim_extractor = ClaimExtractor()
        self.evidence_aggregator = EvidenceAggregator()
        self.query_builder = QueryBuilder()
        self.credibility_scorer = CredibilityScorer()
        self.web_scraper_manager = WebScraperManager() if self.config.enable_web_search else None

        # Initialize optional components
        self.statistical_api = None
        if self.config.enable_statistical_apis:
            self.statistical_api = StatisticalAPIManager()

        self.web_scraper = None
        if self.config.enable_web_search:  # We'll use this flag for web scraping
            self.web_scraper = WebScraperManager()

        # Initialize main verifier
        self.multi_source_verifier = ClaimVerifier(
            max_workers=self.config.max_parallel_requests
        )

        # Initialize integration hooks
        self.analyzer_hooks = AnalyzerHooks(verifier=self.multi_source_verifier)

    async def verify_content(self, request: VerificationRequest) -> RetrievalResult:
        """
        Verify content using the complete retrieval pipeline.

        Args:
            request: Verification request with content and parameters

        Returns:
            Comprehensive verification result
        """
        # Start performance tracking
        tracker = start_tracking("Content Verification")

        start_time = time.time()

        try:
            # Extract claims first
            claims = self.claim_extractor.extract_claims(request.content)

            if not claims:
                # Increment operations counter for claims processed
                tracker.increment_operations(len(request.content.split()) // 10)  # Rough estimate of content units
                
                metrics = stop_tracking(tracker)
                print_performance_summary(metrics)
                
                return RetrievalResult(
                    success=True,
                    verification_report=None,
                    claims_extracted=[],
                    processing_time=time.time() - start_time
                )

            # Create verification context
            context = VerificationContext(
                original_text=request.content,
                content_category=request.content_category,
                user_context=request.user_context,
                language=request.language,
                priority_level=request.priority_level
            )

            # Perform verification with timeout
            try:
                verification_report = await asyncio.wait_for(
                    self.multi_source_verifier.verify_content(context),
                    timeout=self.config.verification_timeout
                )
            except asyncio.TimeoutError:
                # Increment operations counter for timeout
                tracker.increment_operations(1)
                
                metrics = stop_tracking(tracker)
                metrics.success = False
                metrics.error_message = f"Verification timeout after {self.config.verification_timeout}s"
                print_performance_summary(metrics)
                
                return RetrievalResult(
                    success=False,
                    verification_report=None,
                    claims_extracted=claims,
                    processing_time=time.time() - start_time,
                    error_message=f"Verification timeout after {self.config.verification_timeout}s"
                )

            processing_time = time.time() - start_time

            # Increment operations counter for successful verification
            tracker.increment_operations(len(claims))

            metrics = stop_tracking(tracker)
            print_performance_summary(metrics)

            return RetrievalResult(
                success=True,
                verification_report=verification_report,
                claims_extracted=claims,
                processing_time=processing_time
            )

        except Exception as e:
            self.logger.error(f"Content verification failed: {e}")
            
            # Increment operations counter for failed verification
            tracker.increment_operations(1)
            
            metrics = stop_tracking(tracker)
            metrics.success = False
            metrics.error_message = str(e)
            print_performance_summary(metrics)
            
            return RetrievalResult(
                success=False,
                verification_report=None,
                claims_extracted=[],
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    async def verify_claim(self, claim_text: str, claim_type: str = "general",
                          language: str = "es") -> VerificationResult:
        """
        Verify a single claim directly.

        Args:
            claim_text: The claim to verify
            claim_type: Type of claim (numerical, temporal, etc.)
            language: Language code

        Returns:
            Verification result for the claim
        """
        # Start performance tracking
        tracker = start_tracking("Claim Verification")

        # Create a minimal claim object
        from retrieval.core.claim_extractor import VerificationTarget, ClaimType
        claim = VerificationTarget(
            claim_text=claim_text,
            claim_type=ClaimType(claim_type) if claim_type in [ct.value for ct in ClaimType] else ClaimType.NUMERICAL,
            context="Direct verification request",
            confidence=0.8,
            priority=5,
            extracted_value=None,
            start_pos=0,
            end_pos=len(claim_text)
        )

        # Build queries
        queries = self.query_builder.build_fact_checking_queries(claim_text)

        # Search for evidence (simplified version)
        evidence_sources = []

        # Add statistical verification if applicable
        if self.statistical_api and claim_type in ['numerical', 'statistical']:
            try:
                api_results = await self.statistical_api.query_all_sources(claim_text, language)
                
                for result in api_results:
                    # Set verdict contribution and confidence based on verification result
                    verdict_contribution = VerificationVerdict.VERIFIED if result.get('verified', False) else VerificationVerdict.UNVERIFIED
                    confidence = 0.9 if result.get('verified', False) else 0.5
                    
                    evidence_sources.append(EvidenceSource(
                        source_name=result.get('source_name', 'Statistical API'),
                        source_type="statistical",
                        url=result.get('source_url', ''),
                        title=result.get('title', 'Statistical Data'),
                        credibility_score=90,  # Will be overridden by CredibilityScorer
                        content_snippet=result.get('description', ''),
                        verdict_contribution=verdict_contribution,
                        confidence=confidence
                    ))
            except Exception as e:
                self.logger.warning(f"Statistical API query failed: {e}")

        # Add web scraping for temporal claims
        if self.web_scraper_manager and claim_type in ['temporal', 'event']:
            try:
                scraped_results = self.web_scraper_manager.search_specific_sources(claim_text, ['wikipedia'], max_per_source=3)
                
                for result in scraped_results:
                    # For temporal claims, check if the content mentions the date
                    content_mention = claim_text.lower() in result.content.lower()
                    title_mention = any(word in result.title.lower() for word in claim_text.lower().split())
                    
                    verdict_contribution = VerificationVerdict.VERIFIED if (content_mention or title_mention) else VerificationVerdict.UNVERIFIED
                    confidence = 0.8 if content_mention else 0.6
                    
                    evidence_sources.append(EvidenceSource(
                        source_name=result.source_name,
                        source_type="web_search",
                        url=result.url,
                        title=result.title,
                        credibility_score=95,  # Keep high credibility for Wikipedia
                        content_snippet=result.content[:500],  # Limit snippet length
                        verdict_contribution=verdict_contribution,
                        confidence=confidence
                    ))
            except Exception as e:
                self.logger.warning(f"Web scraping failed: {e}")

        # Score sources
        scored_sources = self.credibility_scorer.batch_score_sources(evidence_sources)

        # Aggregate evidence
        result = self.evidence_aggregator.aggregate_evidence(claim, scored_sources)

        # Increment operations counter and print summary
        tracker.increment_operations(1)
        metrics = stop_tracking(tracker)
        print_performance_summary(metrics)

        return result

    async def analyze_with_verification(self, content: str, analyzer_result: Dict[str, Any]) -> AnalysisResult:
        """
        Analyze content with verification data.

        Args:
            content: Original content
            analyzer_result: Result from the main analyzer

        Returns:
            Analysis result with verification data
        """
        return await self.analyzer_hooks.analyze_with_verification(content, analyzer_result)

    def extract_claims(self, content: str) -> List[Claim]:
        """
        Extract verifiable claims from content.

        Args:
            content: Content to analyze

        Returns:
            List of extracted claims
        """
        return self.claim_extractor.extract_claims(content)

    async def batch_verify(self, requests: List[VerificationRequest]) -> List[RetrievalResult]:
        """
        Verify multiple content pieces in batch.

        Args:
            requests: List of verification requests

        Returns:
            List of verification results
        """
        tasks = [self.verify_content(request) for request in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all system components."""
        status = {
            "claim_extractor": "operational",
            "evidence_aggregator": "operational",
            "query_builder": "operational",
            "credibility_scorer": "operational",
            "multi_source_verifier": "operational",
            "analyzer_hooks": "operational"
        }

        # Check optional components
        if self.statistical_api:
            status["statistical_api"] = "operational"
        else:
            status["statistical_api"] = "disabled"

        return status

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "components": self.get_component_status(),
            "config": {
                "statistical_apis_enabled": self.config.enable_statistical_apis,
                "max_parallel_requests": self.config.max_parallel_requests,
                "verification_timeout": self.config.verification_timeout
            }
        }

        issues = []

        # Test basic functionality
        try:
            test_claims = self.extract_claims("Según datos oficiales, hay 47 millones de españoles.")
            if not test_claims:
                issues.append("Claim extraction not working")
        except Exception as e:
            issues.append(f"Claim extraction error: {e}")

        if issues:
            health["status"] = "degraded"
            health["issues"] = issues

        return health


def create_retrieval_api(config: Optional[RetrievalConfig] = None) -> RetrievalAPI:
    """
    Factory function to create configured retrieval API.

    Args:
        config: Optional configuration

    Returns:
        Configured retrieval API instance
    """
    return RetrievalAPI(config=config)


# Convenience functions for common use cases
async def verify_text_content(text: str, category: str = "general", language: str = "es") -> RetrievalResult:
    """
    Convenience function to verify text content.

    Args:
        text: Content to verify
        category: Content category
        language: Language code

    Returns:
        Verification result
    """
    api = create_retrieval_api()
    request = VerificationRequest(
        content=text,
        content_category=category,
        language=language
    )
    return await api.verify_content(request)


async def verify_single_claim(claim_text: str, claim_type: str = "general") -> VerificationResult:
    """
    Convenience function to verify a single claim.

    Args:
        claim_text: Claim to verify
        claim_type: Type of claim

    Returns:
        Verification result
    """
    api = create_retrieval_api()
    return await api.verify_claim(claim_text, claim_type)