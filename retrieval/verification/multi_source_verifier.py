"""
Multi-source verification logic that combines evidence from multiple sources
to provide comprehensive claim verification.
"""

import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging
import re

from ..core.models import VerificationResult, EvidenceSource, VerificationVerdict
from ..core.claim_extractor import ClaimExtractor, Claim, ClaimType
from ..core.evidence_aggregator import EvidenceAggregator
from ..verification.credibility_scorer import CredibilityScorer
from ..verification.temporal_verifier import TemporalVerifier
from ..sources.statistical_apis import StatisticalAPIManager
from ..sources.fact_checkers import FactCheckManager
from ..sources.web_scrapers import WebScraperManager, ScrapedContent
from ..verification.numerical_verifier import NumericalVerifier
from ..core.query_builder import QueryBuilder


@dataclass
class VerificationContext:
    """Context information for verification."""
    original_text: str
    content_category: str
    user_context: Optional[str] = None
    language: str = "es"
    priority_level: str = "balanced"  # fast, balanced, quality


@dataclass
class VerificationReport:
    """Comprehensive verification report."""
    overall_verdict: VerificationVerdict
    confidence_score: float
    claims_verified: List[VerificationResult]
    evidence_sources: List[EvidenceSource]
    temporal_consistency: bool
    contradictions_found: List[str]
    processing_time: float
    verification_method: str


class MultiSourceVerifier:
    """
    Orchestrates multi-source verification by combining:
    - Claim extraction
    - Evidence aggregation from multiple sources
    - Credibility scoring
    - Temporal verification
    - Statistical data validation
    """

    def __init__(self, max_workers: int = 4):
        self.claim_extractor = ClaimExtractor()
        self.evidence_aggregator = EvidenceAggregator()
        self.credibility_scorer = CredibilityScorer()
        self.temporal_verifier = TemporalVerifier()
        self.statistical_api = StatisticalAPIManager()
        self.fact_checker = FactCheckManager()
        self.web_scraper = WebScraperManager()
        self.numerical_verifier = NumericalVerifier()
        self.query_builder = QueryBuilder()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)

    async def verify_content(self, context: VerificationContext) -> VerificationReport:
        """
        Perform comprehensive verification of content.

        Args:
            context: Verification context with text and metadata

        Returns:
            Comprehensive verification report
        """
        import time
        start_time = time.time()

        try:
            # Phase 1: Extract claims from content
            claims = await self._extract_claims_async(context.original_text)

            if not claims:
                return self._create_empty_report(context, time.time() - start_time)

            # Phase 2: Parallel verification of claims
            verification_tasks = []
            for claim in claims:
                task = self._verify_single_claim_async(claim, context)
                verification_tasks.append(task)

            # Execute all verifications in parallel
            verified_claims = await asyncio.gather(*verification_tasks, return_exceptions=True)

            # Filter out exceptions and collect successful results
            successful_verifications = []
            for i, result in enumerate(verified_claims):
                if isinstance(result, Exception):
                    self.logger.error(f"Verification failed for claim {i}: {result}")
                else:
                    successful_verifications.append(result)

            # Phase 3: Aggregate evidence and check consistency
            overall_verdict, confidence = self._aggregate_overall_verdict(successful_verifications)
            contradictions = self._detect_contradictions(successful_verifications)
            temporal_consistency = self._check_temporal_consistency(successful_verifications)

            # Phase 4: Collect all evidence sources
            all_sources = self._collect_evidence_sources(successful_verifications)

            processing_time = time.time() - start_time

            return VerificationReport(
                overall_verdict=overall_verdict,
                confidence_score=confidence,
                claims_verified=successful_verifications,
                evidence_sources=all_sources,
                temporal_consistency=temporal_consistency,
                contradictions_found=contradictions,
                processing_time=processing_time,
                verification_method="multi_source"
            )

        except Exception as e:
            self.logger.error(f"Multi-source verification failed: {e}")
            return self._create_error_report(context, str(e), time.time() - start_time)

    async def _extract_claims_async(self, text: str) -> List[Claim]:
        """Extract claims asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.claim_extractor.extract_claims, text)

    async def _verify_single_claim_async(self, claim: Claim, context: VerificationContext) -> VerificationResult:
        """Verify a single claim asynchronously."""
        loop = asyncio.get_event_loop()

        # Build search queries for this claim
        queries = await loop.run_in_executor(
            self.executor,
            self.query_builder.build_fact_checking_queries,
            claim.claim_text
        )

        # Parallel search across multiple sources
        search_tasks = []
        for query in queries[:3]:  # Limit to top 3 queries
            # Statistical APIs for numerical claims
            if claim.claim_type == ClaimType.NUMERICAL:
                search_tasks.append(self._search_statistical_apis_async(claim, context))

        # Add fact-checking for all claims
        search_tasks.append(self._search_fact_checkers_async(claim.claim_text, context))

        # Add web scraping for additional sources
        search_tasks.append(self._search_web_scrapers_async(claim.claim_text, context))

        # Add numerical verification for numerical claims
        # if claim.claim_type == ClaimType.NUMERICAL:
        #     search_tasks.append(self._verify_numerical_claim_async(claim, context))

        # Execute searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process and aggregate evidence
        evidence_sources = []
        for result in search_results:
            if isinstance(result, list):
                evidence_sources.extend(result)

        # Score credibility of sources
        scored_sources = await loop.run_in_executor(
            self.executor,
            self.credibility_scorer.batch_score_sources,
            evidence_sources,
            claim.claim_text  # Pass claim text for context-aware scoring
        )

        # Aggregate evidence
        aggregated_result = VerificationResult(
            claim=claim.claim_text,
            verdict=VerificationVerdict.UNVERIFIED,  # Will be updated by evidence
            confidence=0.0,  # Will be updated by evidence
            evidence_sources=scored_sources,
            explanation="",
            processing_time_seconds=0.0,
            claim_type=claim.claim_type,
            extracted_value=claim.extracted_value
        )

        # Update verdict based on evidence
        aggregated_result._update_verdict()

        # Add temporal verification if applicable
        has_temporal_info = any(re.search(pattern, claim.claim_text, re.IGNORECASE)
                              for pattern in self.claim_extractor.temporal_patterns)
        if claim.claim_type in [ClaimType.TEMPORAL] or has_temporal_info:
            temporal_verified, temporal_explanation, verified_date = await loop.run_in_executor(
                self.executor,
                self.temporal_verifier.verify_temporal_claim,
                claim.claim_text,
                context.original_text
            )

            # Only adjust verdict for temporal claims or if temporal verification strongly contradicts statistical evidence
            if not temporal_verified:
                # For numerical claims that are VERIFIED by statistical sources, don't downgrade due to temporal issues
                has_verified_statistical_sources = any(
                    source.verdict_contribution == VerificationVerdict.VERIFIED and source.source_type == "statistical"
                    for source in aggregated_result.evidence_sources
                )

                if claim.claim_type == ClaimType.NUMERICAL and has_verified_statistical_sources:
                    # Keep VERIFIED verdict for statistical claims, but add explanation
                    aggregated_result.explanation += f" (Nota temporal: {temporal_explanation})"
                elif aggregated_result.verdict == VerificationVerdict.VERIFIED:
                    aggregated_result.verdict = VerificationVerdict.QUESTIONABLE
                    aggregated_result.explanation += f" Sin embargo, {temporal_explanation}"

        return aggregated_result

    async def _search_statistical_apis_async(self, claim: Claim, context: VerificationContext) -> List[EvidenceSource]:
        """Search statistical APIs for numerical claims."""
        loop = asyncio.get_event_loop()

        # Check if claim has numerical value
        if not claim.extracted_value:
            return []

        # Query statistical APIs
        is_verified, data_point = await loop.run_in_executor(
            self.executor,
            self.statistical_api.verify_numerical_claim,
            f"{claim.claim_text}. {context.original_text[:200]}",  # Include context for better keyword matching
            claim.extracted_value or ""
        )

        # Convert result to EvidenceSource
        evidence_sources = []
        if data_point:
            evidence_sources.append(EvidenceSource(
                source_name=data_point.source_name,
                source_type="statistical",
                url=data_point.source_url,  # Fixed: was trying to pass source_url but EvidenceSource expects url
                title=data_point.title,
                credibility_score=90,  # Statistical APIs are highly credible
                publication_date=datetime.now() - timedelta(days=30),  # Assume recent
                content_snippet=data_point.description,
                verdict_contribution=VerificationVerdict.VERIFIED if is_verified else VerificationVerdict.DEBUNKED,
                confidence=1.0 if is_verified else 0.5  # High confidence for verified statistical data
            ))

        return evidence_sources

    async def _search_fact_checkers_async(self, claim_text: str, context: VerificationContext) -> List[EvidenceSource]:
        """Search fact-checking sites for evidence."""
        loop = asyncio.get_event_loop()

        try:
            fact_check_results = await loop.run_in_executor(
                self.executor,
                self.fact_checker.search_all_sources,
                claim_text,
                2  # max per source
            )

            evidence_sources = []
            for result in fact_check_results:
                evidence_sources.append(EvidenceSource(
                    source_name=result.source_name,
                    source_type="fact_checker",
                    url=result.source_url,
                    title=f"Verificación: {result.claim[:50]}...",
                    credibility_score=0.1,  # TEMP: Test if results come from fact-checkers
                    publication_date=result.publication_date,
                    content_snippet=result.explanation[:200],
                    verdict_contribution=self._map_fact_check_verdict(result.verdict),
                    confidence=result.confidence
                ))

            return evidence_sources

        except Exception as e:
            self.logger.error(f"Fact-checker search failed: {e}")
            return []

    async def _search_web_scrapers_async(self, claim_text: str, context: VerificationContext) -> List[EvidenceSource]:
        """Search web sources using scrapers."""
        loop = asyncio.get_event_loop()

        try:
            scraped_content = await loop.run_in_executor(
                self.executor,
                self.web_scraper.search_all_sources,
                claim_text,
                2  # max per source
            )

            evidence_sources = []
            for content in scraped_content:
                # Adjust verdict contribution and credibility based on claim type
                adjusted_verdict = VerificationVerdict.UNVERIFIED
                adjusted_credibility = content.credibility_score

                claim_lower = claim_text.lower()
                # TEMP: Always apply credibility reduction for testing
                if True:  # context.content_category in ['economy', 'statistics'] or 'pib' in claim_lower or 'gdp' in claim_lower or 'economía' in claim_lower or 'crecimiento' in claim_lower:
                    adjusted_verdict = VerificationVerdict.QUESTIONABLE
                    # Reduce credibility for ALL sources on statistical topics (news sources are not authoritative)
                    adjusted_credibility = min(content.credibility_score * 0.5, 0.6)  # More aggressive reduction, cap at 0.6
                else:
                    adjusted_verdict = VerificationVerdict.UNVERIFIED
                    adjusted_credibility = content.credibility_score

                evidence_sources.append(EvidenceSource(
                    source_name=content.source_name,
                    source_type="news",
                    url=content.url,
                    title=content.title,
                    credibility_score=adjusted_credibility,
                    publication_date=content.publication_date,
                    content_snippet=content.content[:300],
                    verdict_contribution=adjusted_verdict,
                    confidence=0.5  # Moderate confidence for scraped content
                ))

            return evidence_sources

        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            return []

    async def _verify_numerical_claim_async(self, claim: Claim, context: VerificationContext) -> List[EvidenceSource]:
        """Verify numerical claims using specialized verifier."""
        loop = asyncio.get_event_loop()

        try:
            # Create text context for numerical verification
            text_context = f"{claim.claim_text}. {context.original_text[:200]}"

            numerical_results = await loop.run_in_executor(
                self.executor,
                self.numerical_verifier.verify_numerical_claim,
                text_context,
                context.original_text  # Pass full original text for better statistical API matching
            )

            evidence_sources = []
            for result in numerical_results[:2]:  # Limit to top 2 results
                verdict_contribution = VerificationVerdict.UNVERIFIED
                if result.verdict == VerificationVerdict.VERIFIED:
                    verdict_contribution = VerificationVerdict.VERIFIED
                elif result.verdict == VerificationVerdict.DEBUNKED:
                    verdict_contribution = VerificationVerdict.DEBUNKED

                evidence_sources.append(EvidenceSource(
                    source_name=result.source or "Numerical Verification",
                    source_type="statistical",
                    url="",  # No URL for computed results
                    title=f"Numerical claim: {result.claim.original_text}",
                    credibility_score=0.90,  # High credibility for verified data (0.0-1.0 range)
                    content_snippet=result.explanation,
                    verdict_contribution=verdict_contribution,
                    confidence=result.confidence
                ))

            return evidence_sources

        except Exception as e:
            self.logger.error(f"Numerical verification failed: {e}")
            return []

    def _map_fact_check_verdict(self, fact_verdict: str) -> VerificationVerdict:
        """Map fact-check verdict to standard verdict."""
        verdict_lower = fact_verdict.lower()
        if verdict_lower in ['true', 'verdadero', 'cierto']:
            return VerificationVerdict.VERIFIED
        elif verdict_lower in ['false', 'falso', 'mentira']:
            return VerificationVerdict.DEBUNKED
        elif verdict_lower in ['misleading', 'engañoso']:
            return VerificationVerdict.QUESTIONABLE
        else:
            return VerificationVerdict.UNVERIFIED

    def _aggregate_overall_verdict(self, verifications: List[VerificationResult]) -> Tuple[VerificationVerdict, float]:
        """Aggregate overall verdict from individual claim verifications."""
        if not verifications:
            return VerificationVerdict.UNVERIFIED, 0.0

        # Count verdicts
        verdict_counts = {
            VerificationVerdict.VERIFIED: 0,
            VerificationVerdict.QUESTIONABLE: 0,
            VerificationVerdict.DEBUNKED: 0,
            VerificationVerdict.UNVERIFIED: 0
        }

        total_confidence = 0

        for verification in verifications:
            verdict_counts[verification.verdict] += 1
            total_confidence += verification.confidence

        # Determine overall verdict based on majority
        max_count = max(verdict_counts.values())
        majority_verdicts = [v for v, c in verdict_counts.items() if c == max_count]

        # If tie, prefer more conservative verdict
        if len(majority_verdicts) > 1:
            verdict_priority = {
                VerificationVerdict.DEBUNKED: 0,
                VerificationVerdict.QUESTIONABLE: 1,
                VerificationVerdict.UNVERIFIED: 2,
                VerificationVerdict.VERIFIED: 3
            }
            overall_verdict = min(majority_verdicts, key=lambda x: verdict_priority[x])
        else:
            overall_verdict = majority_verdicts[0]

        # Calculate average confidence
        avg_confidence = total_confidence / len(verifications)

        return overall_verdict, avg_confidence

    def _detect_contradictions(self, verifications: List[VerificationResult]) -> List[str]:
        """Detect contradictions between different claims."""
        contradictions = []

        # Group claims by topic
        topic_groups = {}
        for verification in verifications:
            # Simple topic extraction based on keywords
            topic = self._extract_topic(verification.claim)
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(verification)

        # Check for contradictions within each topic
        for topic, claims in topic_groups.items():
            verdicts = [c.verdict for c in claims]
            if VerificationVerdict.VERIFIED in verdicts and VerificationVerdict.DEBUNKED in verdicts:
                contradictions.append(f"Contradicción en {topic}: algunas afirmaciones verificadas, otras desmentidas")

        return contradictions

    def _check_temporal_consistency(self, verifications: List[VerificationResult]) -> bool:
        """Check temporal consistency across claims."""
        temporal_claims = [v for v in verifications if v.claim_type in ['temporal', 'event']]

        if len(temporal_claims) < 2:
            return True  # Not enough temporal claims to check consistency

        # Extract events with dates
        events = []
        for verification in temporal_claims:
            # Try to extract date from verification result
            if hasattr(verification, 'verified_date') and verification.verified_date:
                events.append((verification.claim, verification.verified_date))

        if len(events) < 2:
            return True

        # Check for temporal consistency
        issues = self.temporal_verifier.validate_event_timeline(events)
        return len(issues) == 0

    def _collect_evidence_sources(self, verifications: List[VerificationResult]) -> List[EvidenceSource]:
        """Collect all evidence sources from verifications."""
        all_sources = []
        seen_urls = set()

        for verification in verifications:
            for source in verification.evidence_sources:
                if source.url not in seen_urls:
                    all_sources.append(source)
                    seen_urls.add(source.url)

        return all_sources

    def _extract_topic(self, text: str) -> str:
        """Extract topic from text for grouping."""
        # Simple keyword-based topic extraction
        text_lower = text.lower()

        if any(word in text_lower for word in ['covid', 'pandemia', 'vacuna', 'salud']):
            return "salud"
        elif any(word in text_lower for word in ['elecciones', 'voto', 'partido', 'política']):
            return "política"
        elif any(word in text_lower for word in ['economía', 'paro', 'empleo', 'pib']):
            return "economía"
        else:
            return "general"

    def _create_empty_report(self, context: VerificationContext, processing_time: float) -> VerificationReport:
        """Create report when no claims are found."""
        return VerificationReport(
            overall_verdict=VerificationVerdict.UNVERIFIED,
            confidence_score=0.0,
            claims_verified=[],
            evidence_sources=[],
            temporal_consistency=True,
            contradictions_found=[],
            processing_time=processing_time,
            verification_method="multi_source"
        )

    def _create_error_report(self, context: VerificationContext, error: str, processing_time: float) -> VerificationReport:
        """Create report when verification fails."""
        return VerificationReport(
            overall_verdict=VerificationVerdict.UNVERIFIED,
            confidence_score=0.0,
            claims_verified=[],
            evidence_sources=[],
            temporal_consistency=False,
            contradictions_found=[f"Error en verificación: {error}"],
            processing_time=processing_time,
            verification_method="multi_source"
        )


async def verify_content_async(text: str, category: str = "general", language: str = "es") -> VerificationReport:
    """
    Convenience function for async content verification.

    Args:
        text: Content to verify
        category: Content category
        language: Language code

    Returns:
        Verification report
    """
    context = VerificationContext(
        original_text=text,
        content_category=category,
        language=language
    )

    verifier = MultiSourceVerifier()
    return await verifier.verify_content(context)