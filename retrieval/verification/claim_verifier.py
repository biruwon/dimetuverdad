"""
Consolidated claim verification system.
Combines multi-source, numerical, and temporal verification into a single, simplified verifier.
"""

import asyncio
import re
import math
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
import time

from ..core.models import VerificationResult, EvidenceSource, VerificationVerdict
from ..core.claim_extractor import ClaimExtractor, Claim, ClaimType
from ..verification.credibility_scorer import CredibilityScorer
from ..sources.statistical_apis import StatisticalAPIManager
from ..sources.fact_checkers import FactCheckManager
from ..sources.web_scrapers import WebScraperManager
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


class NumericalClaimType(Enum):
    """Types of numerical claims."""
    POPULATION = "population"
    ECONOMIC = "economic"  # GDP, inflation, unemployment
    HEALTH = "health"      # COVID cases, vaccination rates
    POLITICAL = "political"  # Election results, polling data
    GENERAL = "general"    # Other numerical claims


@dataclass
class NumericalClaim:
    """A parsed numerical claim."""
    original_text: str
    value: Union[int, float]
    unit: str  # millions, billions, percent, etc.
    claim_type: NumericalClaimType
    context: str
    time_reference: Optional[str] = None  # year, month, etc.
    confidence: float = 0.8


@dataclass
class NumericalVerificationResult:
    """Result of numerical claim verification."""
    claim: NumericalClaim
    verdict: VerificationVerdict
    confidence: float
    expected_value: Optional[Union[int, float]] = None
    actual_value: Optional[Union[int, float]] = None
    source: Optional[str] = None
    explanation: str = ""
    margin_of_error: Optional[float] = None


@dataclass
class TemporalClaim:
    """A temporal claim extracted from text."""
    claim_text: str
    date_mentioned: Optional[datetime]
    time_period: Optional[str]  # e.g., "hace 3 días", "en 2020"
    temporal_type: str  # 'specific_date', 'relative_time', 'year_only', 'period'
    context: str


class ClaimVerifier:
    """
    Consolidated claim verification system.
    Combines multi-source verification, numerical validation, and temporal checking
    into a single, simplified verifier with source priority scoring.
    """

    def __init__(self, max_workers: int = 4):
        # Core components
        self.claim_extractor = ClaimExtractor()
        self.credibility_scorer = CredibilityScorer()
        self.statistical_api = StatisticalAPIManager()
        self.fact_checker = FactCheckManager()
        self.web_scraper = WebScraperManager()
        self.query_builder = QueryBuilder()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)

        # Source priority mapping (higher = more trusted)
        self.source_priorities = {
            'ine': 10,        # Spanish National Statistics Institute
            'eurostat': 9,
            'who': 9,         # World Health Organization
            'worldbank': 8,
            'oecd': 8,
            'maldita': 9,     # Fact-checking organizations
            'newtral': 9,
            'snopes': 8,
            'politifact': 8,
            'factcheck': 8,
            'boe': 7,         # Spanish Official Gazette
            'mscbs': 8,       # Health Ministry
            'exteriores': 7,   # Foreign Affairs
            'elpais': 6,      # Major news outlets
            'elmundo': 6,
            'abc': 5,
            'lavanguardia': 5,
            'scholar': 7,     # Academic sources
            'dialnet': 6,
            'unknown': 3      # Default priority
        }

        # Initialize integrated parsers
        self._init_parsers()

        # Initialize integrated credibility scoring
        self._init_credibility_scorer()

        # Add caching for domain extraction
        self._domain_cache = {}
        self._tld_cache = {}

    def _init_credibility_scorer(self):
        """Initialize integrated credibility scoring components."""
        # Base credibility scores for known sources (0.0 to 1.0)
        self.source_base_scores = {
            # Fact-checking organizations (highest credibility)
            'maldita.es': 0.95,
            'maldita': 0.95,
            'newtral.es': 0.95,
            'newtral': 0.95,
            'snopes.com': 0.90,
            'snopes': 0.90,
            'politifact.com': 0.90,
            'politifact': 0.90,
            'factcheck.org': 0.90,
            'factcheck': 0.90,

            # Statistical agencies
            'ine.es': 0.98,
            'ine': 0.98,
            'eurostat.europa.eu': 0.95,
            'eurostat': 0.95,
            'oficina estadística de la unión europea': 0.95,  # Eurostat full name
            'eurostat - oficina estadística de la unión europea': 0.95,  # Exact match
            'who.int': 0.93,
            'who': 0.93,
            'world health organization': 0.93,
            'worldbank.org': 0.92,
            'worldbank': 0.92,
            'world bank': 0.92,
            'banco mundial': 0.92,
            'oecd.org': 0.90,
            'oecd': 0.90,

            # Spanish government sources
            'boe.es': 0.85,
            'boe': 0.85,
            'mscbs.gob.es': 0.88,
            'mscbs': 0.88,
            'exteriores.gob.es': 0.82,
            'exteriores': 0.82,
            'gobierno.es': 0.80,
            'gobierno': 0.80,

            # News outlets (Spanish mainstream)
            'elpais.com': 0.75,
            'elpais': 0.75,
            'elmundo.es': 0.70,
            'elmundo': 0.70,
            'abc.es': 0.68,
            'abc': 0.68,
            'lavanguardia.com': 0.65,
            'lavanguardia': 0.65,

            # Academic and research
            'scholar.google.com': 0.80,
            'scholar': 0.80,
            'dialnet.unirioja.es': 0.75,
            'dialnet': 0.75,
            'sciencedirect.com': 0.85,
            'nature.com': 0.88,
            'wikipedia.org': 0.95,
            'wikipedia': 0.95,
            'wikipedia (es)': 0.95,

            # International news
            'bbc.com': 0.75,
            'bbc': 0.75,
            'reuters.com': 0.80,
            'reuters': 0.80,
            'apnews.com': 0.78,
            'apnews': 0.78,

            # Domain-based scoring for unknown sources
            'gov': 0.75,      # Government domains
            'edu': 0.70,      # Educational institutions
            'org': 0.65,      # Non-profit organizations
            'com': 0.50,      # Commercial domains
            'unknown': 0.40   # Default for unrecognized
        }

        # Content quality indicators (pre-processed for faster lookup)
        self.quality_indicators = {
            'high': ['estudio científico', 'investigación', 'datos oficiales', 'informe oficial',
                    'evidencia empírica', 'análisis estadístico', 'fuentes primarias'],
            'medium': ['análisis', 'investigación', 'datos', 'informe', 'estudio',
                      'verificación', 'comprobación'],
            'low': ['opinión', 'creo que', 'parece', 'probablemente', 'quizás']
        }

        # Pre-compute quality indicator sets for faster lookup
        self._high_quality_set = set(self.quality_indicators['high'])
        self._medium_quality_set = set(self.quality_indicators['medium'])
        self._low_quality_set = set(self.quality_indicators['low'])

        # Bias indicators (reduce credibility)
        self.bias_indicators = [
            'teoría conspirativa', 'fake news', 'desinformación',
            'engaño', 'manipulación', 'propaganda'
        ]

        # Pre-compute statistical keywords set for faster lookup
        self._statistical_keywords = {
            'pib', 'gdp', 'economía', 'crecimiento', 'paro', 'empleo',
            'inflación', 'déficit', 'estadística', 'datos oficiales',
            'tasa de', 'porcentaje', 'millones', 'billones'
        }

        # Pre-compute news domains set for faster lookup
        self._news_domains = {
            'elpais.com', 'elmundo.es', 'abc.es', 'lavanguardia.com',
            'bbc.com', 'reuters.com', 'apnews.com'
        }

    def _init_parsers(self):
        """Initialize integrated parsing components."""
        # Pre-compile numerical parsing patterns for performance
        self.number_patterns = [
            re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'),
            re.compile(r'\b\d+\.\d+\b'),
            re.compile(r'\b\d+\b'),
        ]

        # Pre-compile unit patterns
        self.unit_patterns = {
            'million': re.compile(r'millon(?:es)?', re.IGNORECASE),
            'billion': re.compile(r'billon(?:es)?', re.IGNORECASE),
            'percent': re.compile(r'%|por ciento', re.IGNORECASE),
            'euro': re.compile(r'euros?|€', re.IGNORECASE),
            'people': re.compile(r'personas?|habitantes?', re.IGNORECASE),
        }

        # Pre-compile year pattern for efficiency
        self.year_pattern = re.compile(r'^(19|20)\d{2}$')

        # Temporal verification data
        self.known_events = {
            'pandemia covid': datetime(2020, 3, 11),
            'elecciones generales 2023': datetime(2023, 7, 23),
        }

        # Pre-compile date patterns
        self.date_patterns = [
            re.compile(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b'),
            re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
        ]

        # Pre-compile relative time patterns
        self.relative_time_patterns = [
            re.compile(r'\bhace\s+(\d+)\s+(días?|meses?|años?)\b', re.IGNORECASE),
            re.compile(r'\ben\s+(?:el\s+)?(\d{4})\b', re.IGNORECASE),
        ]

        # Pre-compile year extraction pattern
        self.year_extract_pattern = re.compile(r'\b(19|20)\d{2}\b')

        # Pre-compile word extraction pattern for relevance assessment
        self.word_pattern = re.compile(r'\w{4,}')

    async def verify_content(self, context: VerificationContext) -> VerificationReport:
        """
        Perform comprehensive verification of content.

        Args:
            context: Verification context with text and metadata

        Returns:
            Comprehensive verification report
        """
        start_time = time.time()

        try:
            # Extract claims from content
            claims = await self._extract_claims_async(context.original_text)

            if not claims:
                return self._create_empty_report(context, time.time() - start_time)

            # Verify claims in parallel
            verification_tasks = []
            for claim in claims:
                task = self._verify_single_claim_async(claim, context)
                verification_tasks.append(task)

            verified_claims = await asyncio.gather(*verification_tasks, return_exceptions=True)

            # Filter successful results
            successful_verifications = []
            for i, result in enumerate(verified_claims):
                if isinstance(result, Exception):
                    self.logger.error(f"Verification failed for claim {i}: {result}")
                else:
                    successful_verifications.append(result)

            # Aggregate results using simplified priority scoring
            overall_verdict, confidence = self._aggregate_by_priority(successful_verifications)
            contradictions = self._detect_contradictions(successful_verifications)
            temporal_consistency = self._check_temporal_consistency(successful_verifications)

            # Collect all evidence sources
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
                verification_method="consolidated"
            )

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return self._create_error_report(context, str(e), time.time() - start_time)

    async def _extract_claims_async(self, text: str) -> List[Claim]:
        """Extract claims asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.claim_extractor.extract_claims, text)

    async def _verify_single_claim_async(self, claim: Claim, context: VerificationContext) -> VerificationResult:
        """Verify a single claim using appropriate verification method."""
        # Choose verification strategy based on claim type
        if claim.claim_type == ClaimType.NUMERICAL:
            return await self._verify_numerical_claim_async(claim, context)
        elif claim.claim_type == ClaimType.TEMPORAL:
            return await self._verify_temporal_claim_async(claim, context)
        else:
            return await self._verify_general_claim_async(claim, context)

    async def _verify_numerical_claim_async(self, claim: Claim, context: VerificationContext) -> VerificationResult:
        """Verify numerical claims using statistical APIs."""
        loop = asyncio.get_event_loop()

        # Parse numerical value using integrated parser
        numerical_claims = await loop.run_in_executor(
            self.executor, self._parse_numerical_claims, claim.claim_text
        )

        if not numerical_claims:
            return VerificationResult(
                claim=claim.claim_text,
                verdict=VerificationVerdict.UNVERIFIED,
                confidence=0.0,
                explanation="No se pudo parsear el valor numérico"
            )

        num_claim = numerical_claims[0]  # Use first parsed claim

        # Check statistical APIs
        is_verified, data_point = await loop.run_in_executor(
            self.executor,
            self.statistical_api.verify_numerical_claim,
            f"{claim.claim_text}. {context.original_text[:200]}",
            str(int(num_claim.value)) if num_claim.value.is_integer() else str(num_claim.value)
        )

        evidence_sources = []
        if data_point:
            verdict = VerificationVerdict.VERIFIED if is_verified else VerificationVerdict.DEBUNKED
            confidence = 0.9 if is_verified else 0.6

            evidence_sources.append(EvidenceSource(
                source_name=data_point.source_name,
                source_type="statistical",
                url=data_point.source_url,
                title=f"Verificación: {num_claim.value}",
                credibility_score=self._get_source_priority(data_point.source_name.lower()),
                publication_date=datetime.now() - timedelta(days=30),
                content_snippet=data_point.description,
                verdict_contribution=verdict,
                confidence=confidence
            ))

        # Create result
        result = VerificationResult(
            claim=claim.claim_text,
            verdict=VerificationVerdict.UNVERIFIED,
            confidence=0.0,
            evidence_sources=evidence_sources,
            explanation="",
            claim_type=claim.claim_type,
            extracted_value=num_claim.value
        )

        # Update verdict based on evidence
        result._update_verdict()
        return result

    async def _verify_temporal_claim_async(self, claim: Claim, context: VerificationContext) -> VerificationResult:
        """Verify temporal claims."""
        loop = asyncio.get_event_loop()

        # Use integrated temporal verifier
        is_verified, explanation, verified_date = await loop.run_in_executor(
            self.executor,
            self._verify_temporal_claim,
            claim.claim_text,
            context.original_text
        )

        verdict = VerificationVerdict.VERIFIED if is_verified else VerificationVerdict.DEBUNKED
        confidence = 0.8 if is_verified else 0.6

        # Create evidence source for temporal verification
        evidence_sources = []
        if verified_date:
            evidence_sources.append(EvidenceSource(
                source_name="Verificación Temporal",
                source_type="temporal",
                url="",
                title=f"Fecha verificada: {verified_date.strftime('%d/%m/%Y') if verified_date else 'N/A'}",
                credibility_score=8,  # High credibility for temporal logic
                publication_date=datetime.now(),
                content_snippet=explanation,
                verdict_contribution=verdict,
                confidence=confidence
            ))

        result = VerificationResult(
            claim=claim.claim_text,
            verdict=verdict,
            confidence=confidence,
            evidence_sources=evidence_sources,
            explanation=explanation,
            claim_type=claim.claim_type
        )

        return result

    async def _verify_general_claim_async(self, claim: Claim, context: VerificationContext) -> VerificationResult:
        """Verify general claims using fact-checkers and web sources."""
        loop = asyncio.get_event_loop()

        # Build search queries
        queries = await loop.run_in_executor(
            self.executor,
            self.query_builder.build_fact_checking_queries,
            claim.claim_text
        )

        # Search fact-checkers and web sources in parallel
        search_tasks = [
            self._search_fact_checkers_async(claim.claim_text, context),
            self._search_web_scrapers_async(claim.claim_text, context)
        ]

        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect evidence sources
        evidence_sources = []
        for result in search_results:
            if isinstance(result, list):
                evidence_sources.extend(result)

        # Score sources using async credibility scorer
        scored_sources = await self._batch_score_sources_async(evidence_sources, claim.claim_text)

        # Apply priority adjustments
        for source in scored_sources:
            priority = self._get_source_priority(source.source_name.lower())
            # Blend credibility score with priority (70% credibility, 30% priority)
            source.credibility_score = (source.credibility_score * 0.7) + (priority / 10 * 0.3)

        result = VerificationResult(
            claim=claim.claim_text,
            verdict=VerificationVerdict.UNVERIFIED,
            confidence=0.0,
            evidence_sources=scored_sources,
            explanation="",
            claim_type=claim.claim_type
        )

        # Update verdict based on evidence
        result._update_verdict()
        return result

    async def _search_fact_checkers_async(self, claim_text: str, context: VerificationContext) -> List[EvidenceSource]:
        """Search fact-checking sites."""
        loop = asyncio.get_event_loop()

        try:
            fact_check_results = await loop.run_in_executor(
                self.executor,
                self.fact_checker.search_all_sources,
                claim_text,
                2
            )

            evidence_sources = []
            for result in fact_check_results:
                priority = self._get_source_priority(result.source_name.lower())

                evidence_sources.append(EvidenceSource(
                    source_name=result.source_name,
                    source_type="fact_checker",
                    url=result.source_url,
                    title=f"Verificación: {result.claim[:50]}...",
                    credibility_score=priority,
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
        """Search web sources."""
        loop = asyncio.get_event_loop()

        try:
            scraped_content = await loop.run_in_executor(
                self.executor,
                self.web_scraper.search_all_sources,
                claim_text,
                2
            )

            evidence_sources = []
            for content in scraped_content:
                priority = self._get_source_priority(content.source_name.lower())

                # Adjust priority for statistical/economic content
                if context.content_category in ['economy', 'statistics']:
                    priority = min(priority, 5)  # Reduce priority for news on statistical topics

                evidence_sources.append(EvidenceSource(
                    source_name=content.source_name,
                    source_type="news",
                    url=content.url,
                    title=content.title,
                    credibility_score=priority,
                    publication_date=content.publication_date,
                    content_snippet=content.content[:300],
                    verdict_contribution=VerificationVerdict.UNVERIFIED,
                    confidence=0.5
                ))

            return evidence_sources

        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}")
            return []

    def _get_source_priority(self, source_name: str) -> int:
        """Get priority score for a source (1-10, higher is better)."""
        # Check for exact matches first
        if source_name in self.source_priorities:
            return self.source_priorities[source_name]

        # Check for partial matches
        for key, priority in self.source_priorities.items():
            if key in source_name or source_name in key:
                return priority

        return self.source_priorities['unknown']

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

    def _aggregate_by_priority(self, verifications: List[VerificationResult]) -> Tuple[VerificationVerdict, float]:
        """Aggregate verdicts using simple priority-based scoring."""
        if not verifications:
            return VerificationVerdict.UNVERIFIED, 0.0

        # Count verdicts with priority weighting
        verdict_scores = {VerificationVerdict.VERIFIED: 0, VerificationVerdict.DEBUNKED: 0,
                         VerificationVerdict.QUESTIONABLE: 0, VerificationVerdict.UNVERIFIED: 0}

        total_weight = 0

        for verification in verifications:
            # Use highest priority source for this verification
            if verification.evidence_sources:
                max_priority = max(source.credibility_score for source in verification.evidence_sources)
                weight = max_priority
            else:
                weight = 1  # Default weight

            verdict_scores[verification.verdict] += weight
            total_weight += weight

        if total_weight == 0:
            return VerificationVerdict.UNVERIFIED, 0.0

        # Find winning verdict
        best_verdict = max(verdict_scores.items(), key=lambda x: x[1])
        confidence = best_verdict[1] / total_weight

        return best_verdict[0], confidence

    def _detect_contradictions(self, verifications: List[VerificationResult]) -> List[str]:
        """Detect contradictions between claims."""
        contradictions = []

        # Group by topic
        topic_groups = {}
        for verification in verifications:
            topic = self._extract_topic(verification.claim)
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(verification)

        # Check for contradictions within topics
        for topic, claims in topic_groups.items():
            verdicts = [c.verdict for c in claims]
            if VerificationVerdict.VERIFIED in verdicts and VerificationVerdict.DEBUNKED in verdicts:
                contradictions.append(f"Contradicción en {topic}: verificaciones mixtas")

        return contradictions

    def _check_temporal_consistency(self, verifications: List[VerificationResult]) -> bool:
        """Check temporal consistency."""
        temporal_claims = [v for v in verifications if v.claim_type in ['temporal', 'event']]
        return len(temporal_claims) <= 1  # Simple check: no contradictions if 1 or fewer temporal claims

    def _collect_evidence_sources(self, verifications: List[VerificationResult]) -> List[EvidenceSource]:
        """Collect all evidence sources."""
        all_sources = []
        seen_urls = set()

        for verification in verifications:
            for source in verification.evidence_sources:
                if source.url and source.url not in seen_urls:
                    all_sources.append(source)
                    seen_urls.add(source.url)

        return all_sources

    def _extract_topic(self, text: str) -> str:
        """Extract topic from text."""
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
        """Create report when no claims found."""
        return VerificationReport(
            overall_verdict=VerificationVerdict.UNVERIFIED,
            confidence_score=0.0,
            claims_verified=[],
            evidence_sources=[],
            temporal_consistency=True,
            contradictions_found=[],
            processing_time=processing_time,
            verification_method="consolidated"
        )

    def _create_error_report(self, context: VerificationContext, error: str, processing_time: float) -> VerificationReport:
        """Create report when verification fails."""
        return VerificationReport(
            overall_verdict=VerificationVerdict.UNVERIFIED,
            confidence_score=0.0,
            claims_verified=[],
            evidence_sources=[],
            temporal_consistency=False,
            contradictions_found=[f"Error: {error}"],
            processing_time=processing_time,
            verification_method="consolidated"
        )

    # Integrated parsing methods
    def _parse_numerical_claims(self, text: str) -> List[NumericalClaim]:
        """Parse numerical claims from text (integrated method)."""
        claims = []

        for pattern in self.number_patterns:
            for match in pattern.finditer(text):
                claim = self._parse_single_numerical_claim(text, match)
                if claim:
                    claims.append(claim)

        # Remove duplicates
        unique_claims = []
        seen = set()
        for claim in claims:
            key = (claim.value, claim.original_text)
            if key not in seen:
                unique_claims.append(claim)
                seen.add(key)

        return unique_claims

    def _parse_single_numerical_claim(self, text: str, match) -> Optional[NumericalClaim]:
        """Parse single numerical claim (integrated method)."""
        number_str = match.group()
        start_pos = match.start()

        # Skip years using pre-compiled pattern
        if self.year_pattern.match(number_str):
            return None

        clean_number = number_str.replace(',', '')
        try:
            value = float(clean_number)
        except ValueError:
            return None

        context = text[max(0, start_pos-50):min(len(text), start_pos+50)]
        unit = self._determine_unit(context)

        # Determine claim type
        claim_type = NumericalClaimType.GENERAL
        context_lower = context.lower()
        if any(word in context_lower for word in ['población', 'habitantes']):
            claim_type = NumericalClaimType.POPULATION
        elif any(word in context_lower for word in ['pib', 'paro', 'economía']):
            claim_type = NumericalClaimType.ECONOMIC

        return NumericalClaim(
            original_text=number_str,
            value=value,
            unit=unit,
            claim_type=claim_type,
            context=context,
            confidence=0.8
        )

    def _determine_unit(self, context: str) -> str:
        """Determine unit from context (integrated method)."""
        context_lower = context.lower()

        for unit_name, pattern in self.unit_patterns.items():
            if pattern.search(context_lower):
                return unit_name

        return 'unknown'

    def _verify_temporal_claim(self, claim_text: str, context: str = "") -> Tuple[bool, str, Optional[datetime]]:
        """Verify temporal claim (integrated method)."""
        full_text = f"{claim_text} {context}"

        # Check for specific dates
        for pattern in self.date_patterns:
            match = pattern.search(full_text)
            if match:
                try:
                    if pattern == self.date_patterns[0]:
                        day, month, year = map(int, match.groups())
                    else:
                        year, month, day = map(int, match.groups())

                    date_mentioned = datetime(year, month, day)
                    now = datetime.now()

                    if date_mentioned > now:
                        return False, f"Fecha futura: {date_mentioned.strftime('%d/%m/%Y')}", date_mentioned

                    return True, f"Fecha plausible: {date_mentioned.strftime('%d/%m/%Y')}", date_mentioned
                except ValueError:
                    continue

        # Check for years using pre-compiled pattern
        year_match = self.year_extract_pattern.search(full_text)
        if year_match:
            year = int(year_match.group())
            current_year = datetime.now().year

            if year < 1900 or year > current_year + 2:
                return False, f"Año improbable: {year}", datetime(year, 1, 1)

            return True, f"Año plausible: {year}", datetime(year, 1, 1)

        # Check relative time
        for pattern in self.relative_time_patterns:
            match = pattern.search(full_text)
            if match:
                return True, "Tiempo relativo plausible", None

        return False, "No se pudo verificar información temporal", None

    # Integrated credibility scoring methods
    def _score_source(self, source, context_claim: str = "") -> float:
        """Calculate overall credibility score for a source (integrated method)."""
        # Preserve high credibility scores for known reliable sources
        source_key = source.source_name.lower()
        if source_key in ['wikipedia (es)', 'wikipedia'] or 'wikipedia' in source_key:
            if source.credibility_score >= 0.9:  # If already set high, preserve it
                return source.credibility_score

        # Start with base score from source reputation
        base_score = self._get_base_score(source)

        # Apply content quality modifier
        quality_modifier = self._assess_content_quality(source.content_snippet)

        # Apply freshness modifier
        freshness_modifier = self._assess_freshness(source.publication_date)

        # Apply context relevance modifier
        context_modifier = self._assess_context_relevance(source, context_claim)

        # Apply verdict consistency modifier
        consistency_modifier = self._assess_verdict_consistency(source)

        # Apply claim type modifier (reduce credibility for news sources on statistical claims)
        claim_type_modifier = self._assess_claim_type_modifier(source, context_claim)

        # Calculate final score
        final_score = (base_score * 0.4 +           # 40% base reputation
                      quality_modifier * 0.25 +     # 25% content quality
                      freshness_modifier * 0.15 +   # 15% freshness
                      context_modifier * 0.10 +     # 10% context relevance
                      consistency_modifier * 0.10)  # 10% verdict consistency

        # Apply claim type modifier
        final_score *= claim_type_modifier

        return max(0.0, min(1.0, final_score))

    def _get_base_score(self, source) -> float:
        """Get base credibility score from source reputation (integrated method)."""
        # Try exact source name match first
        source_key = source.source_name.lower()
        if source_key in self.source_base_scores:
            return self.source_base_scores[source_key]

        # Try URL domain matching
        if source.url:
            domain = self._extract_domain(source.url)
            if domain in self.source_base_scores:
                return self.source_base_scores[domain]

            # Try top-level domain matching
            tld = self._extract_tld(domain)
            if tld in self.source_base_scores:
                return self.source_base_scores[tld]

        return self.source_base_scores['unknown']

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL (integrated method with caching)."""
        if url in self._domain_cache:
            return self._domain_cache[url]

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix for better matching
            if domain.startswith('www.'):
                domain = domain[4:]
        except:
            domain = ""

        self._domain_cache[url] = domain
        return domain

    def _extract_tld(self, domain: str) -> str:
        """Extract top-level domain from domain (integrated method with caching)."""
        if domain in self._tld_cache:
            return self._tld_cache[domain]

        if '.' not in domain:
            tld = 'unknown'
        else:
            parts = domain.split('.')
            if len(parts) >= 2:
                tld = parts[-1]  # Last part (com, es, org, etc.)
            else:
                tld = 'unknown'

        self._tld_cache[domain] = tld
        return tld

    def _assess_content_quality(self, content: str) -> float:
        """Assess content quality based on indicators (integrated method with optimized lookup)."""
        if not content:
            return 0.8  # Neutral for missing content

        content_lower = content.lower()

        # Count quality indicators using set intersection for better performance
        high_quality_count = len(self._high_quality_set & set(content_lower.split()))
        medium_quality_count = len(self._medium_quality_set & set(content_lower.split()))
        low_quality_count = len(self._low_quality_set & set(content_lower.split()))

        # Calculate quality score
        quality_score = (high_quality_count * 2 + medium_quality_count * 1 -
                        low_quality_count * 0.5)

        # Convert to modifier (0.5 to 1.5 range)
        if quality_score >= 3:
            return 1.4  # High quality
        elif quality_score >= 1:
            return 1.2  # Good quality
        elif quality_score >= 0:
            return 1.0  # Neutral
        elif quality_score >= -1:
            return 0.8  # Low quality
        else:
            return 0.6  # Poor quality

    def _assess_freshness(self, publication_date: Optional[datetime]) -> float:
        """Assess how fresh the content is (integrated method)."""
        if not publication_date:
            return 0.85  # Slight penalty for unknown dates

        # Ensure both datetimes are naive (no timezone) for comparison
        now = datetime.now()
        if publication_date.tzinfo is not None:
            # Convert aware datetime to naive (assume it's in local timezone)
            publication_date = publication_date.replace(tzinfo=None)
        elif now.tzinfo is not None:
            # This shouldn't happen, but handle it just in case
            now = now.replace(tzinfo=None)

        days_old = (now - publication_date).days

        if days_old <= 7:
            return 1.0    # Very fresh
        elif days_old <= 30:
            return 0.95   # Fresh
        elif days_old <= 90:
            return 0.90   # Recent
        elif days_old <= 365:
            return 0.80   # Within a year
        else:
            # Gradual decay
            return max(0.7, 1.0 - (days_old - 365) / 3650)  # Min 0.7 after decay

    def _assess_context_relevance(self, source, claim: str) -> float:
        """Assess how relevant the source is to the claim (integrated method)."""
        if not claim:
            return 1.0  # Neutral if no claim context

        claim_lower = claim.lower()
        content_lower = (source.content_snippet or "").lower()
        title_lower = (source.title or "").lower()

        # Check for keyword overlap using pre-compiled pattern
        claim_words = set(self.word_pattern.findall(claim_lower))
        content_words = set(self.word_pattern.findall(content_lower))
        title_words = set(self.word_pattern.findall(title_lower))

        # Calculate overlap ratios
        content_overlap = len(claim_words & content_words) / len(claim_words) if claim_words else 0
        title_overlap = len(claim_words & title_words) / len(claim_words) if claim_words else 0

        # Combined relevance score
        relevance_score = (content_overlap * 0.7 + title_overlap * 0.3)

        # Convert to modifier
        if relevance_score >= 0.5:
            return 1.2   # Highly relevant
        elif relevance_score >= 0.3:
            return 1.1   # Relevant
        elif relevance_score >= 0.1:
            return 1.0   # Somewhat relevant
        else:
            return 0.9   # Low relevance

    def _assess_verdict_consistency(self, source) -> float:
        """Assess verdict consistency and confidence (integrated method)."""
        # Higher confidence sources get slight boost
        if source.confidence >= 0.8:
            return 1.05
        elif source.confidence >= 0.6:
            return 1.02
        elif source.confidence >= 0.4:
            return 1.0
        else:
            return 0.95  # Low confidence penalty

    def _assess_claim_type_modifier(self, source, context_claim: str) -> float:
        """Assess claim type modifier (integrated method with optimized lookup)."""
        if not context_claim:
            return 1.0  # No modification if no claim context

        claim_lower = context_claim.lower()
        source_name_lower = source.source_name.lower()

        # Check if this is a statistical/economic claim using pre-computed set
        is_statistical_claim = any(keyword in claim_lower for keyword in self._statistical_keywords)

        if is_statistical_claim:
            # For statistical claims, reduce credibility of news sources (not authoritative)
            # Check if source is a news outlet using pre-computed set
            is_news_source = any(domain in source_name_lower for domain in self._news_domains)

            if is_news_source or source.source_type == 'news':
                return 0.3  # Reduce credibility by 70% for news sources on statistical claims
            elif source.source_type == 'web_search':
                return 0.6  # Reduce credibility by 40% for web search on statistical claims

        return 1.0  # No modification for other cases

    def _batch_score_sources(self, sources, context_claim: str = "") -> List:
        """Score multiple sources efficiently (integrated method)."""
        for source in sources:
            source.credibility_score = self._score_source(source, context_claim)

        return sources

    async def _batch_score_sources_async(self, sources, context_claim: str = "") -> List:
        """Score multiple sources asynchronously for better performance."""
        # Process sources in parallel using asyncio
        tasks = []
        for source in sources:
            # Create a task for each source scoring operation
            task = asyncio.create_task(self._score_source_async(source, context_claim))
            tasks.append(task)

        # Wait for all scoring tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        return sources

    async def _score_source_async(self, source, context_claim: str = "") -> None:
        """Calculate overall credibility score asynchronously."""
        # This is now async but the actual computation is still CPU-bound
        # In a real optimization, expensive operations would be moved to threads
        source.credibility_score = self._score_source(source, context_claim)


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

    verifier = ClaimVerifier()
    return await verifier.verify_content(context)