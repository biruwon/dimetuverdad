"""
Source credibility assessment functionality.
Evaluates the trustworthiness and reliability of information sources.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re

from ..core.models import EvidenceSource


class CredibilityScorer:
    """
    Assesses the credibility of information sources based on multiple factors.
    Provides weighted scoring for evidence aggregation.
    """

    def __init__(self):
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

        # Content quality indicators
        self.quality_indicators = {
            'high': ['estudio científico', 'investigación', 'datos oficiales', 'informe oficial',
                    'evidencia empírica', 'análisis estadístico', 'fuentes primarias'],
            'medium': ['análisis', 'investigación', 'datos', 'informe', 'estudio',
                      'verificación', 'comprobación'],
            'low': ['opinión', 'creo que', 'parece', 'probablemente', 'quizás']
        }

        # Bias indicators (reduce credibility)
        self.bias_indicators = [
            'teoría conspirativa', 'fake news', 'desinformación',
            'engaño', 'manipulación', 'propaganda'
        ]

    def score_source(self, source: EvidenceSource, context_claim: str = "") -> float:
        """
        Calculate overall credibility score for a source.

        Args:
            source: The evidence source to score
            context_claim: The claim being verified (for context-aware scoring)

        Returns:
            Credibility score between 0.0 and 1.0
        """
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

    def _get_base_score(self, source: EvidenceSource) -> float:
        """Get base credibility score from source reputation."""
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
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix for better matching
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ""

    def _extract_tld(self, domain: str) -> str:
        """Extract top-level domain from domain."""
        if '.' not in domain:
            return 'unknown'

        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[-1]  # Last part (com, es, org, etc.)
        return 'unknown'

    def _assess_content_quality(self, content: str) -> float:
        """
        Assess content quality based on indicators.
        Returns modifier between 0.5 and 1.5
        """
        if not content:
            return 0.8  # Neutral for missing content

        content_lower = content.lower()

        # Count quality indicators
        high_quality_count = sum(1 for indicator in self.quality_indicators['high']
                               if indicator in content_lower)
        medium_quality_count = sum(1 for indicator in self.quality_indicators['medium']
                                 if indicator in content_lower)
        low_quality_count = sum(1 for indicator in self.quality_indicators['low']
                              if indicator in content_lower)

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
        """
        Assess how fresh the content is.
        Returns modifier between 0.7 and 1.0
        """
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

    def _assess_context_relevance(self, source: EvidenceSource, claim: str) -> float:
        """
        Assess how relevant the source is to the claim.
        Returns modifier between 0.8 and 1.2
        """
        if not claim:
            return 1.0  # Neutral if no claim context

        claim_lower = claim.lower()
        content_lower = (source.content_snippet or "").lower()
        title_lower = (source.title or "").lower()

        # Check for keyword overlap
        claim_words = set(re.findall(r'\w{4,}', claim_lower))
        content_words = set(re.findall(r'\w{4,}', content_lower))
        title_words = set(re.findall(r'\w{4,}', title_lower))

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

    def _assess_verdict_consistency(self, source: EvidenceSource) -> float:
        """
        Assess verdict consistency and confidence.
        Returns modifier between 0.9 and 1.1
        """
        # Higher confidence sources get slight boost
        if source.confidence >= 0.8:
            return 1.05
        elif source.confidence >= 0.6:
            return 1.02
        elif source.confidence >= 0.4:
            return 1.0
        else:
            return 0.95  # Low confidence penalty

    def _assess_claim_type_modifier(self, source: EvidenceSource, context_claim: str) -> float:
        """
        Assess claim type modifier to reduce credibility for inappropriate sources.
        Returns modifier between 0.3 and 1.0
        """
        if not context_claim:
            return 1.0  # No modification if no claim context

        claim_lower = context_claim.lower()
        source_name_lower = source.source_name.lower()

        # Check if this is a statistical/economic claim
        statistical_keywords = ['pib', 'gdp', 'economía', 'crecimiento', 'paro', 'empleo',
                               'inflación', 'déficit', 'estadística', 'datos oficiales',
                               'tasa de', 'porcentaje', 'millones', 'billones']

        is_statistical_claim = any(keyword in claim_lower for keyword in statistical_keywords)

        if is_statistical_claim:
            # For statistical claims, reduce credibility of news sources (not authoritative)
            news_domains = ['elpais.com', 'elmundo.es', 'abc.es', 'lavanguardia.com',
                           'bbc.com', 'reuters.com', 'apnews.com']

            # Check if source is a news outlet
            is_news_source = any(domain in source_name_lower for domain in news_domains)

            if is_news_source or source.source_type == 'news':
                return 0.3  # Reduce credibility by 70% for news sources on statistical claims
            elif source.source_type == 'web_search':
                return 0.6  # Reduce credibility by 40% for web search on statistical claims

        return 1.0  # No modification for other cases

    def batch_score_sources(self, sources: List[EvidenceSource],
                          context_claim: str = "") -> List[EvidenceSource]:
        """
        Score multiple sources efficiently.

        Args:
            sources: List of sources to score
            context_claim: Context claim for relevance scoring

        Returns:
            Sources with updated credibility scores
        """
        for source in sources:
            source.credibility_score = self.score_source(source, context_claim)

        return sources

    def get_source_reliability_report(self, source: EvidenceSource) -> Dict:
        """
        Generate detailed reliability report for a source.

        Returns:
            Dictionary with scoring breakdown
        """
        base_score = self._get_base_score(source)
        quality_mod = self._assess_content_quality(source.content_snippet)
        freshness_mod = self._assess_freshness(source.publication_date)
        context_mod = self._assess_context_relevance(source, "")
        consistency_mod = self._assess_verdict_consistency(source)

        final_score = (base_score * 0.4 + quality_mod * 0.25 +
                      freshness_mod * 0.15 + context_mod * 0.10 +
                      consistency_mod * 0.10)

        return {
            'source_name': source.source_name,
            'final_score': round(final_score, 3),
            'breakdown': {
                'base_reputation': round(base_score, 3),
                'content_quality': round(quality_mod, 3),
                'freshness': round(freshness_mod, 3),
                'context_relevance': round(context_mod, 3),
                'verdict_consistency': round(consistency_mod, 3)
            },
            'assessment': self._interpret_score(final_score)
        }

    def _interpret_score(self, score: float) -> str:
        """Interpret credibility score into qualitative assessment."""
        if score >= 0.9:
            return "Muy confiable"
        elif score >= 0.8:
            return "Confiable"
        elif score >= 0.7:
            return "Moderadamente confiable"
        elif score >= 0.6:
            return "Baja confiabilidad"
        else:
            return "No confiable"


def score_source_credibility(source: EvidenceSource, context_claim: str = "") -> float:
    """
    Convenience function to score a single source.

    Args:
        source: Source to score
        context_claim: Context claim

    Returns:
        Credibility score
    """
    scorer = CredibilityScorer()
    return scorer.score_source(source, context_claim)


def batch_score_sources(sources: List[EvidenceSource], context_claim: str = "") -> List[EvidenceSource]:
    """
    Convenience function to score multiple sources.

    Args:
        sources: Sources to score
        context_claim: Context claim

    Returns:
        Sources with updated scores
    """
    scorer = CredibilityScorer()
    return scorer.batch_score_sources(sources, context_claim)