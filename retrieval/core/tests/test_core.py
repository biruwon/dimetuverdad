"""
Unit tests for core retrieval components.
Tests claim extraction, query building, evidence aggregation, and caching.
"""

import pytest
from datetime import datetime

from retrieval.core.claim_extractor import ClaimExtractor, ClaimType, VerificationTarget
from retrieval.core.models import VerificationResult, EvidenceSource, VerificationVerdict
from retrieval.core.evidence_aggregator import EvidenceAggregator
from retrieval.core.query_builder import QueryBuilder


class TestClaimExtractor:
    """Test claim extraction functionality."""

    def setup_method(self):
        self.extractor = ClaimExtractor()

    def test_extract_numerical_claims(self):
        """Test extraction of numerical claims."""
        text = "Según datos oficiales, hay 47 millones de españoles y el 15% está vacunado."
        claims = self.extractor.extract_claims(text)

        assert len(claims) >= 2
        numerical_claims = [c for c in claims if c.claim_type == ClaimType.NUMERICAL]
        assert len(numerical_claims) >= 1

        # Check that some claims contain numbers
        has_numbers = any(any(char.isdigit() for char in claim.claim_text) for claim in numerical_claims)
        assert has_numbers

    def test_extract_temporal_claims(self):
        """Test extraction of temporal claims."""
        text = "Hace 3 días ocurrió el evento. En 2020 empezó la pandemia."
        claims = self.extractor.extract_claims(text)

        temporal_claims = [c for c in claims if c.claim_type == ClaimType.TEMPORAL]
        assert len(temporal_claims) >= 1

    def test_extract_attribution_claims(self):
        """Test extraction of attribution claims."""
        text = "Según el Ministerio de Sanidad, los casos han aumentado un 20%."
        claims = self.extractor.extract_claims(text)

        attribution_claims = [c for c in claims if c.claim_type == ClaimType.ATTRIBUTION]
        assert len(attribution_claims) >= 1


class TestEvidenceAggregator:
    """Test evidence aggregation functionality."""

    def setup_method(self):
        self.aggregator = EvidenceAggregator()

    def test_aggregate_evidence_verified(self):
        """Test aggregation when evidence supports the claim."""
        claim = VerificationTarget(
            claim_text="Hay 47 millones de españoles",
            claim_type=ClaimType.NUMERICAL,
            context="Según datos oficiales",
            confidence=0.8,
            priority=5,
            extracted_value="47",
            start_pos=0,
            end_pos=25
        )

        sources = [
            EvidenceSource(
                source_name="INE",
                source_type="statistical_agency",
                url="https://ine.es",
                title="Población española: 47.4 millones",
                credibility_score=0.95,
                publication_date=datetime.now(),
                content_snippet="Población española: 47.4 millones",
                verdict_contribution=VerificationVerdict.VERIFIED,
                confidence=0.9
            )
        ]

        result = self.aggregator.aggregate_evidence(str(claim.claim_text), sources)

        assert result.verdict in [VerificationVerdict.VERIFIED, VerificationVerdict.QUESTIONABLE]
        assert result.confidence > 0.5

    def test_aggregate_evidence_contradictory(self):
        """Test aggregation with contradictory evidence."""
        claim = VerificationTarget(
            claim_text="Hay 100 millones de españoles",
            claim_type=ClaimType.NUMERICAL,
            context="Según una fuente dudosa",
            confidence=0.8,
            priority=5,
            extracted_value="100",
            start_pos=0,
            end_pos=30
        )

        sources = [
            EvidenceSource(
                source_name="INE",
                source_type="statistical_agency",
                url="https://ine.es",
                title="Población española: 47.4 millones",
                credibility_score=0.95,
                publication_date=datetime.now(),
                content_snippet="Población española: 47.4 millones",
                verdict_contribution=VerificationVerdict.DEBUNKED,
                confidence=0.9
            ),
            EvidenceSource(
                source_name="Fuente dudosa",
                source_type="unknown",
                url="https://dudoso.com",
                title="Población española: 100 millones",
                credibility_score=0.3,
                publication_date=datetime.now(),
                content_snippet="Población española: 100 millones",
                verdict_contribution=VerificationVerdict.VERIFIED,
                confidence=0.5
            )
        ]

        result = self.aggregator.aggregate_evidence(str(claim.claim_text), sources)

        # Should be debunked due to high-credibility source outweighing low-credibility source
        assert result.verdict == VerificationVerdict.DEBUNKED
        assert "desmentida" in result.explanation.lower()  # Should mention debunked

    def test_evidence_aggregator_no_sources(self):
        """Test evidence aggregator with no sources."""
        result = self.aggregator.aggregate_evidence("Test claim", [])

        assert result.verdict == VerificationVerdict.UNVERIFIED
        assert result.confidence == 0.0
        assert "No evidence sources found" in result.explanation

    def test_evidence_aggregator_resolve_conflicts(self):
        """Test conflict resolution in evidence aggregator."""
        sources = [
            EvidenceSource(
                source_name="Source A",
                source_type="news",
                url="https://a.com",
                title="Title A",
                credibility_score=0.8,
                verdict_contribution=VerificationVerdict.VERIFIED
            ),
            EvidenceSource(
                source_name="Source B",
                source_type="news",
                url="https://b.com",
                title="Title B",
                credibility_score=0.7,
                verdict_contribution=VerificationVerdict.DEBUNKED
            )
        ]

        resolved = self.aggregator.resolve_conflicts(sources)
        # Should keep both since no clear majority
        assert len(resolved) == 2


class TestQueryBuilder:
    """Test query building functionality."""

    def setup_method(self):
        self.builder = QueryBuilder()

    def test_build_queries_spanish(self):
        """Test query building for Spanish content."""
        claim_text = "La población española es de 47 millones"
        queries = self.builder.build_fact_checking_queries(claim_text)

        assert len(queries) > 0
        assert any("población" in q.lower() for q in queries)
        assert any("español" in q.lower() for q in queries)

    def test_query_diversification(self):
        """Test that queries are diversified."""
        claim_text = "El PIB de España creció un 2%"
        queries = self.builder.build_fact_checking_queries(claim_text)

        # Should generate multiple different queries
        assert len(queries) > 1
        assert len(set(queries)) == len(queries)  # All unique

    def test_query_builder_key_terms(self):
        """Test key term extraction."""
        text = "La población española ha crecido significativamente en los últimos años."
        terms = self.builder._extract_key_terms(text)

        assert len(terms) > 0
        assert "población" in terms
        assert "española" in terms


if __name__ == "__main__":
    # Run basic functionality tests
    print("Running basic functionality tests...")

    # Test claim extraction
    extractor = ClaimExtractor()
    claims = extractor.extract_claims("Según datos oficiales, hay 47 millones de españoles.")
    print(f"Extracted {len(claims)} claims")

    # Test evidence aggregation
    aggregator = EvidenceAggregator()
    sources = [
        EvidenceSource(
            source_name="INE",
            source_type="statistical_agency",
            url="https://ine.es",
            title="Población española: 47.4 millones",
            credibility_score=0.95,
            publication_date=datetime.now(),
            content_snippet="Población española: 47.4 millones",
            verdict_contribution=VerificationVerdict.VERIFIED,
            confidence=0.9
        )
    ]
    result = aggregator.aggregate_evidence("Test claim", sources)
    print(f"Aggregated evidence: {result.verdict}")

    # Test query building
    builder = QueryBuilder()
    queries = builder.build_fact_checking_queries("La población española es de 47 millones")
    print(f"Built {len(queries)} queries")

    print("Basic tests completed successfully!")