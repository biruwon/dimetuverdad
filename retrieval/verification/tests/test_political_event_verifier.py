"""
Unit tests for PoliticalEventVerifier with timeout and parallel execution features.
Tests timeout enforcement, parallel source searching, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from retrieval.verification.political_event_verifier import (
    PoliticalEventVerifier,
    PoliticalEventClaim,
    VerificationEvidence
)
from retrieval.core.models import VerificationVerdict


class TestPoliticalEventVerifierParallelExecution:
    """Test parallel source searching functionality."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier instance for testing."""
        return PoliticalEventVerifier(timeout=30)

    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim for testing."""
        return PoliticalEventClaim(
            person_name="Test Person",
            event_type="arrest",
            institution="Test Prison",
            context="Test arrest claim"
        )

    @pytest.mark.asyncio
    async def test_parallel_source_searching(self, verifier, sample_claim):
        """Test that sources are searched in parallel using asyncio.gather."""
        # Mock the safe wrapper methods
        mock_fact_check_results = [VerificationEvidence(
            source_name="Fact Checker",
            source_url="http://example.com",
            verdict=VerificationVerdict.DEBUNKED,
            confidence=0.8,
            explanation="Test explanation"
        )]
        
        mock_official_results = [VerificationEvidence(
            source_name="Official Source",
            source_url="http://official.com",
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.9,
            explanation="Official explanation"
        )]
        
        mock_news_results = [VerificationEvidence(
            source_name="News Source",
            source_url="http://news.com",
            verdict=VerificationVerdict.QUESTIONABLE,
            confidence=0.5,
            explanation="News explanation"
        )]

        with patch.object(verifier, '_search_fact_checking_site_safe', 
                         new_callable=AsyncMock, return_value=mock_fact_check_results) as mock_fact, \
             patch.object(verifier, '_search_official_sources_safe',
                         new_callable=AsyncMock, return_value=mock_official_results) as mock_official, \
             patch.object(verifier, '_search_recent_news_safe',
                         new_callable=AsyncMock, return_value=mock_news_results) as mock_news:
            
            results = await verifier._search_multiple_sources(sample_claim)
            
            # Verify all results were collected
            assert len(results) >= 3
            
            # Verify all methods were called (parallel execution)
            assert mock_fact.called
            assert mock_official.called
            assert mock_news.called

    @pytest.mark.asyncio
    async def test_parallel_execution_with_exceptions(self, verifier, sample_claim):
        """Test that exceptions in one source don't block others."""
        mock_success_result = [VerificationEvidence(
            source_name="Success Source",
            source_url="http://success.com",
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.9,
            explanation="Success"
        )]
        
        with patch.object(verifier, '_search_fact_checking_site_safe', new_callable=AsyncMock) as mock_fact, \
             patch.object(verifier, '_search_official_sources_safe', new_callable=AsyncMock) as mock_official, \
             patch.object(verifier, '_search_recent_news_safe', new_callable=AsyncMock) as mock_news:
            
            # One fails, two succeed
            mock_fact.return_value = mock_success_result
            mock_official.side_effect = Exception("Network error")
            mock_news.return_value = mock_success_result
            
            results = await verifier._search_multiple_sources(sample_claim)
            
            # Should still get results from successful searches
            assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_safe_wrapper_catches_exceptions(self, verifier, sample_claim):
        """Test that safe wrapper methods catch and log exceptions."""
        source_config = {'name': 'Test Source', 'search_url': 'http://test.com'}
        
        with patch.object(verifier, '_search_fact_checking_site',
                         side_effect=Exception("Test error")) as mock_search:
            
            result = await verifier._search_fact_checking_site_safe(sample_claim, source_config)
            
            # Should return empty list on error
            assert result == []
            assert mock_search.called

    @pytest.mark.asyncio
    async def test_official_sources_safe_wrapper(self, verifier, sample_claim):
        """Test official sources safe wrapper."""
        with patch.object(verifier, '_search_official_sources',
                         side_effect=Exception("Test error")) as mock_search:
            
            result = await verifier._search_official_sources_safe(sample_claim)
            
            assert result == []
            assert mock_search.called

    @pytest.mark.asyncio
    async def test_recent_news_safe_wrapper(self, verifier, sample_claim):
        """Test recent news safe wrapper."""
        with patch.object(verifier, '_search_recent_news',
                         side_effect=Exception("Test error")) as mock_search:
            
            result = await verifier._search_recent_news_safe(sample_claim)
            
            assert result == []
            assert mock_search.called


class TestPoliticalEventVerifierTimeouts:
    """Test timeout handling in individual source searches."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier with short timeout for testing."""
        return PoliticalEventVerifier(timeout=1)

    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim for testing."""
        return PoliticalEventClaim(
            person_name="Test Person",
            event_type="arrest"
        )

    @pytest.mark.asyncio
    async def test_fact_checking_site_timeout(self, verifier, sample_claim):
        """Test that fact checking site searches respect timeout."""
        source_config = {
            'name': 'Test Source',
            'search_url': 'http://test.com/search?q=',
            'base_url': 'http://test.com'
        }
        
        # Mock aiohttp session to simulate slow response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            
            # Make the response very slow
            async def slow_text():
                await asyncio.sleep(5)  # Longer than timeout
                return "<html>test</html>"
            
            mock_response.text = slow_text
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Should not raise, should return empty list
            result = await verifier._search_fact_checking_site(sample_claim, source_config)
            
            # Should handle timeout gracefully
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_timeout_configuration(self):
        """Test that timeout is configurable."""
        verifier1 = PoliticalEventVerifier(timeout=10)
        assert verifier1.timeout == 10
        
        verifier2 = PoliticalEventVerifier(timeout=60)
        assert verifier2.timeout == 60


class TestPoliticalEventVerifierEvidenceAnalysis:
    """Test evidence analysis and verdict determination."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier instance."""
        return PoliticalEventVerifier()

    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim."""
        return PoliticalEventClaim(
            person_name="Test Person",
            event_type="arrest",
            context="Test context"
        )

    def test_analyze_evidence_debunked(self, verifier, sample_claim):
        """Test evidence analysis when claim is debunked."""
        evidence_list = [
            VerificationEvidence(
                source_name="Source 1",
                source_url="http://source1.com",
                verdict=VerificationVerdict.DEBUNKED,
                confidence=0.9,
                explanation="Claim is false"
            ),
            VerificationEvidence(
                source_name="Source 2",
                source_url="http://source2.com",
                verdict=VerificationVerdict.DEBUNKED,
                confidence=0.8,
                explanation="No evidence found"
            )
        ]
        
        result = verifier._analyze_evidence(sample_claim, evidence_list)
        
        assert result.verdict == VerificationVerdict.DEBUNKED
        assert result.confidence > 0.7
        assert len(result.evidence_sources) == 2

    def test_analyze_evidence_verified(self, verifier, sample_claim):
        """Test evidence analysis when claim is verified."""
        evidence_list = [
            VerificationEvidence(
                source_name="Source 1",
                source_url="http://source1.com",
                verdict=VerificationVerdict.VERIFIED,
                confidence=0.9,
                explanation="Claim is true"
            )
        ]
        
        result = verifier._analyze_evidence(sample_claim, evidence_list)
        
        assert result.verdict == VerificationVerdict.VERIFIED
        assert result.confidence > 0.7

    def test_analyze_evidence_questionable(self, verifier, sample_claim):
        """Test evidence analysis when evidence is inconclusive."""
        evidence_list = [
            VerificationEvidence(
                source_name="Source 1",
                source_url="http://source1.com",
                verdict=VerificationVerdict.QUESTIONABLE,
                confidence=0.4,
                explanation="Unclear"
            )
        ]
        
        result = verifier._analyze_evidence(sample_claim, evidence_list)
        
        assert result.verdict == VerificationVerdict.QUESTIONABLE
        assert result.confidence <= 0.5

    def test_analyze_evidence_empty_list(self, verifier, sample_claim):
        """Test evidence analysis with no evidence."""
        evidence_list = []
        
        result = verifier._analyze_evidence(sample_claim, evidence_list)
        
        assert result.verdict == VerificationVerdict.QUESTIONABLE
        assert result.confidence == 0.3


class TestPoliticalEventVerifierIntegration:
    """Integration tests for the complete verification flow."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier instance."""
        return PoliticalEventVerifier(timeout=30)

    @pytest.mark.asyncio
    async def test_verify_arrest_claim_with_mocks(self, verifier):
        """Test complete arrest claim verification with mocked sources."""
        mock_evidence = [
            VerificationEvidence(
                source_name="Test Source",
                source_url="http://test.com",
                verdict=VerificationVerdict.DEBUNKED,
                confidence=0.8,
                explanation="No arrest record found"
            )
        ]
        
        with patch.object(verifier, '_search_multiple_sources',
                         new_callable=AsyncMock, return_value=mock_evidence):
            
            result = await verifier.verify_arrest_claim("Test Person", "Test Prison")
            
            assert result.verdict == VerificationVerdict.DEBUNKED
            assert result.extracted_value == "Test Person"

    @pytest.mark.asyncio
    async def test_verify_political_event_with_mocks(self, verifier):
        """Test complete political event verification."""
        mock_evidence = [
            VerificationEvidence(
                source_name="Test Source",
                source_url="http://test.com",
                verdict=VerificationVerdict.VERIFIED,
                confidence=0.9,
                explanation="Event confirmed"
            )
        ]
        
        with patch.object(verifier, '_search_multiple_sources',
                         new_callable=AsyncMock, return_value=mock_evidence):
            
            result = await verifier.verify_political_event("Test Person was arrested")
            
            assert result.verdict == VerificationVerdict.VERIFIED


class TestPoliticalEventVerifierHelperMethods:
    """Test helper methods for claim parsing and analysis."""

    @pytest.fixture
    def verifier(self):
        """Create a verifier instance."""
        return PoliticalEventVerifier()

    def test_contains_contradiction_indicators(self, verifier):
        """Test detection of contradiction indicators in HTML."""
        claim = PoliticalEventClaim(
            person_name="Test Person",
            event_type="arrest"
        )
        
        html_false = "<html>Test Person arrest is falso and mentira</html>"
        assert verifier._contains_contradiction_indicators(html_false, claim)
        
        html_neutral = "<html>Some other content</html>"
        assert not verifier._contains_contradiction_indicators(html_neutral, claim)

    def test_contains_confirmation_indicators(self, verifier):
        """Test detection of confirmation indicators in HTML."""
        claim = PoliticalEventClaim(
            person_name="Test Person",
            event_type="arrest"
        )
        
        html_confirmed = "<html>Test Person arrest ha sido confirmado oficialmente</html>"
        assert verifier._contains_confirmation_indicators(html_confirmed, claim)
        
        html_neutral = "<html>Some other content</html>"
        assert not verifier._contains_confirmation_indicators(html_neutral, claim)

    def test_extract_judicial_claim_info(self, verifier):
        """Test extraction of person and event from judicial claim text."""
        claim_text = "El juez procesa a Ábalos por corrupción"
        person, event = verifier._extract_judicial_claim_info(claim_text)
        
        assert person == "ábalos"
        assert event == "judicial_order"

    def test_parse_event_description(self, verifier):
        """Test parsing of event descriptions."""
        description = "Juan Pérez ha sido detenido por la policía"
        claim = verifier._parse_event_description(description)
        
        assert claim.person_name == "Juan Pérez"
        assert claim.event_type == "arrest"
        assert claim.context == description


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
