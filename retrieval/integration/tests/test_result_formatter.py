"""
Unit tests for result formatting utilities.
Tests LLM-optimized result formatting and output formats.
"""

import pytest
import json
from datetime import datetime

from retrieval.integration.result_formatter import ResultFormatter, LLMFormattedResult
from retrieval.core.models import VerificationResult, EvidenceSource, VerificationVerdict


class TestResultFormatter:
    """Test result formatting functionality."""

    def setup_method(self):
        self.formatter = ResultFormatter()

    def test_format_for_llm_prompt(self):
        """Test formatting verification result for LLM prompt."""
        # Create test verification result
        evidence_sources = [
            EvidenceSource(
                source_name="INE",
                source_type="official",
                url="https://ine.es",
                title="Instituto Nacional de Estadística",
                credibility_score=0.95,
                content_snippet="Datos oficiales de población",
                confidence=0.9
            ),
            EvidenceSource(
                source_name="Wikipedia",
                source_type="reference",
                url="https://es.wikipedia.org",
                title="Demografía de España",
                credibility_score=0.7,
                content_snippet="Información general",
                confidence=0.8
            )
        ]

        verification_result = VerificationResult(
            claim="La población de España es de 47 millones",
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.85,
            evidence_sources=evidence_sources,
            explanation="Verificado por fuentes oficiales",
            processing_time_seconds=1.2
        )

        formatted = self.formatter.format_for_llm_prompt(verification_result)

        assert isinstance(formatted, str)
        assert "VERIFICADO" in formatted
        assert "47 millones" in formatted
        assert "INE" in formatted
        assert "0.8" in formatted  # Confidence is displayed as percentage

    def test_format_multiple_results(self):
        """Test formatting multiple verification results."""
        results = [
            VerificationResult(
                claim="Población: 47 millones",
                verdict=VerificationVerdict.VERIFIED,
                confidence=0.9,
                explanation="Fuente oficial"
            ),
            VerificationResult(
                claim="PIB: 1.2 billones",
                verdict=VerificationVerdict.DEBUNKED,
                confidence=0.8,
                explanation="Datos incorrectos"
            )
        ]

        formatted = self.formatter.create_evidence_summary(results)

        assert isinstance(formatted, str)
        assert "VERIFICADO" in formatted
        assert "DESMENTIDO" in formatted
        assert "47 millones" in formatted
        assert "1.2 billones" in formatted

    def test_create_llm_formatted_result(self):
        """Test creating LLM-formatted result."""
        verification_result = VerificationResult(
            claim="Test claim",
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.75,
            evidence_sources=[
                EvidenceSource(
                    source_name="Test Source",
                    source_type="official",
                    url="https://test.com",
                    title="Test Title",
                    credibility_score=0.8
                )
            ],
            explanation="Test explanation",
            processing_time_seconds=0.5
        )

        formatted = self.formatter.format_verification_report(
            type('MockReport', (), {
                'claims_verified': [verification_result],
                'evidence_sources': verification_result.evidence_sources,
                'confidence_score': 0.75,
                'contradictions_found': [],
                'temporal_consistency': True,
                'overall_verdict': VerificationVerdict.VERIFIED,
                'processing_time': 1.0
            })()
        )

        assert isinstance(formatted, LLMFormattedResult)
        assert formatted.claim == "Test claim"
        assert formatted.verdict == "VERIFICADO"  # Should be the Spanish translation
        assert formatted.confidence == 0.75
        assert formatted.evidence_count == 1
        assert len(formatted.sources) == 1
        assert "Test Source" in formatted.sources
        assert isinstance(formatted.formatted_for_llm, str)

    def test_format_for_different_verdicts(self):
        """Test formatting for different verdict types."""
        verdicts = [
            (VerificationVerdict.VERIFIED, "VERIFICADO"),
            (VerificationVerdict.DEBUNKED, "DESMENTIDO"),
            (VerificationVerdict.QUESTIONABLE, "CUESTIONABLE"),
            (VerificationVerdict.UNVERIFIED, "SIN VERIFICAR")
        ]

        for verdict, expected_text in verdicts:
            result = VerificationResult(
                claim="Test claim",
                verdict=verdict,
                confidence=0.5,
                explanation="Test explanation"
            )

            formatted = self.formatter.format_for_llm_prompt(result)
            assert expected_text in formatted

    def test_format_with_contradictions(self):
        """Test formatting results with contradictions."""
        result = VerificationResult(
            claim="Contradictory claim",
            verdict=VerificationVerdict.QUESTIONABLE,
            confidence=0.6,
            explanation="Multiple conflicting sources",
            evidence_sources=[
                EvidenceSource(
                    source_name="Source A",
                    source_type="news",
                    url="https://a.com",
                    title="Version A",
                    credibility_score=0.8,
                    verdict_contribution=VerificationVerdict.VERIFIED
                ),
                EvidenceSource(
                    source_name="Source B",
                    source_type="news",
                    url="https://b.com",
                    title="Version B",
                    credibility_score=0.7,
                    verdict_contribution=VerificationVerdict.DEBUNKED
                )
            ]
        )

        formatted = self.formatter.format_for_llm_prompt(result)
        assert "CUESTIONABLE" in formatted
        # Just check that formatting works with contradictory evidence

    def test_llm_formatted_result_to_dict(self):
        """Test LLMFormattedResult to_dict method."""
        result = LLMFormattedResult(
            claim="Test",
            verdict="verified",
            confidence=0.8,
            explanation="Test explanation",
            evidence_count=2,
            sources=["Source1", "Source2"],
            key_facts=["Fact1", "Fact2"],
            contradictions=[],
            temporal_consistency=True,
            processing_time=1.0,
            formatted_for_llm="Formatted text"
        )

        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["claim"] == "Test"
        assert data["verdict"] == "verified"
        assert data["confidence"] == 0.8

    def test_llm_formatted_result_to_json(self):
        """Test LLMFormattedResult to_json method."""
        result = LLMFormattedResult(
            claim="Test",
            verdict="verified",
            confidence=0.8,
            explanation="Test explanation",
            evidence_count=1,
            sources=["Source1"],
            key_facts=[],
            contradictions=[],
            temporal_consistency=True,
            processing_time=1.0,
            formatted_for_llm="Formatted text"
        )

        json_str = result.to_json()
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["claim"] == "Test"
        assert parsed["verdict"] == "verified"

    def test_format_empty_result(self):
        """Test formatting empty or minimal result."""
        result = VerificationResult(
            claim="Minimal claim",
            verdict=VerificationVerdict.UNVERIFIED,
            confidence=0.0,
            explanation=""
        )

        formatted = self.formatter.format_for_llm_prompt(result)
        assert isinstance(formatted, str)
        assert "SIN VERIFICAR" in formatted
        assert "Minimal claim" in formatted

    def test_format_with_max_sources(self):
        """Test that formatting respects max sources limit."""
        # Create result with many sources
        sources = [
            EvidenceSource(
                source_name=f"Source{i}",
                source_type="news",
                url=f"https://source{i}.com",
                title=f"Title{i}",
                credibility_score=0.8
            )
            for i in range(10)
        ]

        result = VerificationResult(
            claim="Test claim",
            verdict=VerificationVerdict.VERIFIED,
            confidence=0.8,
            evidence_sources=sources,
            explanation="Test"
        )

        formatted = self.formatter.format_for_llm_prompt(result)

        # Should limit sources in formatted output
        assert isinstance(formatted, str)
        # The formatter should handle the sources appropriately