"""
Unit tests for integration components.
Tests analyzer hooks and cross-component integration.
"""

import pytest

from retrieval.integration.analyzer_hooks import AnalyzerHooks
from retrieval.core.models import VerificationResult, VerificationVerdict


class TestAnalyzerHooks:
    """Test analyzer integration hooks."""

    def setup_method(self):
        self.hooks = AnalyzerHooks()

    def test_hooks_initialization(self):
        """Test that hooks initialize properly."""
        assert self.hooks.verifier is not None

    def test_should_trigger_verification(self):
        """Test when verification should be triggered."""
        # High confidence disinformation should trigger
        should_trigger, reason = self.hooks.should_trigger_verification(
            "Hay 100 millones de españoles según fuentes secretas",
            {
                "category": "disinformation",
                "confidence": 0.9
            }
        )
        assert should_trigger

        # Low confidence general content should not trigger
        should_trigger, reason = self.hooks.should_trigger_verification(
            "Esto es contenido normal",
            {
                "category": "general",
                "confidence": 0.3
            }
        )
        assert not should_trigger

    @pytest.mark.asyncio
    async def test_analyze_with_verification(self):
        """Test analyzer result verification."""
        analyzer_result = {
            "category": "disinformation",
            "confidence": 0.8,
            "explanation": "Contenido potencialmente falso"
        }

        result = await self.hooks.analyze_with_verification(
            "Hay 100 millones de españoles según fuentes secretas",
            analyzer_result
        )

        assert result.original_result["category"] == "disinformation"
        assert result.verification_data.get('verification_confidence', 0) >= 0.0
        assert len(result.explanation_with_verification) > 0