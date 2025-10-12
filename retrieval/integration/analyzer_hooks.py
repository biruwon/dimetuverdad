"""
Integration hooks for connecting the retrieval system with the main analyzer.
Provides conditional triggering and analysis capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..core.models import VerificationResult, VerificationVerdict
from ..verification.claim_verifier import ClaimVerifier, VerificationContext, VerificationReport


@dataclass
class AnalysisTrigger:
    """Configuration for when to trigger verification."""
    content_categories: List[str]  # Categories that should trigger verification
    confidence_threshold: float   # Minimum analyzer confidence to trigger
    keywords: List[str]           # Keywords that indicate verification needed
    claim_types: List[str]        # Types of claims to verify
    min_claims: int              # Minimum number of claims to trigger verification


@dataclass
class AnalysisResult:
    """Result of analysis with verification data."""

    def __init__(self, original_result, verification_data=None, explanation_with_verification=None):
        self.original_result = original_result
        self.verification_data = verification_data or {}
        self.explanation_with_verification = explanation_with_verification


class AnalyzerHooks:
    """
    Integration layer between the main analyzer and retrieval system.
    Provides conditional triggering and result verification.
    """

    def __init__(self, verifier: Optional[ClaimVerifier] = None):
        self.verifier = verifier or ClaimVerifier()
        self.logger = logging.getLogger(__name__)

        # Default trigger configuration
        self.default_trigger = AnalysisTrigger(
            content_categories=['disinformation', 'conspiracy_theory', 'far_right_bias'],
            confidence_threshold=0.6,
            keywords=[
                'estadística', 'dato', 'cifra', 'millón', 'porcentaje', '%',
                'según', 'informe', 'estudio', 'investigación',
                'pandemia', 'covid', 'vacuna', 'muerte', 'contagio',
                'elecciones', 'voto', 'partido', 'gobierno',
                'economía', 'paro', 'empleo', 'pib', 'déficit'
            ],
            claim_types=['numerical', 'statistical', 'temporal', 'attribution'],
            min_claims=1
        )

    def should_trigger_verification(self, content: str, analyzer_result: Dict[str, Any],
                                  trigger_config: Optional[AnalysisTrigger] = None) -> Tuple[bool, str]:
        """
        Determine if verification should be triggered based on content and analysis.

        Args:
            content: Original content text
            analyzer_result: Result from main analyzer
            trigger_config: Custom trigger configuration

        Returns:
            (should_trigger, reason)
        """
        trigger = trigger_config or self.default_trigger

        # Check content category
        content_category = analyzer_result.get('category', '')
        if content_category in trigger.content_categories:
            return True, f"Categoría de contenido: {content_category}"

        # Check analyzer confidence
        analyzer_confidence = analyzer_result.get('confidence', 0.0)
        if analyzer_confidence >= trigger.confidence_threshold:
            return True, f"Confianza del analizador: {analyzer_confidence:.2f}"

        # Check for trigger keywords
        content_lower = content.lower()
        found_keywords = [kw for kw in trigger.keywords if kw in content_lower]
        if found_keywords:
            return True, f"Palabras clave encontradas: {', '.join(found_keywords[:3])}"

        # Check for verifiable claims
        from ..core.claim_extractor import ClaimExtractor
        extractor = ClaimExtractor()
        claims = extractor.extract_claims(content)

        # Filter claims by type
        relevant_claims = [c for c in claims if c.claim_type in trigger.claim_types]
        if len(relevant_claims) >= trigger.min_claims:
            return True, f"Afirmaciones verificables encontradas: {len(relevant_claims)}"

        return False, "No se cumplen criterios de activación"

    async def analyze_with_verification(self, content: str, original_result=None,
                             trigger_config: Optional[AnalysisTrigger] = None) -> AnalysisResult:
        """
        Analyze content with verification data.

        Args:
            content: Original content
            original_result: Original analyzer result
            trigger_config: Trigger configuration

        Returns:
            Analysis result with verification data
        """
        should_trigger, reason = self.should_trigger_verification(
            content, original_result, trigger_config
        )

        if not should_trigger:
            # Return original result without verification
            return AnalysisResult(
                original_result=original_result,
                verification_data=None,
                explanation_with_verification=original_result.get('explanation', '')
            )

        try:
            # Perform verification
            verification_context = VerificationContext(
                original_text=content,
                content_category=original_result.get('category', 'general'),
                user_context=f"Analysis category: {original_result.get('category', 'unknown')}",
                language="es",
                priority_level="balanced"
            )

            verification_report = await self.verifier.verify_content(verification_context)

            # Combine the explanation with verification results
            explanation_with_verification = self._combine_explanation(
                original_result.get('explanation', ''),
                verification_report
            )

            # Extract sources and contradictions
            sources_cited = [s.source_name for s in verification_report.evidence_sources[:5]]
            contradictions_detected = verification_report.contradictions_found

            return AnalysisResult(
                original_result=original_result,
                verification_data={
                    'verification_report': verification_report,
                    'sources_cited': sources_cited,
                    'contradictions_detected': contradictions_detected,
                    'verification_confidence': verification_report.confidence_score
                },
                explanation_with_verification=explanation_with_verification
            )

        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            # Return original result on error
            return AnalysisResult(
                original_result=original_result,
                verification_data=None,
                explanation_with_verification=original_result.get('explanation', '')
            )

    def _combine_explanation(self, original_explanation: str, verification_report: VerificationReport) -> str:
        """Combine the original explanation with verification results."""
        if not verification_report.claims_verified:
            return original_explanation

        # Build verification summary
        verified_count = sum(1 for v in verification_report.claims_verified
                           if v.verdict == VerificationVerdict.VERIFIED)
        debunked_count = sum(1 for v in verification_report.claims_verified
                           if v.verdict == VerificationVerdict.DEBUNKED)
        questionable_count = sum(1 for v in verification_report.claims_verified
                               if v.verdict == VerificationVerdict.QUESTIONABLE)

        verification_summary = f"Verificación de {len(verification_report.claims_verified)} afirmaciones: "
        parts = []
        if verified_count > 0:
            parts.append(f"{verified_count} verificadas")
        if debunked_count > 0:
            parts.append(f"{debunked_count} desmentidas")
        if questionable_count > 0:
            parts.append(f"{questionable_count} cuestionables")

        if parts:
            verification_summary += ", ".join(parts)
        else:
            verification_summary += "sin resultados concluyentes"

        # Add confidence score
        verification_summary += f" (confianza: {verification_report.confidence_score:.1f}%)"

        # Add contradictions if any
        if verification_report.contradictions_found:
            verification_summary += f". Contradicciones detectadas: {len(verification_report.contradictions_found)}"

        # Combine with original explanation
        explanation_with_verification = f"{original_explanation}\n\n{verification_summary}"

        # Add source information
        if verification_report.evidence_sources:
            source_names = [s.source_name for s in verification_report.evidence_sources[:3]]
            explanation_with_verification += f"\nFuentes consultadas: {', '.join(source_names)}"

        return explanation_with_verification

    def get_verification_prompt_addition(self, verification_report: VerificationReport) -> str:
        """
        Generate prompt addition for LLM analysis based on verification results.

        Args:
            verification_report: Verification report

        Returns:
            Additional prompt text for LLM
        """
        if not verification_report.claims_verified:
            return ""

        additions = []

        # Add verified facts
        verified_claims = [v for v in verification_report.claims_verified
                          if v.verdict == VerificationVerdict.VERIFIED]
        if verified_claims:
            additions.append("HECHOS VERIFICADOS:")
            for claim in verified_claims[:3]:  # Limit to top 3
                additions.append(f"- {claim.claim}: {claim.explanation}")

        # Add debunked claims
        debunked_claims = [v for v in verification_report.claims_verified
                          if v.verdict == VerificationVerdict.DEBUNKED]
        if debunked_claims:
            additions.append("\nAFIRMACIONES DESMENTIDAS:")
            for claim in debunked_claims[:3]:
                additions.append(f"- {claim.claim}: {claim.explanation}")

        # Add contradictions
        if verification_report.contradictions_found:
            additions.append(f"\nCONTRADICCIONES DETECTADAS:")
            for contradiction in verification_report.contradictions_found[:2]:
                additions.append(f"- {contradiction}")

        # Add temporal consistency
        if not verification_report.temporal_consistency:
            additions.append("\nINCONSISTENCIAS TEMPORALES: Las fechas y eventos mencionados no siguen una secuencia lógica.")

        if additions:
            return "\n\n".join(additions)
        else:
            return ""

    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get statistics about verification triggering."""
        # This would track triggering patterns over time
        # For now, return placeholder
        return {
            "total_triggers": 0,
            "triggers_by_category": {},
            "triggers_by_keyword": {},
            "average_processing_time": 0.0
        }


def create_analyzer_hooks(verifier: Optional[ClaimVerifier] = None) -> AnalyzerHooks:
    """
    Factory function to create analyzer hooks with proper configuration.

    Args:
        verifier: Optional custom verifier instance

    Returns:
        Configured analyzer hooks
    """
    return AnalyzerHooks(verifier=verifier)