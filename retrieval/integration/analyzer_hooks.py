"""
Integration hooks for connecting the retrieval system with the main analyzer.
Provides conditional triggering and analysis capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..core.models import VerificationResult, VerificationVerdict
from ..verification.claim_verifier import ClaimVerifier, VerificationContext, VerificationReport
from ..verification.political_event_verifier import PoliticalEventVerifier


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

    def __init__(self, verifier: Optional[ClaimVerifier] = None, verbose: bool = False):
        self.verifier = verifier or ClaimVerifier()
        self.political_verifier = PoliticalEventVerifier()
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose

        # Default trigger configuration
        self.default_trigger = AnalysisTrigger(
            content_categories=['disinformation', 'conspiracy_theory', 'anti_immigration', 'anti_government', 'political_general', 'call_to_action'],
            confidence_threshold=0.4,  # Lower threshold to catch more cases
            keywords=[
                # EXISTING - Keep all existing keywords
                'estadística', 'dato', 'cifra', 'millón', 'porcentaje', '%',
                'según', 'informe', 'estudio', 'investigación',
                'pandemia', 'covid', 'vacuna', 'muerte', 'contagio',
                'elecciones', 'voto', 'partido', 'gobierno',
                'economía', 'paro', 'empleo', 'pib', 'déficit',
                
                # NEW: Breaking news/urgency markers
                'última hora', 'urgente', 'breaking', 'bombazo', 
                'exclusiva', 'confirmado', 'confirmada', 'al descubierto',
                
                # NEW: Legal/judicial events
                'ingresa', 'ingreso', 'prisión', 'cárcel', 'soto del real',
                'juez', 'jueza', 'fiscal', 'tribunal', 'juzgado', 'corte',
                'arresta', 'arrestado', 'detiene', 'detenido', 'detención',
                'imputa', 'imputado', 'imputación', 
                'procesa', 'procesado', 'procesamiento',
                'condena', 'condenado', 'sentencia',
                
                # NEW: High-profile political targets (common fake news subjects)
                'ábalos', 'sánchez', 'iglesias', 'montero', 'grande-marlaska',
                'redondo', 'calvo', 'belarra', 'yolanda díaz',
                
                # NEW: Government action keywords (decrees, laws, resignations)
                'decreto', 'ley', 'real decreto', 'dimite', 'dimisión', 'dimitido', 'renuncia', 'renunciado',
                'aprueba', 'aprobado', 'firma', 'firmado', 'promulga', 'promulgado',
                'prohíbe', 'prohibido', 'obliga', 'obligatorio', 'manda', 'orden',
                'cesa', 'cesado', 'destituye', 'destituido', 'nombramiento', 'nombrado',
                'alianza', 'pacto', 'acuerdo', 'coalición', 'gobierno', 'consejo de ministros',
                
                # NEW: Confirmation and finality markers
                'ya está', 'ya es oficial', 'confirmado por', 'según fuentes oficiales',
                'ha sido', 'se ha confirmado', 'queda demostrado', 'está claro que',
                'oficialmente', 'formalmente', 'definitivamente', 'inmediatamente',
            ],
            claim_types=['numerical', 'statistical', 'temporal', 'attribution'],
            min_claims=1
        )

    def explanation_indicates_disinformation(self, explanation: str) -> bool:
        """
        Check if the LLM explanation indicates the content should be categorized as disinformation.
        
        Args:
            explanation: The explanation text from the LLM
            
        Returns:
            True if explanation suggests disinformation, False otherwise
        """
        explanation_lower = explanation.lower()
        
        # Check for key phrases that indicate disinformation detection
        disinformation_indicators = [
            "desinformación",
            "sin citar fuentes",
            "sin fuente oficial",
            "no menciona fuente",
            "fuentes verificables",
            "afirmación sin fundamento",
            "sin evidencia",
            "no aporta evidencia",
            "carece de fuentes",
            "sin respaldo oficial"
        ]
        
        return any(indicator in explanation_lower for indicator in disinformation_indicators)
    
    def external_analysis_indicates_disinformation(self, external_explanation: str) -> bool:
        """
        Check if the external analysis indicates disinformation.
        
        Args:
            external_explanation: The external analysis explanation
            
        Returns:
            True if external analysis detects disinformation
        """
        if not external_explanation:
            return False
            
        explanation_lower = external_explanation.lower()
        
        # Strong indicators from external analysis (Gemini is more reliable)
        # Expanded list to include more Spanish variations and synonyms
        disinformation_indicators = [
            "desinformación",
            "información falsa",
            "noticia falsa",
            "completamente falsa",
            "totalmente falsa",
            "verificar con fuentes",
            "sin fundamento",
            "engaño",
            "manipulación",
            "contenido falso",
            "afirmación falsa",
            "bulo",  # Spanish for hoax
            "fake news",
            "noticia falsa",
            "información manipulada",
            "contenido manipulado",
            "falsificación",
            "documentos falsos",
            "pruebas falsas",
            "evidencia falsa",
            "afirmación no verificada",
            "sin verificación",
            "carece de fundamento",
            "no tiene base",
            "es falso",
            "son falsos"
        ]
        
        return any(indicator in explanation_lower for indicator in disinformation_indicators)

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
            # Check if this is a political event claim that needs specialized verification
            political_event_claim = self._extract_political_event_claim(content)
            
            if political_event_claim:
                if self.verbose:
                    print(f"🎯 Detected political event claim: {political_event_claim.event_type} for {political_event_claim.person_name}")
                
                # Use specialized political event verifier
                verification_report = await self._verify_political_event(political_event_claim)
            else:
                # Use general claim verifier
                verification_context = VerificationContext(
                    original_text=content,
                    content_category=original_result.get('category', 'general'),
                    user_context=f"Analysis category: {original_result.get('category', 'unknown')}",
                    language="es",
                    priority_level="balanced"
                )

                verification_report = await self.verifier.verify_content(verification_context)

            # Create simplified verification data structure
            simplified_verification = self._create_simplified_verification_data(verification_report)
            
            # Enhance explanation with verification results
            original_explanation = original_result.get('explanation', '')
            verification_addition = self.get_verification_prompt_addition(verification_report)
            
            # Always add verification information, even if minimal
            if verification_addition:
                explanation_with_verification = f"{original_explanation}\n\n{verification_addition}"
            else:
                # Add basic verification note if no specific claims were verified
                verification_note = f"Verificación realizada: {len(verification_report.claims_verified)} afirmaciones analizadas."
                if verification_report.contradictions_found:
                    verification_note += f" Contradicciones detectadas: {len(verification_report.contradictions_found)}."
                explanation_with_verification = f"{original_explanation}\n\n{verification_note}"

            return AnalysisResult(
                original_result=original_result,
                verification_data=simplified_verification,
                explanation_with_verification=explanation_with_verification
            )

        except Exception as e:
            # Log the full exception with stack trace for debugging
            self.logger.exception("Verification failed with exception")

            # Annotate the result with error information so callers can detect verification failure
            error_info = {
                'error': f"{type(e).__name__}: {str(e)}",
                'verification_failed': True
            }

            return AnalysisResult(
                original_result=original_result,
                verification_data=error_info,
                explanation_with_verification=original_result.get('explanation', '')
            )


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

    def _extract_political_event_claim(self, content: str) -> Optional[Any]:
        """
        Extract political event claims from content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            PoliticalEventClaim if detected, None otherwise
        """
        from ..verification.political_event_verifier import PoliticalEventClaim
        
        content_lower = content.lower()
        
        # Check for arrest/imprisonment claims
        arrest_indicators = ['ingresa', 'ingreso', 'prisión', 'cárcel', 'arrestado', 'detenido']
        has_arrest = any(indicator in content_lower for indicator in arrest_indicators)
        
        # Check for political figures
        political_figures = ['ábalos', 'sánchez', 'iglesias', 'montero', 'casado', 'abascal', 'rivera']
        person_detected = None
        for figure in political_figures:
            if figure in content_lower:
                person_detected = figure
                break
        
        if has_arrest and person_detected:
            return PoliticalEventClaim(
                person_name=person_detected,
                event_type='arrest',
                context=content
            )
        
        return None

    async def _verify_political_event(self, claim) -> VerificationReport:
        """
        Verify a political event claim using the specialized verifier.
        
        Args:
            claim: PoliticalEventClaim to verify
            
        Returns:
            VerificationReport with results
        """
        try:
            if claim.event_type == 'arrest':
                result = await self.political_verifier.verify_arrest_claim(
                    claim.person_name, 
                    claim.institution
                )
            else:
                result = await self.political_verifier.verify_political_event(claim.context)
            
            # Convert VerificationResult to VerificationReport format
            from ..verification.claim_verifier import VerificationReport
            
            # Use the VerificationResult directly as claims_verified expects List[VerificationResult]
            verified_claims = [result]
            
            # Extract contradictions from the result if available
            contradictions = []
            if hasattr(result, 'contradictions_found'):
                contradictions = result.contradictions_found
            elif "desmentido" in result.explanation.lower() or "falso" in result.explanation.lower():
                contradictions = [result.explanation]
            
            return VerificationReport(
                overall_verdict=result.verdict,
                confidence_score=result.confidence,
                claims_verified=verified_claims,
                evidence_sources=result.evidence_sources,
                contradictions_found=contradictions,
                temporal_consistency=True,
                processing_time=0.0,
                verification_method="political_event_verifier"
            )
            
        except Exception as e:
            self.logger.error(f"Political event verification failed: {e}")
            # Return empty report on failure
            return VerificationReport(
                overall_verdict=VerificationVerdict.QUESTIONABLE,
                confidence_score=0.0,
                claims_verified=[],
                evidence_sources=[],
                contradictions_found=[],
                temporal_consistency=True,
                processing_time=0.0,
                verification_method="political_event_verifier"
            )


    def _create_simplified_verification_data(self, verification_report: VerificationReport) -> Dict[str, Any]:
        """
        Create a simplified verification data structure for storage and display.
        
        Args:
            verification_report: Complex verification report
            
        Returns:
            Simplified dict with essential verification information
        """
        # Count verdicts
        verified_count = sum(1 for v in verification_report.claims_verified 
                           if v.verdict == VerificationVerdict.VERIFIED)
        debunked_count = sum(1 for v in verification_report.claims_verified 
                           if v.verdict == VerificationVerdict.DEBUNKED)
        questionable_count = sum(1 for v in verification_report.claims_verified 
                               if v.verdict == VerificationVerdict.QUESTIONABLE)
        
        # Get top evidence sources (limit to 5 for display)
        evidence_sources = []
        for source in verification_report.evidence_sources[:5]:
            evidence_sources.append({
                'name': source.source_name,
                'type': source.source_type,
                'url': source.url,
                'credibility': source.credibility_score,
                'verdict': source.verdict_contribution.value,
                'confidence': source.confidence
            })
        
        # Create simplified structure
        simplified = {
            'overall_verdict': verification_report.overall_verdict.value,
            'confidence_score': verification_report.confidence_score,
            'verification_confidence': verification_report.confidence_score,  # For backward compatibility
            'claims_summary': {
                'total': len(verification_report.claims_verified),
                'verified': verified_count,
                'debunked': debunked_count,
                'questionable': questionable_count
            },
            'evidence_sources': evidence_sources,
            'contradictions_found': verification_report.contradictions_found,
            'temporal_consistency': verification_report.temporal_consistency,
            'verification_method': verification_report.verification_method
        }
        
        return simplified


def create_analyzer_hooks(verifier: Optional[ClaimVerifier] = None, verbose: bool = False) -> AnalyzerHooks:
    """
    Factory function to create analyzer hooks with proper configuration.

    Args:
        verifier: Optional custom verifier instance
        verbose: Enable verbose logging

    Returns:
        Configured analyzer hooks
    """
    return AnalyzerHooks(verifier=verifier, verbose=verbose)