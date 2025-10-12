"""
Result formatting utilities for LLM consumption.
Formats verification results and evidence for easy integration with language models.
"""

import json
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from ..core.models import VerificationResult, EvidenceSource, VerificationVerdict
from ..verification.claim_verifier import VerificationReport
from ..sources.fact_checkers import FactCheckResult
from ..sources.web_scrapers import ScrapedContent


@dataclass
class LLMFormattedResult:
    """Formatted result optimized for LLM consumption."""
    claim: str
    verdict: str
    confidence: float
    explanation: str
    evidence_count: int
    sources: List[str]
    key_facts: List[str]
    contradictions: List[str]
    temporal_consistency: bool
    processing_time: float
    formatted_for_llm: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ResultFormatter:
    """
    Formats verification results for LLM consumption.
    Provides multiple output formats optimized for different use cases.
    """

    def __init__(self):
        self.max_explanation_length = 1000
        self.max_sources_per_result = 5

    def format_for_llm_prompt(self, verification_result: VerificationResult) -> str:
        """
        Format a single verification result for LLM prompt addition.

        Args:
            verification_result: Single claim verification result

        Returns:
            Formatted string for LLM consumption
        """
        verdict_text = self._verdict_to_spanish(verification_result.verdict)

        # Build evidence summary
        evidence_parts = []
        for source in verification_result.evidence_sources[:self.max_sources_per_result]:
            source_info = f"- {source.source_name}: {source.verdict_contribution.value}"
            if source.content_snippet:
                source_info += f" ({source.content_snippet[:100]}...)"
            evidence_parts.append(source_info)

        evidence_summary = "\n".join(evidence_parts) if evidence_parts else "Sin fuentes de evidencia disponibles"

        # Format explanation
        explanation = verification_result.explanation[:self.max_explanation_length]
        if len(verification_result.explanation) > self.max_explanation_length:
            explanation += "..."

        formatted = f"""
VERIFICACIÓN DE AFIRMACIÓN:
Afirmación: {verification_result.claim}
Veredicto: {verdict_text}
Confianza: {verification_result.confidence:.1f}%

Explicación: {explanation}

Evidencia encontrada ({len(verification_result.evidence_sources)} fuentes):
{evidence_summary}

Tipo de afirmación: {verification_result.claim_type}
Valor extraído: {verification_result.extracted_value or 'N/A'}
"""

        return formatted.strip()

    def format_verification_report(self, report: VerificationReport) -> LLMFormattedResult:
        """
        Format a complete verification report for LLM consumption.

        Args:
            report: Complete verification report

        Returns:
            Formatted result object
        """
        # Overall verdict
        overall_verdict = self._verdict_to_spanish(report.overall_verdict)

        # Collect all sources
        all_sources = []
        seen_urls = set()
        for source in report.evidence_sources:
            if source.url not in seen_urls:
                all_sources.append(source.source_name)
                seen_urls.add(source.url)

        # Extract key facts from verified claims
        key_facts = []
        for claim_result in report.claims_verified:
            if claim_result.verdict in [VerificationVerdict.VERIFIED, VerificationVerdict.DEBUNKED]:
                fact = f"{claim_result.claim} -> {self._verdict_to_spanish(claim_result.verdict)}"
                key_facts.append(fact)

        # Build comprehensive explanation
        explanation_parts = []

        if report.claims_verified:
            verified_count = sum(1 for v in report.claims_verified if v.verdict == VerificationVerdict.VERIFIED)
            debunked_count = sum(1 for v in report.claims_verified if v.verdict == VerificationVerdict.DEBUNKED)
            questionable_count = sum(1 for v in report.claims_verified if v.verdict == VerificationVerdict.QUESTIONABLE)

            summary = f"Se verificaron {len(report.claims_verified)} afirmaciones: "
            parts = []
            if verified_count > 0:
                parts.append(f"{verified_count} verificadas")
            if debunked_count > 0:
                parts.append(f"{debunked_count} desmentidas")
            if questionable_count > 0:
                parts.append(f"{questionable_count} cuestionables")

            if parts:
                summary += ", ".join(parts)
            else:
                summary += "sin resultados concluyentes"

            explanation_parts.append(summary)

        if report.contradictions_found:
            explanation_parts.append(f"Contradicciones detectadas: {len(report.contradictions_found)}")

        if not report.temporal_consistency:
            explanation_parts.append("Inconsistencias temporales en las afirmaciones")

        explanation = ". ".join(explanation_parts) if explanation_parts else "Sin explicación disponible"

        # Build LLM-optimized formatted text
        llm_text = f"""
ANÁLISIS DE VERIFICACIÓN COMPLETO:

VEREDICTO GENERAL: {overall_verdict}
CONFIANZA GENERAL: {report.confidence_score:.1f}%

EXPLICACIÓN: {explanation}

FUENTES CONSULTADAS ({len(all_sources)}):
{chr(10).join(f"- {source}" for source in all_sources[:self.max_sources_per_result])}

HECHOS CLAVE:
{chr(10).join(f"- {fact}" for fact in key_facts[:5])}

CONTRADICCIONES:
{chr(10).join(f"- {contradiction}" for contradiction in report.contradictions_found[:3])}

CONSISTENCIA TEMPORAL: {"Sí" if report.temporal_consistency else "No"}

TIEMPO DE PROCESAMIENTO: {report.processing_time:.2f} segundos
"""

        return LLMFormattedResult(
            claim=report.claims_verified[0].claim if report.claims_verified else "Múltiples afirmaciones",
            verdict=overall_verdict,
            confidence=report.confidence_score,
            explanation=explanation,
            evidence_count=len(report.evidence_sources),
            sources=all_sources[:self.max_sources_per_result],
            key_facts=key_facts[:5],
            contradictions=report.contradictions_found[:3],
            temporal_consistency=report.temporal_consistency,
            processing_time=report.processing_time,
            formatted_for_llm=llm_text.strip()
        )

    def format_fact_check_result(self, fact_check: FactCheckResult) -> str:
        """
        Format a fact-check result for LLM consumption.

        Args:
            fact_check: Fact-check result from fact-checking site

        Returns:
            Formatted string for LLM
        """
        verdict_text = self._normalize_fact_check_verdict(fact_check.verdict)

        formatted = f"""
RESULTADO DE VERIFICACIÓN DE HECHOS:
Afirmación verificada: {fact_check.claim}
Veredicto: {verdict_text}
Confianza: {fact_check.confidence:.1f}%

Explicación: {fact_check.explanation}

Fuente: {fact_check.source_name}
URL: {fact_check.source_url}
Fecha: {fact_check.publication_date.strftime('%Y-%m-%d') if fact_check.publication_date else 'N/A'}

Etiquetas: {', '.join(fact_check.tags) if fact_check.tags else 'Ninguna'}
"""

        return formatted.strip()

    def format_scraped_content(self, content: ScrapedContent) -> str:
        """
        Format scraped web content for LLM consumption.

        Args:
            content: Scraped content from web source

        Returns:
            Formatted string for LLM
        """
        formatted = f"""
CONTENIDO EXTRAÍDO DE FUENTE WEB:
Título: {content.title}
Fuente: {content.source_name}
URL: {content.url}
Fecha: {content.publication_date.strftime('%Y-%m-%d') if content.publication_date else 'N/A'}

Puntuación de credibilidad: {content.credibility_score:.1f}/100
Puntuación de relevancia: {content.relevance_score:.2f}

Contenido:
{content.content}

Etiquetas: {', '.join(content.tags) if content.tags else 'Ninguna'}
"""

        return formatted.strip()

    def create_evidence_summary(self, results: List[Union[VerificationResult, FactCheckResult, ScrapedContent]]) -> str:
        """
        Create a comprehensive evidence summary from multiple sources.

        Args:
            results: List of verification results from different sources

        Returns:
            Comprehensive evidence summary for LLM
        """
        if not results:
            return "No se encontró evidencia disponible."

        summary_parts = []
        summary_parts.append("RESUMEN DE EVIDENCIA RECOPILADA:")
        summary_parts.append("=" * 50)

        # Categorize results
        verification_results = [r for r in results if isinstance(r, VerificationResult)]
        fact_checks = [r for r in results if isinstance(r, FactCheckResult)]
        scraped_content = [r for r in results if isinstance(r, ScrapedContent)]

        # Add verification results
        if verification_results:
            summary_parts.append(f"\nRESULTADOS DE VERIFICACIÓN ({len(verification_results)}):")
            for result in verification_results[:3]:  # Limit to top 3
                verdict = self._verdict_to_spanish(result.verdict)
                summary_parts.append(f"- {result.claim[:100]}... -> {verdict} ({result.confidence:.1f}%)")

        # Add fact-checks
        if fact_checks:
            summary_parts.append(f"\nVERIFICACIONES DE HECHOS ({len(fact_checks)}):")
            for fact_check in fact_checks[:3]:  # Limit to top 3
                verdict = self._normalize_fact_check_verdict(fact_check.verdict)
                summary_parts.append(f"- {fact_check.source_name}: {verdict} ({fact_check.confidence:.1f}%)")

        # Add scraped content
        if scraped_content:
            summary_parts.append(f"\nCONTENIDO WEB EXTRAÍDO ({len(scraped_content)}):")
            for content in scraped_content[:3]:  # Limit to top 3
                summary_parts.append(f"- {content.source_name}: {content.title[:80]}... (credibilidad: {content.credibility_score:.1f})")

        # Overall assessment
        total_results = len(results)
        high_confidence = sum(1 for r in results if getattr(r, 'confidence', 0) > 0.8)

        summary_parts.append(f"\nEVALUACIÓN GENERAL:")
        summary_parts.append(f"- Total de resultados: {total_results}")
        summary_parts.append(f"- Resultados de alta confianza: {high_confidence}")
        summary_parts.append(f"- Cobertura de fuentes: {len(set(getattr(r, 'source_name', 'Unknown') for r in results))}")

        return "\n".join(summary_parts)

    def _verdict_to_spanish(self, verdict: VerificationVerdict) -> str:
        """Convert verdict enum to Spanish text."""
        verdict_map = {
            VerificationVerdict.VERIFIED: "VERIFICADO",
            VerificationVerdict.DEBUNKED: "DESMENTIDO",
            VerificationVerdict.QUESTIONABLE: "CUESTIONABLE",
            VerificationVerdict.UNVERIFIED: "SIN VERIFICAR"
        }
        return verdict_map.get(verdict, "DESCONOCIDO")

    def _normalize_fact_check_verdict(self, verdict: str) -> str:
        """Normalize fact-check verdict to Spanish."""
        verdict_lower = verdict.lower()
        if verdict_lower in ['true', 'verdadero', 'cierto']:
            return "VERDADERO"
        elif verdict_lower in ['false', 'falso', 'mentira']:
            return "FALSO"
        elif verdict_lower in ['misleading', 'engañoso', 'manipulado']:
            return "ENGAÑOSO"
        else:
            return verdict.upper()


# Convenience functions
def format_single_result(result: VerificationResult) -> str:
    """
    Convenience function to format a single verification result.

    Args:
        result: Verification result to format

    Returns:
        Formatted string for LLM
    """
    formatter = ResultFormatter()
    return formatter.format_for_llm_prompt(result)


def format_verification_report(report: VerificationReport) -> LLMFormattedResult:
    """
    Convenience function to format a verification report.

    Args:
        report: Verification report to format

    Returns:
        Formatted result object
    """
    formatter = ResultFormatter()
    return formatter.format_verification_report(report)


def create_evidence_summary(results: List[Union[VerificationResult, FactCheckResult, ScrapedContent]]) -> str:
    """
    Convenience function to create evidence summary.

    Args:
        results: List of results to summarize

    Returns:
        Comprehensive evidence summary
    """
    formatter = ResultFormatter()
    return formatter.create_evidence_summary(results)