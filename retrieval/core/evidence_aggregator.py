"""
Evidence aggregation functionality.
Combines and reconciles evidence from multiple sources.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from ..core.models import VerificationResult, EvidenceSource, VerificationVerdict


class EvidenceAggregator:
    """
    Aggregates evidence from multiple sources and produces unified verdicts.
    Handles conflicts, weights sources by credibility, and provides explanations.
    """

    def __init__(self):
        # Source credibility weights (0.0 to 1.0)
        self.source_credibility = {
            # Fact-checking organizations (highest credibility)
            'maldita': 0.95,
            'newtral': 0.95,
            'snopes': 0.90,
            'politifact': 0.90,
            'factcheck': 0.90,

            # Statistical agencies
            'ine': 0.98,  # Spanish National Statistics Institute
            'eurostat': 0.95,
            'who': 0.93,
            'worldbank': 0.92,
            'oecd': 0.90,

            # Government sources
            'boe': 0.85,  # Spanish Official Gazette
            'mscbs': 0.88,  # Health Ministry
            'exteriores': 0.82,  # Foreign Affairs

            # News outlets (vary by context)
            'elpais': 0.75,
            'elmundo': 0.70,
            'abc': 0.68,
            'lavanguardia': 0.65,

            # Academic sources
            'scholar': 0.80,
            'dialnet': 0.75,

            # Default for unknown sources
            'unknown': 0.50
        }

        # Freshness weights (how much recency matters)
        self.freshness_weights = {
            'statistical': 0.8,  # Statistical data should be recent
            'news': 0.6,         # News can be somewhat dated
            'fact_check': 0.4,   # Fact checks are timeless but context matters
        }

    def aggregate_evidence(self, claim: str, evidence_sources: List[EvidenceSource],
                          claim_type: str = "unknown") -> VerificationResult:
        """
        Aggregate evidence from multiple sources into a unified verdict.

        Args:
            claim: The claim being verified
            evidence_sources: List of evidence sources
            claim_type: Type of claim (affects aggregation strategy)

        Returns:
            VerificationResult with unified verdict
        """
        if not evidence_sources:
            return VerificationResult(
                claim=claim,
                verdict=VerificationVerdict.UNVERIFIED,
                confidence=0.0,
                explanation="No evidence sources found"
            )

        start_time = datetime.now()

        # Filter and weight sources
        weighted_sources = self._weight_sources(evidence_sources, claim_type)

        # Aggregate verdicts
        aggregated_verdict, confidence, explanation = self._aggregate_verdicts(
            weighted_sources, claim_type
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        result = VerificationResult(
            claim=claim,
            verdict=aggregated_verdict,
            confidence=confidence,
            evidence_sources=weighted_sources,
            explanation=explanation,
            processing_time_seconds=processing_time,
            claim_type=claim_type
        )

        return result

    def _weight_sources(self, sources: List[EvidenceSource],
                       claim_type: str) -> List[EvidenceSource]:
        """
        Weight sources by credibility, freshness, and relevance.
        Updates credibility_score on each source.
        """
        weighted_sources = []

        for source in sources:
            # Use the credibility score already calculated by CredibilityScorer as base
            base_credibility = source.credibility_score

            # Adjust for freshness if we have publication date
            freshness_multiplier = self._calculate_freshness_multiplier(
                source, claim_type
            )

            # Adjust for content relevance (based on verdict confidence)
            relevance_multiplier = source.confidence

            # Calculate final credibility score
            final_credibility = base_credibility * freshness_multiplier * relevance_multiplier
            source.credibility_score = min(final_credibility, 1.0)

            weighted_sources.append(source)

        return weighted_sources

    def _calculate_freshness_multiplier(self, source: EvidenceSource,
                                      claim_type: str) -> float:
        """
        Calculate how much recency affects this source's credibility.
        """
        if not source.publication_date:
            return 0.9  # Slight penalty for unknown dates

        days_old = (datetime.now() - source.publication_date).days

        # Get freshness weight for this claim type
        freshness_weight = self.freshness_weights.get(claim_type, 0.5)

        if days_old <= 30:
            return 1.0  # Very recent
        elif days_old <= 90:
            return 0.95  # Recent
        elif days_old <= 365:
            return 0.85  # Within a year
        elif days_old <= 730:
            return 0.70  # Within 2 years
        else:
            # Gradual decay for older content
            return max(0.3, 1.0 - (days_old - 730) / 3650)  # Min 0.3 after 10+ years

    def _aggregate_verdicts(self, sources: List[EvidenceSource],
                          claim_type: str) -> Tuple[VerificationVerdict, float, str]:
        """
        Aggregate individual source verdicts into unified verdict.

        Returns:
            (verdict, confidence, explanation)
        """
        if not sources:
            return VerificationVerdict.UNCLEAR, 0.0, "No sources to aggregate"

        # Group sources by verdict
        verdict_groups = defaultdict(list)
        total_weight = 0.0

        for source in sources:
            weight = source.credibility_score
            verdict_groups[source.verdict_contribution].append((source, weight))
            total_weight += weight

        if total_weight == 0:
            return VerificationVerdict.UNCLEAR, 0.0, "All sources have zero weight"

        # Calculate weighted scores for each verdict
        verdict_scores = {}
        for verdict in VerificationVerdict:
            sources_for_verdict = verdict_groups.get(verdict, [])
            if sources_for_verdict:
                weighted_score = sum(weight for _, weight in sources_for_verdict) / total_weight
                verdict_scores[verdict] = weighted_score
            else:
                verdict_scores[verdict] = 0.0

        # Find the highest scoring verdict
        best_verdict = max(verdict_scores.items(), key=lambda x: x[1])
        confidence = best_verdict[1]

        # Check for contradictory evidence
        verified_score = verdict_scores.get(VerificationVerdict.VERIFIED, 0)
        debunked_score = verdict_scores.get(VerificationVerdict.DEBUNKED, 0)

        if verified_score > 0.3 and debunked_score > 0.3:
            # Significant contradictory evidence
            return (VerificationVerdict.CONTRADICTORY,
                   min(verified_score, debunked_score),
                   self._generate_contradictory_explanation(verdict_groups, total_weight))

        # Generate explanation for the winning verdict
        explanation = self._generate_verdict_explanation(
            best_verdict[0], verdict_groups, total_weight, claim_type
        )

        return best_verdict[0], confidence, explanation

    def _generate_contradictory_explanation(self, verdict_groups: Dict,
                                          total_weight: float) -> str:
        """Generate explanation for contradictory evidence."""
        verified_sources = verdict_groups.get(VerificationVerdict.VERIFIED, [])
        debunked_sources = verdict_groups.get(VerificationVerdict.DEBUNKED, [])

        verified_weight = sum(w for _, w in verified_sources) / total_weight
        debunked_weight = sum(w for _, w in debunked_sources) / total_weight

        explanation = f"Contradictory evidence found. "
        explanation += f"Supporting sources ({verified_weight:.1%} weighted evidence): "
        explanation += ", ".join([src.source_name for src, _ in verified_sources[:3]])
        explanation += ". Opposing sources "
        explanation += f"({debunked_weight:.1%} weighted evidence): "
        explanation += ", ".join([src.source_name for src, _ in debunked_sources[:3]])

        return explanation

    def _generate_verdict_explanation(self, verdict: VerificationVerdict,
                                    verdict_groups: Dict, total_weight: float,
                                    claim_type: str) -> str:
        """Generate explanation for the winning verdict."""
        sources_for_verdict = verdict_groups.get(verdict, [])
        supporting_weight = sum(w for _, w in sources_for_verdict) / total_weight

        verdict_names = {
            VerificationVerdict.VERIFIED: "verificada",
            VerificationVerdict.DEBUNKED: "desmentida",
            VerificationVerdict.QUESTIONABLE: "cuestionable"
        }

        explanation = f"La afirmación ha sido {verdict_names.get(verdict, 'evaluada')} "
        explanation += f"con {supporting_weight:.1%} de evidencia ponderada de "
        explanation += f"{len(sources_for_verdict)} fuente(s): "
        explanation += ", ".join([src.source_name for src, _ in sources_for_verdict[:5]])

        if len(sources_for_verdict) > 5:
            explanation += f" y {len(sources_for_verdict) - 5} más"

        return explanation

    def resolve_conflicts(self, conflicting_sources: List[EvidenceSource]) -> List[EvidenceSource]:
        """
        Attempt to resolve conflicts between sources.
        Returns filtered list with conflicts resolved.
        """
        if len(conflicting_sources) <= 1:
            return conflicting_sources

        # Group by verdict
        verdict_groups = defaultdict(list)
        for source in conflicting_sources:
            verdict_groups[source.verdict_contribution].append(source)

        # If we have clear winners, keep only those
        max_group_size = max(len(group) for group in verdict_groups.values())

        # If one verdict has majority, keep only those sources
        majority_verdicts = [
            verdict for verdict, sources in verdict_groups.items()
            if len(sources) == max_group_size
        ]

        if len(majority_verdicts) == 1:
            return verdict_groups[majority_verdicts[0]]

        # Otherwise, keep all but down-weight conflicting sources
        for source in conflicting_sources:
            if len(verdict_groups[source.verdict_contribution]) < max_group_size:
                # Reduce credibility for minority opinions
                source.credibility_score *= 0.7

        return conflicting_sources


def aggregate_evidence(claim: str, evidence_sources: List[EvidenceSource],
                      claim_type: str = "unknown") -> VerificationResult:
    """
    Convenience function to aggregate evidence.

    Args:
        claim: The claim being verified
        evidence_sources: List of evidence sources
        claim_type: Type of claim

    Returns:
        Aggregated verification result
    """
    aggregator = EvidenceAggregator()
    return aggregator.aggregate_evidence(claim, evidence_sources, claim_type)