"""
Claim extraction functionality for evidence retrieval.
Identifies claims that warrant verification from text content.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ClaimType(Enum):
    """Types of claims that can be verified."""
    NUMERICAL = "numerical"  # Statistics, percentages, counts
    TEMPORAL = "temporal"    # Dates, timelines, time-based claims
    ATTRIBUTION = "attribution"  # Claims about who said/did what
    CAUSAL = "causal"        # Cause-effect relationships
    EXISTENTIAL = "existential"  # Claims about existence/non-existence
    LEGAL_EVENT = "legal_event"  # Arrests, court proceedings, convictions
    BREAKING_NEWS = "breaking_news"  # Urgent news claims


@dataclass
class VerificationTarget:
    """A claim that should be verified."""
    claim_text: str
    claim_type: ClaimType
    context: str  # Surrounding text for better understanding
    confidence: float  # How likely this needs verification (0.0-1.0)
    priority: int  # Verification priority (1-10, higher = more important)
    extracted_value: Optional[str] = None  # The actual number/date/etc extracted
    start_pos: int = 0  # Position in original text
    end_pos: int = 0


class ClaimExtractor:
    """
    Extracts claims from text that warrant verification.
    Focuses on numerical, temporal, and factual claims.
    """

    def __init__(self):
        # Numerical patterns (percentages, counts, monetary values)
        self.numerical_patterns = [
            # Percentages: 25%, 15.5%, etc.
            r'\b\d+(?:\.\d+)?%\b',
            # Large numbers with commas: 1,234,567 or 1234567
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',
            # Numbers followed by units: 500 million, 2.5 billion, etc.
            r'\b\d+(?:\.\d+)?\s+(?:millones?|billones?|mil|trillones?)\b',
            # Monetary values: 100 euros, $500, etc.
            r'\b\d+(?:\.\d+)?\s*(?:euros?|dólares?|€|\$|£)\b',
        ]

        # Temporal patterns (dates, times, periods)
        self.temporal_patterns = [
            # Dates: 15/10/2023, 2023-10-15, etc.
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            # Years: 2023, 1995, etc.
            r'\b(19|20)\d{2}\b',
            # Time periods: hace 3 días, en 2020, etc.
            r'\b(?:hace|en|desde|durante)\s+\d+\s+(?:días?|meses?|años?|horas?)\b',
        ]

        # Attribution patterns (who said/did what)
        self.attribution_patterns = [
            # Direct attributions: "X dijo que...", "Según Y..."
            r'\b(?:según|como\s+dice|afirmó|declaró|dijo)\s+[^,.:;]{10,50}[,:]',
            # Source attributions: "el gobierno afirma", "la OMS dice"
            r'\b(?:el\s+gobierno|la\s+OMS|la\s+ONU|expertos|científicos)\s+(?:afirman?|dicen?|indican?)\b',
        ]

        # Causal patterns (cause-effect relationships)
        self.causal_patterns = [
            # Causal connectors: "porque", "debido a", "por eso"
            r'\b(?:porque|debido\s+a|por\s+eso|ya\s+que|como\s+resultado)\b[^.]{20,100}\.',
        ]

        # Legal/political event patterns (arrests, court proceedings, etc.)
        self.legal_event_patterns = [
            # Arrest/imprisonment claims
            r'\b(?:arrestado|detenido|detención\s+de)\s+[A-ZÁ-ÚÑ][a-zá-úñ]+\b',
            r'\b[A-ZÁ-ÚÑ][a-zá-úñ]+\s+(?:ingresa|entra|llega)\s+(?:en|a)\s+(?:prisión|cárcel|Soto\s+del\s+Real)\b',
            r'\b(?:ingresa|ingreso\s+de)\s+[A-ZÁ-ÚÑ][a-zá-úñ]+\s+(?:en|a)\s+(?:prisión|cárcel)\b',
            
            # Judicial proceedings
            r'\b(?:juez|jueza|fiscal|tribunal)\s+(?:ordena|dicta|envía|procesa)\s+[A-ZÁ-ÚÑ][a-zá-úñ]+\b',
            r'\b[A-ZÁ-ÚÑ][a-zá-úñ]+\s+(?:procesado|imputado|condenado)\s+(?:por|en)\b',
            
            # Legal actions
            r'\b(?:procesamiento|imputación|sentencia)\s+(?:de|contra)\s+[A-ZÁ-ÚÑ][a-zá-úñ]+\b',
        ]

        # Breaking news patterns
        self.breaking_news_patterns = [
            # Urgency markers + claims
            r'\b(?:ÚLTIMA\s+HORA|URGENTE|BREAKING|BOMBAZO|EXCLUSIVA)\b[^.]*?(?:ingresa|arrestan|detienen|procesan|condenan)\b',
            r'\b(?:ÚLTIMA\s+HORA|URGENTE|BREAKING|BOMBAZO)\b[^.]*?(?:[A-ZÁ-ÚÑ][a-zá-úñ]+)\s+(?:ingresa|arrestado|detenido|procesado)\b',
            
            # Confirmation claims without sources
            r'\b(?:se\s+confirma|ya\s+está|acabamos\s+de)\s+(?:que\s+)?(?:[A-ZÁ-ÚÑ][a-zá-úñ]+)\s+(?:ingresa|arrestado|detenido)\b',
        ]

    def extract_verification_targets(self, text: str, max_targets: int = 5) -> List[VerificationTarget]:
        """
        Extract verification targets from text.
        This is the main method that combines all extraction strategies.
        """
        all_targets = []

        # Extract different types of claims
        numerical_targets = self._extract_numerical_claims(text)
        temporal_targets = self._extract_temporal_claims(text)
        attribution_targets = self._extract_attribution_claims(text)
        causal_targets = self._extract_causal_claims(text)
        legal_event_targets = self._extract_legal_event_claims(text)
        breaking_news_targets = self._extract_breaking_news_claims(text)
        # Extract full contextual claims containing numbers/dates
        full_claims = self._extract_full_contextual_claims(text)

        # Combine and prioritize
        all_targets.extend(numerical_targets)
        all_targets.extend(temporal_targets)
        all_targets.extend(attribution_targets)
        all_targets.extend(causal_targets)
        all_targets.extend(legal_event_targets)
        all_targets.extend(breaking_news_targets)
        all_targets.extend(full_claims)

        # Sort by priority and confidence
        all_targets.sort(key=lambda x: (x.priority, x.confidence), reverse=True)

        # Return top targets
        return all_targets[:max_targets]

    def extract_claims(self, text: str, max_targets: int = 5) -> List[VerificationTarget]:
        """
        Convenience method - alias for extract_verification_targets.
        """
        return self.extract_verification_targets(text, max_targets)

    def _extract_numerical_claims(self, text: str) -> List[VerificationTarget]:
        """Extract numerical/statistical claims."""
        targets = []

        for pattern in self.numerical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim_text = match.group()
                context = self._extract_context(text, match.start(), match.end())

                # Calculate confidence and priority based on claim characteristics
                confidence, priority = self._assess_numerical_claim(claim_text, context)

                target = VerificationTarget(
                    claim_text=claim_text,
                    claim_type=ClaimType.NUMERICAL,
                    context=context,
                    confidence=confidence,
                    priority=priority,
                    extracted_value=claim_text,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                targets.append(target)

        return targets

    def _extract_temporal_claims(self, text: str) -> List[VerificationTarget]:
        """Extract temporal/date-based claims."""
        targets = []

        for pattern in self.temporal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim_text = match.group()
                context = self._extract_context(text, match.start(), match.end())

                confidence, priority = self._assess_temporal_claim(claim_text, context)

                target = VerificationTarget(
                    claim_text=claim_text,
                    claim_type=ClaimType.TEMPORAL,
                    context=context,
                    confidence=confidence,
                    priority=priority,
                    extracted_value=claim_text,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                targets.append(target)

        return targets

    def _extract_attribution_claims(self, text: str) -> List[VerificationTarget]:
        """Extract attribution claims (who said what)."""
        targets = []

        for pattern in self.attribution_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim_text = match.group()
                context = self._extract_context(text, match.start(), match.end(), window=100)

                confidence, priority = self._assess_attribution_claim(claim_text, context)

                target = VerificationTarget(
                    claim_text=claim_text,
                    claim_type=ClaimType.ATTRIBUTION,
                    context=context,
                    confidence=confidence,
                    priority=priority,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                targets.append(target)

        return targets

    def _extract_causal_claims(self, text: str) -> List[VerificationTarget]:
        """Extract causal relationship claims."""
        targets = []

        for pattern in self.causal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim_text = match.group()
                context = self._extract_context(text, match.start(), match.end(), window=150)

                confidence, priority = self._assess_causal_claim(claim_text, context)

                target = VerificationTarget(
                    claim_text=claim_text,
                    claim_type=ClaimType.CAUSAL,
                    context=context,
                    confidence=confidence,
                    priority=priority,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                targets.append(target)

        return targets

    def _extract_legal_event_claims(self, text: str) -> List[VerificationTarget]:
        """Extract legal/political event claims (arrests, court proceedings, etc.)."""
        targets = []

        for pattern in self.legal_event_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim_text = match.group()
                context = self._extract_context(text, match.start(), match.end(), window=80)

                confidence, priority = self._assess_legal_event_claim(claim_text, context)

                target = VerificationTarget(
                    claim_text=claim_text,
                    claim_type=ClaimType.LEGAL_EVENT,
                    context=context,
                    confidence=confidence,
                    priority=priority,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                targets.append(target)

        return targets

    def _extract_breaking_news_claims(self, text: str) -> List[VerificationTarget]:
        """Extract breaking news claims that may be disinformation."""
        targets = []

        for pattern in self.breaking_news_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim_text = match.group()
                context = self._extract_context(text, match.start(), match.end(), window=100)

                confidence, priority = self._assess_breaking_news_claim(claim_text, context)

                target = VerificationTarget(
                    claim_text=claim_text,
                    claim_type=ClaimType.BREAKING_NEWS,
                    context=context,
                    confidence=confidence,
                    priority=priority,
                    start_pos=match.start(),
                    end_pos=match.end()
                )
                targets.append(target)

        return targets

    def _extract_full_contextual_claims(self, text: str) -> List[VerificationTarget]:
        """Extract full contextual claims containing numbers or dates."""
        targets = []
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            # Check if sentence contains numerical or temporal elements
            has_numerical = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in self.numerical_patterns)
            has_temporal = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in self.temporal_patterns)

            if has_numerical or has_temporal:
                # Extract the actual numerical/temporal value from the sentence
                extracted_value = None
                claim_type = ClaimType.NUMERICAL  # Default to numerical

                if has_numerical:
                    # Find the first number in the sentence
                    num_match = re.search(r'(\d+(?:[.,]\d+)*)', sentence)
                    if num_match:
                        extracted_value = num_match.group(1)
                        # Check for multipliers after the number
                        remaining_text = sentence[num_match.end():].lower()
                        if 'millones' in remaining_text or 'million' in remaining_text:
                            extracted_value = f"{extracted_value} millones"
                        elif 'billones' in remaining_text or 'billion' in remaining_text:
                            extracted_value = f"{extracted_value} billones"
                        elif '%' in remaining_text:
                            extracted_value = f"{extracted_value}%"

                if has_temporal:
                    # Find temporal expressions
                    temp_match = re.search(r'(\d{4})', sentence)  # Year
                    if temp_match:
                        temp_value = temp_match.group(1)
                        # If we also have numerical, determine which is primary
                        if has_numerical and extracted_value:
                            # For sentences with both, check keywords to determine primary type
                            if any(word in sentence.lower() for word in ['creció', 'aumento', 'disminuyó', 'bajó', 'pib', 'gdp', 'tasa', 'porcentaje', 'rate']):
                                claim_type = ClaimType.NUMERICAL  # Keep numerical for economic indicators
                            elif any(word in sentence.lower() for word in ['empezó', 'terminó', 'ocurrió', 'pasó', 'inicio', 'final', 'started', 'ended']):
                                claim_type = ClaimType.TEMPORAL
                                extracted_value = temp_value
                            else:
                                claim_type = ClaimType.NUMERICAL  # Default to numerical
                        else:
                            # Pure temporal claim
                            claim_type = ClaimType.TEMPORAL
                            extracted_value = temp_value

                confidence, priority = self._assess_full_claim(sentence)

                target = VerificationTarget(
                    claim_text=sentence.strip(),
                    claim_type=claim_type,
                    context=sentence.strip(),
                    confidence=confidence,
                    priority=priority,
                    extracted_value=extracted_value,
                    start_pos=text.find(sentence),
                    end_pos=text.find(sentence) + len(sentence)
                )
                targets.append(target)

        return targets

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on periods, question marks, exclamation marks
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _assess_full_claim(self, claim: str) -> Tuple[float, int]:
        """Assess confidence and priority for full contextual claims."""
        confidence = 0.7
        priority = 6

        # Higher priority for claims with percentages
        if '%' in claim:
            priority = 9
            confidence = 0.85

        # Higher priority for large numbers
        if any(word in claim.lower() for word in ['millones', 'billones', 'million', 'billion', 'pib', 'gdp']):
            priority = 10
            confidence = 0.9

        # Higher priority for recent years
        current_year = datetime.now().year
        if str(current_year) in claim or str(current_year - 1) in claim:
            priority = 8
            confidence = 0.8

        # Higher priority for important topics
        important_keywords = ['pandemia', 'covid', 'coronavirus', 'guerra', 'elecciones', 'crisis', 'muerte', 'accidente']
        if any(keyword in claim.lower() for keyword in important_keywords):
            priority = max(priority, 9)  # At least 9, but don't reduce if already higher
            confidence = max(confidence, 0.85)

        # Higher confidence for claims with official sources
        official_indicators = ['según', 'datos oficiales', 'ministerio', 'gobierno', 'oms', 'onu']
        if any(indicator in claim.lower() for indicator in official_indicators):
            confidence += 0.1
            priority += 1

        return min(confidence, 1.0), min(priority, 10)

    def _extract_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Extract context around a match position."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()

    def _assess_numerical_claim(self, claim: str, context: str) -> Tuple[float, int]:
        """Assess confidence and priority for numerical claims."""
        confidence = 0.8  # High base confidence for numerical claims
        priority = 7     # High priority

        # Increase priority for large numbers or percentages
        if '%' in claim:
            priority = 9
            confidence = 0.9
        elif any(word in claim.lower() for word in ['millones', 'billones', 'million', 'billion']):
            priority = 10
            confidence = 0.95

        # Increase confidence if context suggests statistical claim
        statistical_indicators = ['estadísticas', 'datos', 'cifras', 'según', 'informe', 'estudio']
        if any(indicator in context.lower() for indicator in statistical_indicators):
            confidence += 0.1
            priority += 1

        return min(confidence, 1.0), min(priority, 10)

    def _assess_temporal_claim(self, claim: str, context: str) -> Tuple[float, int]:
        """Assess confidence and priority for temporal claims."""
        confidence = 0.7
        priority = 6

        # Higher priority for recent dates
        current_year = datetime.now().year
        if str(current_year) in claim or str(current_year - 1) in claim:
            priority = 8
            confidence = 0.85

        # Higher confidence for specific dates vs years
        if '/' in claim or '-' in claim:
            confidence = 0.9
            priority = 9

        return confidence, priority

    def _assess_attribution_claim(self, claim: str, context: str) -> Tuple[float, int]:
        """Assess confidence and priority for attribution claims."""
        confidence = 0.6
        priority = 5

        # Higher priority for official sources
        official_sources = ['gobierno', 'oms', 'onu', 'ministro', 'presidente']
        if any(source in claim.lower() for source in official_sources):
            priority = 8
            confidence = 0.8

        return confidence, priority

    def _assess_causal_claim(self, claim: str, context: str) -> Tuple[float, int]:
        """Assess confidence and priority for causal claims."""
        confidence = 0.5
        priority = 4

        # Higher priority for claims with evidence words
        evidence_words = ['pruebas', 'evidencias', 'demuestra', 'confirman']
        if any(word in context.lower() for word in evidence_words):
            priority = 6
            confidence = 0.7

        return confidence, priority

    def _assess_legal_event_claim(self, claim: str, context: str) -> Tuple[float, int]:
        """Assess confidence and priority for legal event claims."""
        confidence = 0.8  # High confidence - these are specific factual claims
        priority = 9      # High priority - legal events are important to verify

        # Even higher priority for high-profile political figures
        political_figures = ['ábalos', 'sánchez', 'iglesias', 'montero', 'casado', 'abascal', 'rivera']
        if any(figure in claim.lower() for figure in political_figures):
            priority = 10
            confidence = 0.95

        # Higher priority for serious legal actions
        serious_actions = ['prisión', 'cárcel', 'soto del real', 'condenado', 'procesado']
        if any(action in claim.lower() for action in serious_actions):
            priority = max(priority, 10)
            confidence = max(confidence, 0.9)

        # Lower confidence if claim lacks specifics
        if not any(word in claim.lower() for word in ['por', 'porqué', 'motivos', 'acusación']):
            confidence -= 0.1

        return max(confidence, 0.0), min(priority, 10)

    def _assess_breaking_news_claim(self, claim: str, context: str) -> Tuple[float, int]:
        """Assess confidence and priority for breaking news claims."""
        confidence = 0.85  # High confidence - breaking news format is suspicious
        priority = 9       # High priority - urgent claims need verification

        # Very high priority for political figures in breaking news
        political_figures = ['ábalos', 'sánchez', 'iglesias', 'montero', 'casado', 'abascal', 'rivera']
        if any(figure in claim.lower() for figure in political_figures):
            priority = 10
            confidence = 0.95

        # Check for source attribution (reduces suspicion)
        source_indicators = ['según', 'fuente', 'informa', 'confirma', 'el país', 'abc', 'eldiario']
        if any(indicator in context.lower() for indicator in source_indicators):
            confidence -= 0.2  # Less suspicious if sourced
            priority -= 1

        return max(confidence, 0.0), min(priority, 10)


def extract_verification_targets(text: str, max_targets: int = 5) -> List[VerificationTarget]:
    """
    Convenience function to extract verification targets from text.

    Args:
        text: Input text to analyze
        max_targets: Maximum number of targets to return

    Returns:
        List of VerificationTarget objects
    """
    extractor = ClaimExtractor()
    return extractor.extract_verification_targets(text, max_targets)


# Type alias for cleaner code
Claim = VerificationTarget


def extract_claims(text: str, max_targets: int = 5) -> List[VerificationTarget]:
    """
    Convenience function - extracts verification targets from text.
    """
    return extract_verification_targets(text, max_targets)