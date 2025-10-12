"""
Temporal verification functionality.
Validates date and time-based claims against authoritative sources.
"""

import re
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.models import VerificationResult, EvidenceSource, VerificationVerdict


@dataclass
class TemporalClaim:
    """A temporal claim extracted from text."""
    claim_text: str
    date_mentioned: Optional[datetime]
    time_period: Optional[str]  # e.g., "hace 3 días", "en 2020"
    temporal_type: str  # 'specific_date', 'relative_time', 'year_only', 'period'
    context: str


class TemporalVerifier:
    """
    Verifies temporal claims by checking dates and time periods against
    known events, official records, and logical consistency.
    """

    def __init__(self):
        # Known major events with their dates (for validation)
        self.known_events = {
            # COVID-19 related
            'pandemia covid': datetime(2020, 3, 11),  # WHO declares pandemic
            'confinamiento españa': datetime(2020, 3, 14),  # Spain lockdown
            'fin estado alarma': datetime(2021, 5, 9),  # End of alarm state

            # Political events
            'elecciones generales 2019': datetime(2019, 11, 10),
            'elecciones generales 2023': datetime(2023, 7, 23),
            'investidura sánchez 2020': datetime(2020, 1, 7),

            # International events
            'brexit': datetime(2020, 1, 31),
            'elecciones eeuu 2020': datetime(2020, 11, 3),
            'invasión ucrania': datetime(2022, 2, 24),
        }

        # Date patterns for extraction
        self.date_patterns = [
            # DD/MM/YYYY or DD-MM-YYYY
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
            # MM/DD/YYYY (US format, less common in Spanish)
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
            # YYYY-MM-DD (ISO format)
            r'\b(\d{4})-(\d{2})-(\d{2})\b',
        ]

        # Relative time patterns
        self.relative_time_patterns = [
            r'\bhace\s+(\d+)\s+(días?|meses?|años?|horas?)\b',
            r'\b(\d+)\s+(días?|meses?|años?|horas?)\s+(?:atrás|antes)\b',
            r'\ben\s+(?:el\s+)?(\d{4})\b',
            r'\bdurante\s+(\d{4})\b',
            r'\bdesde\s+(\d{4})\b',
        ]

    def verify_temporal_claim(self, claim_text: str, context: str = "") -> Tuple[bool, str, Optional[datetime]]:
        """
        Verify a temporal claim.

        Args:
            claim_text: The temporal claim to verify
            context: Surrounding context for better understanding

        Returns:
            (is_verified, explanation, verified_date)
        """
        # Extract temporal information
        temporal_claim = self._extract_temporal_claim(claim_text, context)

        if not temporal_claim:
            return False, "No se pudo extraer información temporal de la afirmación", None

        # Verify based on temporal type
        if temporal_claim.temporal_type == 'specific_date':
            return self._verify_specific_date(temporal_claim)
        elif temporal_claim.temporal_type == 'relative_time':
            return self._verify_relative_time(temporal_claim)
        elif temporal_claim.temporal_type == 'year_only':
            return self._verify_year_only(temporal_claim)
        elif temporal_claim.temporal_type == 'period':
            return self._verify_period(temporal_claim)
        else:
            return False, f"Tipo temporal no reconocido: {temporal_claim.temporal_type}", None

    def _extract_temporal_claim(self, claim_text: str, context: str = "") -> Optional[TemporalClaim]:
        """Extract temporal information from claim text."""
        full_text = f"{claim_text} {context}".strip()

        # Try to extract specific dates
        for pattern in self.date_patterns:
            match = re.search(pattern, full_text)
            if match:
                try:
                    if len(match.groups()) == 3:
                        # Handle different date formats
                        if pattern == self.date_patterns[0]:  # DD/MM/YYYY or DD-MM-YYYY
                            day, month, year = map(int, match.groups())
                        elif pattern == self.date_patterns[1]:  # MM/DD/YYYY
                            month, day, year = map(int, match.groups())
                        else:  # YYYY-MM-DD
                            year, month, day = map(int, match.groups())

                        # Validate date
                        if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100:
                            date_mentioned = datetime(year, month, day)
                            return TemporalClaim(
                                claim_text=claim_text,
                                date_mentioned=date_mentioned,
                                time_period=None,
                                temporal_type='specific_date',
                                context=context
                            )
                except ValueError:
                    continue  # Invalid date, try next pattern

        # Try to extract relative time
        for pattern in self.relative_time_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                return TemporalClaim(
                    claim_text=claim_text,
                    date_mentioned=None,
                    time_period=match.group(),
                    temporal_type='relative_time',
                    context=context
                )

        # Try to extract year only
        year_match = re.search(r'\b(19|20)\d{2}\b', full_text)
        if year_match:
            year = int(year_match.group())
            return TemporalClaim(
                claim_text=claim_text,
                date_mentioned=datetime(year, 1, 1),  # January 1st of that year
                time_period=f"año {year}",
                temporal_type='year_only',
                context=context
            )

        return None

    def _verify_specific_date(self, claim: TemporalClaim) -> Tuple[bool, str, Optional[datetime]]:
        """Verify a specific date claim."""
        if not claim.date_mentioned:
            return False, "Fecha no especificada", None

        now = datetime.now()
        claim_date = claim.date_mentioned

        # Check if date is in the future (impossible for past events)
        if claim_date > now:
            # Allow some tolerance for very recent claims (within 24 hours)
            if (claim_date - now).days > 1:
                return False, f"La fecha {claim_date.strftime('%d/%m/%Y')} está en el futuro", claim_date

        # Check if date corresponds to known major events
        for event_name, event_date in self.known_events.items():
            if event_name.lower() in claim.claim_text.lower():
                # Allow some tolerance (same day or very close)
                days_diff = abs((claim_date - event_date).days)
                if days_diff <= 1:  # Same day or adjacent
                    return True, f"Fecha correcta para {event_name}", claim_date
                else:
                    return False, f"Fecha incorrecta para {event_name}. La fecha correcta es {event_date.strftime('%d/%m/%Y')}", event_date

        # For unknown events, we can't verify the exact date but can check logical consistency
        # Check if it's a reasonable date (not too far in the past/future)
        years_diff = abs((now.year - claim_date.year))
        if years_diff > 50:  # More than 50 years ago
            return False, f"Fecha demasiado antigua: {claim_date.strftime('%d/%m/%Y')}", claim_date

        # Default: assume date is plausible if it passes basic checks
        return True, f"Fecha plausible: {claim_date.strftime('%d/%m/%Y')}", claim_date

    def _verify_relative_time(self, claim: TemporalClaim) -> Tuple[bool, str, Optional[datetime]]:
        """Verify a relative time claim like 'hace 3 días'."""
        if not claim.time_period:
            return False, "Periodo temporal no especificado", None

        # Parse the relative time
        time_match = re.search(r'(\d+)\s+(día|mes|año|hora)', claim.time_period, re.IGNORECASE)
        if not time_match:
            return False, f"No se pudo parsear el periodo temporal: {claim.time_period}", None

        quantity = int(time_match.group(1))
        unit = time_match.group(2).lower()

        # Convert to days
        if 'día' in unit:
            days_ago = quantity
        elif 'mes' in unit:
            days_ago = quantity * 30  # Approximate
        elif 'año' in unit:
            days_ago = quantity * 365  # Approximate
        elif 'hora' in unit:
            days_ago = quantity / 24  # Convert to days
        else:
            return False, f"Unidad de tiempo no reconocida: {unit}", None

        # Calculate the implied date
        implied_date = datetime.now() - timedelta(days=days_ago)

        # Check if this makes sense in context
        if days_ago > 365 * 10:  # More than 10 years ago
            return False, f"Periodo temporal demasiado largo: {claim.time_period}", implied_date

        if days_ago < 0:  # Future dates
            return False, f"Periodo temporal en el futuro: {claim.time_period}", implied_date

        return True, f"Periodo temporal plausible: {claim.time_period}", implied_date

    def _verify_year_only(self, claim: TemporalClaim) -> Tuple[bool, str, Optional[datetime]]:
        """Verify a year-only claim."""
        if not claim.date_mentioned:
            return False, "Año no especificado", None

        year = claim.date_mentioned.year
        current_year = datetime.now().year

        # Check if year is reasonable
        if year < 1900:
            return False, f"Año demasiado antiguo: {year}", claim.date_mentioned
        elif year > current_year + 2:  # Allow some future tolerance
            return False, f"Año en el futuro: {year}", claim.date_mentioned

        return True, f"Año plausible: {year}", claim.date_mentioned

    def _verify_period(self, claim: TemporalClaim) -> Tuple[bool, str, Optional[datetime]]:
        """Verify a time period claim."""
        # For now, be permissive with period claims
        # Could be improved with more sophisticated period validation
        return True, f"Periodo temporal aceptado: {claim.time_period}", claim.date_mentioned

    def validate_event_timeline(self, events: List[Tuple[str, datetime]]) -> List[str]:
        """
        Validate that a series of events follows a logical timeline.

        Args:
            events: List of (event_description, event_date) tuples

        Returns:
            List of validation issues found
        """
        issues = []

        # Sort events by date
        sorted_events = sorted(events, key=lambda x: x[1])

        # Check for logical inconsistencies
        for i in range(len(sorted_events) - 1):
            current_event, current_date = sorted_events[i]
            next_event, next_date = sorted_events[i + 1]

            # Check if events are in chronological order
            if current_date > next_date:
                issues.append(f"Orden cronológico incorrecto: '{current_event}' ocurre después de '{next_event}'")

            # Check for impossible time gaps (events happening seconds apart when they shouldn't)
            time_diff = next_date - current_date
            if time_diff.total_seconds() < 60 and "elección" in current_event.lower() and "resultado" in next_event.lower():
                issues.append(f"Intervalo temporal imposible entre '{current_event}' y '{next_event}'")

        return issues

    def cross_reference_with_known_events(self, claim_text: str) -> List[Tuple[str, datetime, int]]:
        """
        Cross-reference claim with known events to find temporal anchors.

        Returns:
            List of (event_name, event_date, confidence_score) tuples
        """
        matches = []
        claim_lower = claim_text.lower()

        for event_name, event_date in self.known_events.items():
            if event_name in claim_lower:
                # Calculate confidence based on how closely the event name matches
                confidence = 90 if event_name in claim_lower else 70
                matches.append((event_name, event_date, confidence))

        return matches


def verify_temporal_claim(claim_text: str, context: str = "") -> Tuple[bool, str, Optional[datetime]]:
    """
    Convenience function to verify a temporal claim.

    Args:
        claim_text: The temporal claim to verify
        context: Surrounding context

    Returns:
        (is_verified, explanation, verified_date)
    """
    verifier = TemporalVerifier()
    return verifier.verify_temporal_claim(claim_text, context)