"""
Dedicated numerical claim verification.
Specialized verification for statistical, numerical, and quantitative claims.
"""

import re
import math
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..core.models import VerificationResult, EvidenceSource, VerificationVerdict
from ..sources.statistical_apis import StatisticalAPIManager


class NumericalClaimType(Enum):
    """Types of numerical claims."""
    POPULATION = "population"
    ECONOMIC = "economic"  # GDP, inflation, unemployment
    HEALTH = "health"      # COVID cases, vaccination rates
    POLITICAL = "political"  # Election results, polling data
    GENERAL = "general"    # Other numerical claims


@dataclass
class NumericalClaim:
    """A parsed numerical claim."""
    original_text: str
    value: Union[int, float]
    unit: str  # millions, billions, percent, etc.
    claim_type: NumericalClaimType
    context: str
    time_reference: Optional[str] = None  # year, month, etc.
    confidence: float = 0.8


@dataclass
class NumericalVerificationResult:
    """Result of numerical claim verification."""
    claim: NumericalClaim
    verdict: VerificationVerdict
    confidence: float
    expected_value: Optional[Union[int, float]] = None
    actual_value: Optional[Union[int, float]] = None
    source: Optional[str] = None
    explanation: str = ""
    margin_of_error: Optional[float] = None


class NumericalClaimParser:
    """
    Parses numerical claims from text.
    Extracts numbers, units, and context for verification.
    """

    def __init__(self):
        # Patterns for different numerical formats
        self.number_patterns = [
            # Large numbers with commas: 1,234,567 or 1234567
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',
            # Numbers with decimals: 15.5, 2.3
            r'\b\d+\.\d+\b',
            # Simple integers
            r'\b\d+\b',
        ]

        # Unit patterns
        self.unit_patterns = {
            'million': r'millon(?:es)?',
            'billion': r'billon(?:es)?',
            'thousand': r'mil',
            'percent': r'%|por ciento|porcentaje',
            'euro': r'euros?|€',
            'dollar': r'dólares?|\$',
            'people': r'personas?|habitantes?',
        }

        # Time reference patterns
        self.time_patterns = [
            r'\b20\d{2}\b',  # years like 2023
            r'\b\d{4}\b',    # years like 2023
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # dates
        ]

    def parse_numerical_claims(self, text: str) -> List[NumericalClaim]:
        """
        Parse all numerical claims from text.

        Args:
            text: Text to parse

        Returns:
            List of parsed numerical claims
        """
        claims = []

        # Find all numbers in text
        for pattern in self.number_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim = self._parse_single_claim(text, match)
                if claim:
                    claims.append(claim)

        # Remove duplicates based on value and position
        unique_claims = []
        seen = set()
        for claim in claims:
            key = (claim.value, claim.original_text)
            if key not in seen:
                unique_claims.append(claim)
                seen.add(key)

        return unique_claims

    def _parse_single_claim(self, text: str, match) -> Optional[NumericalClaim]:
        """Parse a single numerical claim from a regex match."""
        number_str = match.group()
        start_pos = match.start()
        end_pos = match.end()

        # Skip obvious years (4 digits starting with 19 or 20)
        if re.match(r'^(19|20)\d{2}$', number_str):
            return None

        # Clean the number
        clean_number = number_str.replace(',', '')
        try:
            value = float(clean_number)
        except ValueError:
            return None

        # Extract context (surrounding text)
        context_start = max(0, start_pos - 100)
        context_end = min(len(text), end_pos + 100)
        context = text[context_start:context_end].strip()

        # Check for multipliers in the context and adjust value
        multipliers = {
            'trillones': 1000000000000,
            'trillón': 1000000000000,
            'billones': 1000000000,
            'billón': 1000000000,
            'millones': 1000000,
            'millón': 1000000,
            'mil': 1000,
        }

        # Sort by length (longest first) to avoid substring matches
        sorted_multipliers = sorted(multipliers.items(), key=lambda x: len(x[0]), reverse=True)

        multiplier_found = None
        for word, multiplier in sorted_multipliers:
            if re.search(r'\b' + re.escape(word) + r'\b', context.lower()):
                # Find the position of the multiplier word relative to our number
                word_match = re.search(r'\b' + re.escape(word) + r'\b', context.lower())
                if word_match:
                    # Check if the multiplier is within 30 characters of our number
                    number_pos_in_context = start_pos - context_start
                    word_pos_in_context = word_match.start()
                    distance = abs(number_pos_in_context - word_pos_in_context)
                    if distance <= 30:  # Multiplier is close to our number
                        multiplier_found = word
                        break

        # Don't multiply the value here - keep it raw and let unit determination handle it
        # The unit will be determined separately

        # Determine unit
        unit = self._determine_unit(text, start_pos, end_pos)

        # Determine claim type
        claim_type = self._determine_claim_type(context)

        # Extract time reference
        time_ref = self._extract_time_reference(context)

        # Calculate confidence based on context clarity
        confidence = self._calculate_confidence(context, unit, time_ref)

        return NumericalClaim(
            original_text=number_str,
            value=value,
            unit=unit,
            claim_type=claim_type,
            context=context,
            time_reference=time_ref,
            confidence=confidence
        )

    def _determine_unit(self, text: str, start_pos: int, end_pos: int) -> str:
        """Determine the unit of the numerical value."""
        # Look for units in surrounding context
        context_window = text[max(0, start_pos-20):min(len(text), end_pos+20)].lower()

        for unit_name, pattern in self.unit_patterns.items():
            if re.search(pattern, context_window, re.IGNORECASE):
                return unit_name

        # Check for common Spanish units
        if any(word in context_window for word in ['habitantes', 'personas', 'población']):
            return 'people'
        elif any(word in context_window for word in ['€', 'euros']):
            return 'euro'
        elif '%' in context_window:
            return 'percent'

        return 'unknown'

    def _determine_claim_type(self, context: str) -> NumericalClaimType:
        """Determine the type of numerical claim."""
        context_lower = context.lower()

        if any(word in context_lower for word in ['población', 'habitantes', 'personas', 'millones']):
            return NumericalClaimType.POPULATION
        elif any(word in context_lower for word in ['pib', 'gdp', 'producto interior bruto', 'desempleo', 'paro', 'unemployment', 'inflación', 'ipc']):
            return NumericalClaimType.ECONOMIC
        elif any(word in context_lower for word in ['covid', 'vacuna', 'casos', 'muertes', 'salud']):
            return NumericalClaimType.HEALTH
        elif any(word in context_lower for word in ['elecciones', 'votos', 'escaños', 'partido']):
            return NumericalClaimType.POLITICAL

        return NumericalClaimType.GENERAL

    def _extract_time_reference(self, context: str) -> Optional[str]:
        """Extract time reference from context."""
        for pattern in self.time_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group()
        return None

    def _calculate_confidence(self, context: str, unit: str, time_ref: Optional[str]) -> float:
        """Calculate confidence score for the claim."""
        confidence = 0.5  # Base confidence

        # Higher confidence with clear units
        if unit != 'unknown':
            confidence += 0.2

        # Higher confidence with time references
        if time_ref:
            confidence += 0.2

        # Higher confidence with specific context words
        context_indicators = ['según', 'datos', 'oficial', 'informe', 'estudio']
        if any(word in context.lower() for word in context_indicators):
            confidence += 0.1

        return min(confidence, 1.0)


class NumericalVerifier:
    """
    Dedicated verifier for numerical claims.
    Uses statistical APIs and known data sources for verification.
    """

    def __init__(self):
        self.parser = NumericalClaimParser()
        self.statistical_api = StatisticalAPIManager()

        # Known reference values for common claims (fallback when APIs fail)
        self.reference_data = {
            'population_spain_2023': {'value': 47.4, 'unit': 'million', 'source': 'INE'},
            'population_spain_2024': {'value': 47.6, 'unit': 'million', 'source': 'INE'},
            'unemployment_spain_2023': {'value': 11.8, 'unit': 'percent', 'source': 'INE'},
            'gdp_spain_2023': {'value': 1.45, 'unit': 'trillion_euro', 'source': 'Eurostat'},
        }

    def verify_numerical_claim(self, text: str, full_context: Optional[str] = None) -> List[NumericalVerificationResult]:
        """
        Verify all numerical claims in the given text.

        Args:
            text: Text containing numerical claims
            full_context: Full original text for better statistical API matching

        Returns:
            List of verification results
        """
        claims = self.parser.parse_numerical_claims(text)
        results = []

        for claim in claims:
            result = self._verify_single_claim(claim, full_context or text)
            results.append(result)

        return results

    def _verify_single_claim(self, claim: NumericalClaim, full_context: str) -> NumericalVerificationResult:
        """Verify a single numerical claim."""
        # Try statistical API first
        api_result = self._check_statistical_api(claim, full_context)
        if api_result:
            return api_result

        # Fall back to reference data
        reference_result = self._check_reference_data(claim)
        if reference_result:
            return reference_result

        # If no data available, return unverified
        return NumericalVerificationResult(
            claim=claim,
            verdict=VerificationVerdict.UNVERIFIED,
            confidence=0.0,
            explanation="No se encontraron datos para verificar esta afirmación numérica"
        )

    def _check_statistical_api(self, claim: NumericalClaim, full_context: str) -> Optional[NumericalVerificationResult]:
        """Check claim against statistical APIs."""
        # Only check statistical APIs for claims that are likely to have statistical data
        if claim.claim_type not in [NumericalClaimType.POPULATION, NumericalClaimType.ECONOMIC, NumericalClaimType.HEALTH]:
            return None

        try:
            # Use the full context text that contains keywords for better API matching
            # Query statistical APIs with the full context and the numerical value
            is_verified, data_point = self.statistical_api.verify_numerical_claim(
                full_context, str(int(claim.value)) if claim.value.is_integer() else str(claim.value)
            )

            if data_point:
                # Compare values
                expected_value = self._extract_value_from_data_point(data_point)
                actual_value = claim.value

                if expected_value is not None:
                    # Calculate difference
                    if expected_value > 0:
                        difference_pct = abs(actual_value - expected_value) / expected_value
                        margin_of_error = 0.05  # 5% margin for statistical data

                        if difference_pct <= margin_of_error:
                            verdict = VerificationVerdict.VERIFIED
                            confidence = min(0.9, 1.0 - difference_pct)
                            explanation = f"Verificado contra {data_point.source_name}: valor esperado {expected_value}, afirmado {actual_value}"
                        else:
                            verdict = VerificationVerdict.DEBUNKED
                            confidence = min(0.8, difference_pct)
                            explanation = f"Desmentido por {data_point.source_name}: valor real {expected_value}, afirmado {actual_value}"
                    else:
                        verdict = VerificationVerdict.UNVERIFIED
                        confidence = 0.3
                        explanation = f"Datos disponibles pero no comparables: {data_point.description}"

                    return NumericalVerificationResult(
                        claim=claim,
                        verdict=verdict,
                        confidence=confidence,
                        expected_value=expected_value,
                        actual_value=actual_value,
                        source=data_point.source_name,
                        explanation=explanation,
                        margin_of_error=margin_of_error if 'margin_of_error' in locals() else None
                    )

        except Exception as e:
            print(f"Statistical API check failed: {e}")

        return None

    def _check_reference_data(self, claim: NumericalClaim) -> Optional[NumericalVerificationResult]:
        """Check claim against known reference data."""
        # Create lookup key based on claim type and time
        key_parts = []

        if claim.claim_type == NumericalClaimType.POPULATION:
            key_parts.append('population_spain')
        elif claim.claim_type == NumericalClaimType.ECONOMIC:
            if 'paro' in claim.context.lower() or 'desempleo' in claim.context.lower():
                key_parts.append('unemployment_spain')
            elif 'pib' in claim.context.lower():
                key_parts.append('gdp_spain')

        if claim.time_reference:
            # Extract year from time reference
            year_match = re.search(r'\b(20\d{2})\b', claim.time_reference)
            if year_match:
                key_parts.append(year_match.group(1))

        if key_parts:
            lookup_key = '_'.join(key_parts)
            reference = self.reference_data.get(lookup_key)

            if reference:
                expected_value = reference['value']
                actual_value = claim.value

                # Adjust for units
                expected_adjusted = self._adjust_for_units(expected_value, reference['unit'], claim.unit)

                if expected_adjusted is not None:
                    difference_pct = abs(actual_value - expected_adjusted) / expected_adjusted

                    if difference_pct <= 0.05:  # 5% margin
                        return NumericalVerificationResult(
                            claim=claim,
                            verdict=VerificationVerdict.VERIFIED,
                            confidence=0.8,
                            expected_value=expected_adjusted,
                            actual_value=actual_value,
                            source=reference['source'],
                            explanation=f"Verificado contra datos de referencia: valor real {expected_adjusted}"
                        )
                    else:
                        return NumericalVerificationResult(
                            claim=claim,
                            verdict=VerificationVerdict.DEBUNKED,
                            confidence=0.7,
                            expected_value=expected_adjusted,
                            actual_value=actual_value,
                            source=reference['source'],
                            explanation=f"Desmentido por datos oficiales: valor real {expected_adjusted}"
                        )

        return None

    def _extract_value_from_data_point(self, data_point) -> Optional[float]:
        """Extract numerical value from data point."""
        # This would depend on the structure of data_point
        # Simplified implementation
        if hasattr(data_point, 'value'):
            return float(data_point.value)
        elif hasattr(data_point, 'data'):
            # Try to extract from description or other fields
            return None
        return None

    def _adjust_for_units(self, value: float, source_unit: str, claim_unit: str) -> Optional[float]:
        """Adjust value for unit differences."""
        # Handle common unit conversions
        if source_unit == claim_unit:
            return value

        # Million to absolute
        if source_unit == 'million' and claim_unit == 'unknown':
            return value * 1_000_000
        elif source_unit == 'million' and claim_unit == 'people':
            return value * 1_000_000

        # Billion to million
        if source_unit == 'billion' and claim_unit == 'million':
            return value * 1000

        # For now, return None if units don't match
        # In production, would implement more comprehensive unit conversion
        return None


# Convenience functions
def verify_numerical_claims(text: str) -> List[NumericalVerificationResult]:
    """
    Convenience function to verify numerical claims in text.

    Args:
        text: Text containing numerical claims

    Returns:
        List of verification results
    """
    verifier = NumericalVerifier()
    return verifier.verify_numerical_claim(text)


def parse_numerical_claims(text: str) -> List[NumericalClaim]:
    """
    Convenience function to parse numerical claims from text.

    Args:
        text: Text to parse

    Returns:
        List of parsed numerical claims
    """
    parser = NumericalClaimParser()
    return parser.parse_numerical_claims(text)