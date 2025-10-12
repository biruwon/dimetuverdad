"""
Statistical API clients for verifying numerical claims.
Connects to official data sources for real-time verification.
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import re

from .http_client import HttpClient, create_http_client


@dataclass
class StatisticalDataPoint:
    """A data point from a statistical source."""
    value: Any
    year: int
    month: Optional[int] = None
    quarter: Optional[int] = None
    source: str = ""
    source_name: str = ""  # Added for compatibility
    source_url: str = ""   # Added for compatibility
    title: str = ""        # Added for compatibility
    description: str = ""  # Added for compatibility
    indicator_name: str = ""
    unit: str = ""
    last_updated: Optional[datetime] = None


class INEClient:
    """Client for Spanish National Statistics Institute (INE) API."""

    BASE_URL = "https://servicios.ine.es/wstempus/js/ES"

    def __init__(self):
        self.http_client = HttpClient(language="es")

    def search_population_data(self, year: Optional[int] = None) -> List[StatisticalDataPoint]:
        """
        Search for population data from INE.

        Args:
            year: Specific year to search for, or None for latest

        Returns:
            List of population data points
        """
        try:
            # INE Population data - use their API
            # Operation code for total population: 2915 (Cifras de población)
            operation_code = "2915"

            # Build API URL for population data
            url = f"{self.BASE_URL}/DATOS_TABLA/{operation_code}"

            params = {
                'nult': '1',  # Get latest data
                'det': '2'    # Detailed data
            }

            response = self.http_client.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            results = []
            if data and len(data) > 0:
                # Parse INE response format
                for item in data:
                    if 'Data' in item and len(item['Data']) > 0:
                        latest_data = item['Data'][0]  # Most recent data

                        # Extract population value
                        value_str = str(latest_data.get('Valor', ''))
                        if value_str and value_str != 'None':
                            try:
                                # Remove commas and convert to int
                                population = int(value_str.replace(',', '').replace('.', ''))

                                # Extract year
                                period = latest_data.get('Anyo', str(datetime.now().year))

                                results.append(StatisticalDataPoint(
                                    value=population,
                                    year=int(period),
                                    source="INE",
                                    source_name="INE - Instituto Nacional de Estadística",
                                    source_url="https://www.ine.es",
                                    title="Cifras oficiales de población",
                                    description=f"Población total de España: {population:,} habitantes",
                                    indicator_name="Población total de España",
                                    unit="personas",
                                    last_updated=datetime.now()
                                ))
                            except (ValueError, KeyError) as e:
                                self.logger.warning(f"Error parsing INE population data: {e}")
                                continue

            # If API call fails or returns no data, return empty list
            # (Don't fall back to mock data)
            return results

        except Exception as e:
            print(f"INE API error: {e}")
            return []

    def verify_population_claim(self, claimed_population: int,
                              year: Optional[int] = None,
                              tolerance_percent: float = 5.0) -> Tuple[bool, Optional[StatisticalDataPoint]]:
        """
        Verify a population claim against INE data.

        Args:
            claimed_population: The claimed population number
            year: Year for the claim
            tolerance_percent: Acceptable margin of error

        Returns:
            (is_verified, official_data_point)
        """
        data_points = self.search_population_data(year)

        for point in data_points:
            if isinstance(point.value, (int, float)):
                official_value = float(point.value)
                claimed_value = float(claimed_population)

                # Check if within tolerance
                tolerance = official_value * (tolerance_percent / 100)
                if abs(official_value - claimed_value) <= tolerance:
                    return True, point

        return False, None


class EurostatClient:
    """Client for Eurostat statistical data."""

    BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination"

    def __init__(self):
        self.http_client = create_http_client()

    def search_economic_indicators(self, indicator_code: str = "GDP",
                                 year: Optional[int] = None) -> List[StatisticalDataPoint]:
        """
        Search for economic indicators from Eurostat.

        Args:
            indicator_code: Type of indicator (GDP, unemployment, etc.)
            year: Specific year

        Returns:
            List of economic data points
        """
        try:
            # Eurostat GDP data for Spain
            # Dataset: nama_10_gdp (GDP and main components)
            dataset = "nama_10_gdp"

            # Build Eurostat API URL
            url = f"{self.BASE_URL}/statistics/1.0/data/{dataset}"

            params = {
                'format': 'JSON',
                'lang': 'en',
                'geo': 'ES',  # Spain
                'unit': 'CP_MEUR',  # Current prices, million euro
                'na_item': 'B1GQ'  # Gross domestic product at market prices
            }

            if year:
                params['time'] = str(year)

            response = self.http_client.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            results = []
            if 'value' in data:
                values = data['value']
                dimensions = data.get('dimension', {})
                time_dimension = dimensions.get('time', {}).get('category', {}).get('index', {})

                # Create reverse mapping: index -> year
                index_to_year = {str(idx): year for year, idx in time_dimension.items()}

                # Eurostat returns data in a specific format
                for time_key, value in values.items():
                    year_str = index_to_year.get(time_key)
                    if year_str:
                        try:
                            year_int = int(year_str)
                            gdp_value = float(value) * 1000000  # Convert from million euro to euro

                            results.append(StatisticalDataPoint(
                                value=gdp_value,
                                year=year_int,
                                source="Eurostat",
                                source_name="Eurostat - Oficina Estadística de la Unión Europea",
                                source_url="https://ec.europa.eu/eurostat",
                                title="Producto Interior Bruto de España",
                                description=f"PIB España {year_int}: {gdp_value:,.0f} EUR",
                                indicator_name="PIB España (precios corrientes)",
                                unit="EUR",
                                last_updated=datetime.now()
                            ))
                        except (ValueError, KeyError):
                            continue

                # Sort by year descending and return most recent
                results.sort(key=lambda x: x.year, reverse=True)

            return results[:5]  # Return up to 5 most recent data points

        except Exception as e:
            print(f"Eurostat API error: {e}")
            return []

    def search_gdp_growth_rates(self, year: Optional[int] = None) -> List[StatisticalDataPoint]:
        """
        Search for GDP growth rates from Eurostat.

        Args:
            year: Specific year

        Returns:
            List of GDP growth rate data points
        """
        try:
            # Eurostat GDP growth rates
            # Dataset: tec00115 (Real GDP growth rate - volume)
            dataset = "tec00115"

            url = f"{self.BASE_URL}/statistics/1.0/data/{dataset}"

            params = {
                'format': 'JSON',
                'lang': 'en',
                'geo': 'ES',  # Spain
                'unit': 'CLV_PCH_PRE',  # Chain linked volumes, percentage change on previous period
            }

            if year:
                params['time'] = str(year)

            response = self.http_client.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()

            results = []
            if 'value' in data:
                values = data['value']
                dimensions = data.get('dimension', {})
                time_dimension = dimensions.get('time', {}).get('category', {}).get('index', {})

                # Create reverse mapping: index -> year
                index_to_year = {str(idx): year for year, idx in time_dimension.items()}

                for time_key, value in values.items():
                    year_str = index_to_year.get(time_key)
                    if year_str:
                        try:
                            year_int = int(year_str)
                            growth_rate = float(value)

                            results.append(StatisticalDataPoint(
                                value=growth_rate,
                                year=year_int,
                                source="Eurostat",
                                source_name="Eurostat - Oficina Estadística de la Unión Europea",
                                source_url="https://ec.europa.eu/eurostat",
                                title="Tasa de crecimiento del PIB real - España",
                                description=f"Crecimiento PIB España {year_int}: {growth_rate:.1f}%",
                                indicator_name="Tasa de crecimiento del PIB real - España",
                                unit="%",
                                last_updated=datetime.now()
                            ))
                        except (ValueError, KeyError):
                            continue

                # Sort by year descending
                results.sort(key=lambda x: x.year, reverse=True)

            return results[:5]  # Return up to 5 most recent

        except Exception as e:
            print(f"Eurostat GDP growth API error: {e}")
            return []

    def verify_gdp_claim(self, claimed_gdp: float,
                        year: Optional[int] = None,
                        tolerance_percent: float = 10.0) -> Tuple[bool, Optional[StatisticalDataPoint]]:
        """
        Verify a GDP claim against Eurostat data.

        Args:
            claimed_gdp: Claimed GDP value
            year: Year for the claim
            tolerance_percent: Acceptable margin

        Returns:
            (is_verified, official_data_point)
        """
        data_points = self.search_economic_indicators("GDP", year)

        for point in data_points:
            if isinstance(point.value, (int, float)):
                official_value = float(point.value)
                claimed_value = float(claimed_gdp)

                tolerance = official_value * (tolerance_percent / 100)
                if abs(official_value - claimed_value) <= tolerance:
                    return True, point

        return False, None

    def verify_gdp_growth_claim(self, claimed_growth: float,
                               year: Optional[int] = None,
                               tolerance_percent: float = 5.0) -> Tuple[bool, Optional[StatisticalDataPoint]]:
        """
        Verify a GDP growth rate claim against Eurostat data.

        Args:
            claimed_growth: Claimed GDP growth rate (percentage)
            year: Year for the claim
            tolerance_percent: Acceptable margin

        Returns:
            (is_verified, official_data_point)
        """
        data_points = self.search_gdp_growth_rates(year)

        for point in data_points:
            if isinstance(point.value, (int, float)):
                official_value = float(point.value)
                claimed_value = float(claimed_growth)

                tolerance = max(0.5, abs(official_value) * (tolerance_percent / 100))  # Minimum tolerance of 0.5%
                if abs(official_value - claimed_value) <= tolerance:
                    return True, point

        return False, None


class WHOClient:
    """Client for World Health Organization data."""

    BASE_URL = "https://ghoapi.azureedge.net/api"

    def __init__(self):
        self.http_client = create_http_client()

    def search_health_indicators(self, indicator_code: str = "WHOSIS",
                               year: Optional[int] = None) -> List[StatisticalDataPoint]:
        """
        Search for health indicators from WHO.

        Args:
            indicator_code: Type of health indicator
            year: Specific year

        Returns:
            List of health data points
        """
        try:
            # WHO Life expectancy data
            # Indicator: WHOSIS_000001 (Life expectancy at birth)
            indicator = "WHOSIS_000001"

            url = f"{self.BASE_URL}/{indicator}"

            params = {
                'filter': 'COUNTRY:ESP',  # Spain
                '$orderby': 'Year desc',  # Most recent first
                '$top': '5'  # Limit results
            }

            response = self.http_client.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            results = []
            if 'value' in data:
                for item in data['value']:
                    try:
                        year_val = item.get('Year')
                        life_expectancy = item.get('NumericValue')

                        if year_val and life_expectancy:
                            results.append(StatisticalDataPoint(
                                value=float(life_expectancy),
                                year=int(year_val),
                                source="WHO",
                                source_name="World Health Organization (WHO)",
                                source_url="https://www.who.int",
                                title="Esperanza de vida al nacer",
                                description=f"Esperanza de vida España {year_val}: {life_expectancy} años",
                                indicator_name="Esperanza de vida al nacer - España",
                                unit="años",
                                last_updated=datetime.now()
                            ))
                    except (ValueError, KeyError, TypeError):
                        continue

            return results

        except Exception as e:
            print(f"WHO API error: {e}")
            return []

    def verify_life_expectancy_claim(self, claimed_expectancy: float,
                                   year: Optional[int] = None,
                                   tolerance_percent: float = 3.0) -> Tuple[bool, Optional[StatisticalDataPoint]]:
        """
        Verify a life expectancy claim against WHO data.

        Args:
            claimed_expectancy: Claimed life expectancy
            year: Year for the claim
            tolerance_percent: Acceptable margin

        Returns:
            (is_verified, official_data_point)
        """
        data_points = self.search_health_indicators("LIFE_EXPECTANCY", year)

        for point in data_points:
            if isinstance(point.value, (int, float)):
                official_value = float(point.value)
                claimed_value = float(claimed_expectancy)

                tolerance = official_value * (tolerance_percent / 100)
                if abs(official_value - claimed_value) <= tolerance:
                    return True, point

        return False, None


class WorldBankClient:
    """Client for World Bank Open Data."""

    BASE_URL = "https://api.worldbank.org/v2"

    def __init__(self):
        self.http_client = create_http_client()

    def search_development_indicators(self, indicator_code: str = "NY.GDP.MKTP.CD",
                                    country_code: str = "ESP",
                                    year: Optional[int] = None) -> List[StatisticalDataPoint]:
        """
        Search for development indicators from World Bank.

        Args:
            indicator_code: World Bank indicator code
            country_code: ISO country code
            year: Specific year

        Returns:
            List of development data points
        """
        try:
            # World Bank GDP per capita data
            # NY.GDP.PCAP.CD = GDP per capita (current US$)
            url = f"{self.BASE_URL}/country/{country_code}/indicator/{indicator_code}"

            params = {
                'format': 'json',
                'per_page': '10',  # Get last 10 years
                'date': year if year else '2010:2023'  # Date range
            }

            response = self.http_client.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            results = []
            if isinstance(data, list) and len(data) > 1:
                # World Bank API returns [metadata, data_array]
                indicators = data[1]

                for item in indicators:
                    try:
                        if item.get('value') is not None:
                            year_val = item.get('date')
                            gdp_value = item.get('value')

                            if year_val and gdp_value:
                                results.append(StatisticalDataPoint(
                                    value=float(gdp_value),
                                    year=int(year_val),
                                    source="World Bank",
                                    source_name="World Bank Open Data",
                                    source_url="https://data.worldbank.org",
                                    title="PIB per cápita (USD corrientes)",
                                    description=f"PIB per cápita España {year_val}: ${gdp_value:,.0f} USD",
                                    indicator_name="PIB per cápita (USD corrientes) - España",
                                    unit="USD",
                                    last_updated=datetime.now()
                                ))
                    except (ValueError, KeyError, TypeError):
                        continue

                # Sort by year descending
                results.sort(key=lambda x: x.year, reverse=True)

            return results[:5]  # Return up to 5 most recent

        except Exception as e:
            print(f"World Bank API error: {e}")
            return []

    def verify_gdp_per_capita_claim(self, claimed_gdp_pc: float,
                                   country_code: str = "ESP",
                                   year: Optional[int] = None,
                                   tolerance_percent: float = 15.0) -> Tuple[bool, Optional[StatisticalDataPoint]]:
        """
        Verify a GDP per capita claim against World Bank data.

        Args:
            claimed_gdp_pc: Claimed GDP per capita
            country_code: ISO country code
            year: Year for the claim
            tolerance_percent: Acceptable margin

        Returns:
            (is_verified, official_data_point)
        """
        data_points = self.search_development_indicators("NY.GDP.PCAP.CD", country_code, year)

        for point in data_points:
            if isinstance(point.value, (int, float)):
                official_value = float(point.value)
                claimed_value = float(claimed_gdp_pc)

                tolerance = official_value * (tolerance_percent / 100)
                if abs(official_value - claimed_value) <= tolerance:
                    return True, point

        return False, None


class StatisticalAPIManager:
    """
    Manager for multiple statistical API clients.
    Provides unified interface for numerical claim verification.
    """

    def __init__(self):
        self.clients = {
            'ine': INEClient(),
            'eurostat': EurostatClient(),
            'who': WHOClient(),
            'worldbank': WorldBankClient()
        }

    def verify_numerical_claim(self, claim_text: str, extracted_value: str) -> Tuple[bool, Optional[StatisticalDataPoint]]:
        """
        Attempt to verify a numerical claim using appropriate statistical APIs.

        Args:
            claim_text: The full claim text
            extracted_value: The numerical value extracted from the claim

        Returns:
            (is_verified, data_point_if_verified)
        """
        # Parse the numerical value, considering multipliers in the full claim text
        try:
            # First try to parse just the extracted value
            numerical_value = self._parse_numerical_value(extracted_value)
            
            # Check if extracted_value is a plain number (no multipliers)
            try:
                plain_value = float(extracted_value.replace(',', '').replace(' ', ''))
                # If the extracted value is plain and matches what we parsed, use it
                if numerical_value == plain_value:
                    # Try parsing the full claim text for additional context/multipliers
                    full_parsed = self._parse_numerical_value(claim_text)
                    if full_parsed != plain_value:
                        numerical_value = full_parsed
            except ValueError:
                # extracted_value contains non-numeric characters (like multipliers), 
                # so our parsed value from _parse_numerical_value is correct
                pass
        except ValueError:
            return False, None

        # Determine claim type and route to appropriate client
        claim_lower = claim_text.lower()

        # Population claims
        if any(word in claim_lower for word in ['población', 'habitantes', 'personas', 'millones']):
            is_verified, data_point = self.clients['ine'].verify_population_claim(
                int(numerical_value) if numerical_value.is_integer() else int(numerical_value)
            )
            if is_verified:
                return True, data_point

        # GDP claims
        if any(word in claim_lower for word in ['pib', 'gdp', 'producto interior bruto']):
            # Check if this is a growth/percentage claim
            if any(growth_word in claim_lower for growth_word in ['creció', 'crecimiento', 'aumento', 'growth', 'increased', '%']):
                is_verified, data_point = self.clients['eurostat'].verify_gdp_growth_claim(numerical_value)
            elif 'per cápita' in claim_lower or 'capita' in claim_lower:
                is_verified, data_point = self.clients['worldbank'].verify_gdp_per_capita_claim(numerical_value)
            else:
                is_verified, data_point = self.clients['eurostat'].verify_gdp_claim(numerical_value)
            if is_verified:
                return True, data_point

        # Health/life expectancy claims
        if any(word in claim_lower for word in ['esperanza de vida', 'life expectancy', 'mortalidad']):
            is_verified, data_point = self.clients['who'].verify_life_expectancy_claim(numerical_value)
            if is_verified:
                return True, data_point

        return False, None

    async def query_all_sources(self, query: str, language: str = "es") -> List[Dict[str, Any]]:
        """
        Query all statistical sources for a given query.
        This method provides compatibility with the API interface.
        """
        # Extract numerical values from the query
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', query)

        if not numbers:
            return []

        results = []

        # Try to verify the claim using our existing methods
        for number in numbers:
            is_verified, data_point = self.verify_numerical_claim(query, number)
            if data_point:
                results.append({
                    'source_name': data_point.source_name,
                    'source_url': data_point.source_url,
                    'title': data_point.title,
                    'description': data_point.description,
                    'relevance_score': 90 if is_verified else 50,
                    'verified': is_verified
                })

        return results

    def _parse_numerical_value(self, value_str: str) -> float:
        """
        Parse a numerical value string into a float.

        Handles percentages, commas, and multipliers.
        """
        original_str = value_str  # Keep original for multiplier detection

        # Remove percentage signs and convert to decimal if needed
        if '%' in value_str:
            value_str = value_str.replace('%', '')
            # Note: percentages would need context to convert properly

        # Handle multipliers (millones, billones, etc.) - check longest first
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

        for word, multiplier in sorted_multipliers:
            if re.search(r'\b' + re.escape(word) + r'\b', original_str.lower()):
                # Extract the number part - look for digits near the multiplier word
                # Find the position of the multiplier word
                word_match = re.search(r'\b' + re.escape(word) + r'\b', original_str.lower())
                if word_match:
                    # Look for numbers within 20 characters before the multiplier
                    start_pos = max(0, word_match.start() - 20)
                    context = original_str[start_pos:word_match.end()]
                    num_match = re.search(r'(\d+(?:[.,]\d+)*)', context)
                    if num_match:
                        num_str = num_match.group(1).replace(',', '').replace(' ', '')
                        return float(num_str) * multiplier

        # Remove commas and spaces for simple numbers
        value_str = re.sub(r'[,\s]', '', value_str)

        # Simple float conversion
        return float(value_str)


# Convenience functions
def verify_numerical_claim(claim_text: str, extracted_value: str) -> Tuple[bool, Optional[StatisticalDataPoint]]:
    """
    Convenience function to verify a numerical claim.

    Args:
        claim_text: Full claim text
        extracted_value: Extracted numerical value

    Returns:
        (is_verified, data_point)
    """
    manager = StatisticalAPIManager()
    return manager.verify_numerical_claim(claim_text, extracted_value)