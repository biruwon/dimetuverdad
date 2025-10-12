"""
Fact-checking site integration for verifying claims against trusted sources.
Provides access to Spanish and international fact-checking organizations.
"""

import requests
import re
import time
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json
from bs4 import BeautifulSoup

from .http_client import HttpClient, create_http_client


@dataclass
class FactCheckResult:
    """Result from a fact-checking site."""
    claim: str
    verdict: str  # 'true', 'false', 'misleading', 'unverified'
    confidence: float  # 0.0 to 1.0
    explanation: str
    source_name: str
    source_url: str
    publication_date: Optional[datetime] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class FactCheckerClient:
    """
    Base client for fact-checking site integration.
    Provides common functionality for different fact-checking sources.
    """

    def __init__(self, base_url: str, name: str, language: str = "es"):
        self.base_url = base_url
        self.name = name
        self.language = language
        self.http_client = create_http_client()

    def search_claims(self, query: str, max_results: int = 5) -> List[FactCheckResult]:
        """
        Search for fact-checks related to a query.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of fact-check results
        """
        raise NotImplementedError("Subclasses must implement search_claims")

    def _normalize_verdict(self, verdict: str) -> Tuple[str, float]:
        """
        Normalize verdict text to standard format.

        Returns:
            (normalized_verdict, confidence)
        """
        verdict_lower = verdict.lower().strip()

        # Spanish verdicts
        if any(word in verdict_lower for word in ['verdadero', 'cierto', 'confirmado', 'true']):
            return 'true', 0.9
        elif any(word in verdict_lower for word in ['falso', 'mentira', 'desmentido', 'false']):
            return 'false', 0.9
        elif any(word in verdict_lower for word in ['engañoso', 'manipulado', 'misleading', 'manipulado']):
            return 'misleading', 0.7
        elif any(word in verdict_lower for word in ['parcialmente', ' parcialmente', 'mixta', 'mixta']):
            return 'misleading', 0.6
        else:
            return 'unverified', 0.3


class MalditaClient(FactCheckerClient):
    """Client for Maldita.es fact-checking site."""

    def __init__(self):
        super().__init__("https://maldita.es", "Maldita.es", "es")

    def search_claims(self, query: str, max_results: int = 5) -> List[FactCheckResult]:
        """Search Maldita.es for fact-checks."""
        try:
            # Use Maldita's WordPress search endpoint
            search_url = f"{self.base_url}/"
            params = {'s': query}

            response = self.http_client.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            # Parse search results (simplified - would need actual HTML parsing)
            results = self._parse_maldita_results(response.text, query, max_results)
            return results

        except Exception as e:
            self.logger.error(f"Maldita.es search failed: {e}")
            return []

    def _parse_maldita_results(self, html: str, query: str, max_results: int) -> List[FactCheckResult]:
        """Parse Maldita.es search results."""
        results = []
        soup = BeautifulSoup(html, 'html.parser')

        # Find article containers - Maldita.es uses various layouts
        article_selectors = [
            'article',
            '.post',
            '.entry',
            '.article-item',
            '.search-result'
        ]

        articles = []
        for selector in article_selectors:
            articles = soup.select(selector)
            if articles:
                break

        for article in articles[:max_results]:
            try:
                # Extract title
                title_elem = article.select_one('h2, h3, .entry-title, .post-title')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                # Extract URL
                link_elem = article.select_one('a[href]')
                if not link_elem:
                    continue
                url = link_elem.get('href')
                if not url.startswith('http'):
                    url = f"{self.base_url}{url}"

                # Extract excerpt/description
                excerpt_elem = article.select_one('.entry-excerpt, .post-excerpt, .summary, p')
                excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                # Extract date
                date_elem = article.select_one('time, .entry-date, .post-date')
                pub_date = None
                if date_elem:
                    date_str = date_elem.get('datetime') or date_elem.get_text(strip=True)
                    try:
                        # Try parsing various date formats
                        pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        try:
                            pub_date = datetime.strptime(date_str, '%Y-%m-%d')
                        except:
                            pub_date = None

                # Determine verdict based on content analysis
                verdict, confidence = self._analyze_maldita_content(title, excerpt)

                if verdict != 'unverified':  # Only include if we found a verdict
                    results.append(FactCheckResult(
                        claim=title,
                        verdict=verdict,
                        confidence=confidence,
                        explanation=excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
                        source_name="Maldita.es",
                        source_url=url,
                        publication_date=pub_date,
                        tags=self._extract_tags(title, excerpt)
                    ))

            except Exception as e:
                self.logger.warning(f"Error parsing Maldita.es article: {e}")
                continue

        return results[:max_results]

    def _analyze_maldita_content(self, title: str, excerpt: str) -> Tuple[str, float]:
        """Analyze content to determine verdict and confidence."""
        content = (title + " " + excerpt).lower()

        # Look for verdict indicators in Spanish
        if any(word in content for word in ['verdadero', 'cierto', 'confirmado', 'correcto']):
            return 'true', 0.85
        elif any(word in content for word in ['falso', 'mentira', 'desmentido', 'incorrecto']):
            return 'false', 0.85
        elif any(word in content for word in ['engañoso', 'manipulado', 'parcialmente', 'mixta']):
            return 'misleading', 0.7
        else:
            return 'unverified', 0.3

    def _extract_tags(self, title: str, excerpt: str) -> List[str]:
        """Extract relevant tags from content."""
        content = (title + " " + excerpt).lower()
        tags = []

        # Economic and financial topics
        if any(word in content for word in ['economía', 'pib', 'gdp', 'crecimiento', 'inflación', 'paro', 'desempleo', 'salarios', 'impuestos', 'déficit', 'deuda', 'presupuesto', 'banco', 'mercado', 'bolsa', 'inversión', 'exportaciones', 'importaciones']):
            tags.extend(['economía', 'finanzas', 'pib', 'crecimiento'])

        # Demographic and population topics
        if any(word in content for word in ['población', 'habitantes', 'personas', 'millones', 'demografía', 'migración', 'natalidad', 'mortalidad', 'esperanza de vida', 'envejecimiento']):
            tags.extend(['población', 'demografía', 'estadísticas'])

        # Health and pandemic topics
        if any(word in content for word in ['pandemia', 'covid', 'coronavirus', 'vacuna', 'salud', 'enfermedad', 'hospital', 'médico', 'farmacia', 'contagio', 'muerte', 'casos', 'pruebas', 'uci']):
            tags.extend(['salud', 'pandemia', 'covid', 'medicina'])

        # Political topics
        if any(word in content for word in ['política', 'gobierno', 'elecciones', 'partido', 'votación', 'parlamento', 'congreso', 'senado', 'presidente', 'ministro', 'alcalde', 'diputado', 'ley', 'reforma', 'constitución']):
            tags.append('política')

        # Social and media topics
        if any(word in content for word in ['redes sociales', 'twitter', 'facebook', 'instagram', 'tiktok', 'internet', 'desinformación', 'fake news', 'mentira', 'engaño', 'manipulación']):
            tags.extend(['redes sociales', 'internet', 'desinformación'])

        # Education topics
        if any(word in content for word in ['educación', 'escuela', 'universidad', 'estudiante', 'profesor', 'examen', 'beca', 'formación']):
            tags.append('educación')

        # Environment and climate topics
        if any(word in content for word in ['medio ambiente', 'clima', 'cambio climático', 'contaminación', 'energía', 'renovable', 'emisiones', 'calentamiento']):
            tags.extend(['medio ambiente', 'clima'])

        # Crime and justice topics
        if any(word in content for word in ['delito', 'crimen', 'policía', 'justicia', 'prisión', 'juez', 'tribunal', 'ley', 'seguridad']):
            tags.extend(['justicia', 'seguridad'])

        # International relations
        if any(word in content for word in ['internacional', 'ue', 'unión europea', 'eeuu', 'estados unidos', 'china', 'rusia', 'guerra', 'conflicto', 'paz', 'diplomacia']):
            tags.append('internacional')

        # Science and technology
        if any(word in content for word in ['ciencia', 'tecnología', 'investigación', 'innovación', 'ia', 'inteligencia artificial', 'robot', 'internet', 'digital']):
            tags.extend(['ciencia', 'tecnología'])

        # Sports and entertainment
        if any(word in content for word in ['deporte', 'fútbol', 'baloncesto', 'tenis', 'olimpiada', 'cine', 'música', 'televisión', 'entretenimiento']):
            tags.extend(['deporte', 'entretenimiento'])

        # Return unique tags
        return list(set(tags))


class NewtralClient(FactCheckerClient):
    """Client for Newtral.es fact-checking site."""

    def __init__(self):
        super().__init__("https://www.newtral.es", "Newtral", "es")

    def search_claims(self, query: str, max_results: int = 5) -> List[FactCheckResult]:
        """Search Newtral for fact-checks."""
        try:
            search_url = f"{self.base_url}/"
            params = {'s': query}

            response = self.http_client.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            results = self._parse_newtral_results(response.text, query, max_results)
            return results

        except Exception as e:
            self.logger.error(f"Newtral search failed: {e}")
            return []

    def _parse_newtral_results(self, html: str, query: str, max_results: int) -> List[FactCheckResult]:
        """Parse Newtral search results."""
        results = []
        soup = BeautifulSoup(html, 'html.parser')

        # Newtral.es search results structure
        article_selectors = [
            'article',
            '.post',
            '.entry',
            '.article-item',
            '.search-result-item'
        ]

        articles = []
        for selector in article_selectors:
            articles = soup.select(selector)
            if articles:
                break

        for article in articles[:max_results]:
            try:
                # Extract title
                title_elem = article.select_one('h2, h3, .entry-title, .post-title, .article-title')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                # Extract URL
                link_elem = article.select_one('a[href]') or title_elem.select_one('a[href]')
                if not link_elem:
                    continue
                url = link_elem.get('href')
                if not url.startswith('http'):
                    url = f"{self.base_url}{url}"

                # Extract excerpt/description
                excerpt_elem = article.select_one('.entry-excerpt, .post-excerpt, .summary, .article-excerpt, p')
                excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                # Extract date
                date_elem = article.select_one('time, .entry-date, .post-date, .article-date')
                pub_date = None
                if date_elem:
                    date_str = date_elem.get('datetime') or date_elem.get_text(strip=True)
                    try:
                        pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        try:
                            pub_date = datetime.strptime(date_str, '%d/%m/%Y')
                        except:
                            pub_date = None

                # Determine verdict based on content analysis
                verdict, confidence = self._analyze_newtral_content(title, excerpt)

                if verdict != 'unverified':  # Only include if we found a verdict
                    results.append(FactCheckResult(
                        claim=title,
                        verdict=verdict,
                        confidence=confidence,
                        explanation=excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
                        source_name="Newtral",
                        source_url=url,
                        publication_date=pub_date,
                        tags=self._extract_newtral_tags(title, excerpt)
                    ))

            except Exception as e:
                self.logger.warning(f"Error parsing Newtral article: {e}")
                continue

        return results[:max_results]

    def _analyze_newtral_content(self, title: str, excerpt: str) -> Tuple[str, float]:
        """Analyze content to determine verdict and confidence."""
        content = (title + " " + excerpt).lower()

        # Look for verdict indicators in Spanish
        if any(word in content for word in ['verdadero', 'cierto', 'confirmado', 'correcto', 'verdadera']):
            return 'true', 0.88
        elif any(word in content for word in ['falso', 'mentira', 'desmentido', 'incorrecto', 'falsa']):
            return 'false', 0.88
        elif any(word in content for word in ['engañoso', 'manipulado', 'parcialmente', 'mixta', 'engañoso']):
            return 'misleading', 0.75
        else:
            return 'unverified', 0.3

    def _extract_newtral_tags(self, title: str, excerpt: str) -> List[str]:
        """Extract relevant tags from content."""
        content = (title + " " + excerpt).lower()
        tags = []

        # Economic and financial topics
        if any(word in content for word in ['economía', 'pib', 'gdp', 'crecimiento', 'inflación', 'paro', 'desempleo', 'salarios', 'impuestos', 'déficit', 'deuda', 'presupuesto', 'banco', 'mercado', 'bolsa', 'inversión', 'exportaciones', 'importaciones']):
            tags.extend(['economía', 'finanzas', 'pib', 'crecimiento'])

        # Demographic and population topics
        if any(word in content for word in ['población', 'habitantes', 'personas', 'millones', 'demografía', 'migración', 'natalidad', 'mortalidad', 'esperanza de vida', 'envejecimiento']):
            tags.extend(['población', 'demografía', 'estadísticas'])

        # Health and pandemic topics
        if any(word in content for word in ['pandemia', 'covid', 'coronavirus', 'vacuna', 'salud', 'enfermedad', 'hospital', 'médico', 'farmacia', 'contagio', 'muerte', 'casos', 'pruebas', 'uci']):
            tags.extend(['salud', 'pandemia', 'covid', 'medicina'])

        # Political topics
        if any(word in content for word in ['política', 'gobierno', 'elecciones', 'partido', 'votación', 'parlamento', 'congreso', 'senado', 'presidente', 'ministro', 'alcalde', 'diputado', 'ley', 'reforma', 'constitución']):
            tags.append('política')

        # Social and media topics
        if any(word in content for word in ['redes sociales', 'twitter', 'facebook', 'instagram', 'tiktok', 'internet', 'desinformación', 'fake news', 'mentira', 'engaño', 'manipulación']):
            tags.extend(['redes sociales', 'internet', 'desinformación'])

        # Education topics
        if any(word in content for word in ['educación', 'escuela', 'universidad', 'estudiante', 'profesor', 'examen', 'beca', 'formación']):
            tags.append('educación')

        # Environment and climate topics
        if any(word in content for word in ['medio ambiente', 'clima', 'cambio climático', 'contaminación', 'energía', 'renovable', 'emisiones', 'calentamiento']):
            tags.extend(['medio ambiente', 'clima'])

        # Crime and justice topics
        if any(word in content for word in ['delito', 'crimen', 'policía', 'justicia', 'prisión', 'juez', 'tribunal', 'ley', 'seguridad']):
            tags.extend(['justicia', 'seguridad'])

        # International relations
        if any(word in content for word in ['internacional', 'ue', 'unión europea', 'eeuu', 'estados unidos', 'china', 'rusia', 'guerra', 'conflicto', 'paz', 'diplomacia']):
            tags.append('internacional')

        # Science and technology
        if any(word in content for word in ['ciencia', 'tecnología', 'investigación', 'innovación', 'ia', 'inteligencia artificial', 'robot', 'internet', 'digital']):
            tags.extend(['ciencia', 'tecnología'])

        # Sports and entertainment
        if any(word in content for word in ['deporte', 'fútbol', 'baloncesto', 'tenis', 'olimpiada', 'cine', 'música', 'televisión', 'entretenimiento']):
            tags.extend(['deporte', 'entretenimiento'])

        # Return unique tags
        return list(set(tags))


class FactCheckManager:
    """
    Manages multiple fact-checking sources and provides unified search interface.
    """

    def __init__(self):
        self.clients = {
            'maldita': MalditaClient(),
            'newtral': NewtralClient(),
        }

        # Add international fact-checkers
        self.clients.update({
            'snopes': SnopesClient(),
            'politifact': PolitiFactClient(),
            'factcheck': FactCheckOrgClient(),
        })

        self.logger = logging.getLogger(__name__)

    def search_all_sources(self, query: str, max_per_source: int = 3) -> List[FactCheckResult]:
        """
        Search across all fact-checking sources.

        Args:
            query: Search query
            max_per_source: Maximum results per source

        Returns:
            Combined list of fact-check results
        """
        all_results = []

        for client_name, client in self.clients.items():
            try:
                results = client.search_claims(query, max_per_source)
                all_results.extend(results)
                self.logger.debug(f"{client_name}: {len(results)} results")
            except Exception as e:
                self.logger.error(f"Error searching {client_name}: {e}")

        # Sort by confidence and recency
        all_results.sort(key=lambda x: (x.confidence, x.publication_date or datetime.min), reverse=True)

        return all_results

    def verify_claim(self, claim: str, language: str = "es") -> Optional[FactCheckResult]:
        """
        Find the most relevant fact-check for a specific claim.

        Args:
            claim: The claim to verify
            language: Language preference

        Returns:
            Best matching fact-check result, or None
        """
        results = self.search_all_sources(claim, max_per_source=2)

        if not results:
            return None

        # Return the highest confidence result
        return max(results, key=lambda x: x.confidence)

    def get_source_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available fact-checking sources."""
        return {
            'maldita': {
                'name': 'Maldita.es',
                'country': 'Spain',
                'language': 'es',
                'credibility_score': 95,
                'specialties': ['politics', 'health', 'social media']
            },
            'newtral': {
                'name': 'Newtral',
                'country': 'Spain',
                'language': 'es',
                'credibility_score': 92,
                'specialties': ['politics', 'economy', 'international']
            },
            'snopes': {
                'name': 'Snopes',
                'country': 'USA',
                'language': 'en',
                'credibility_score': 90,
                'specialties': ['urban legends', 'internet rumors']
            },
            'politifact': {
                'name': 'PolitiFact',
                'country': 'USA',
                'language': 'en',
                'credibility_score': 93,
                'specialties': ['politics', 'elections', 'government']
            },
            'factcheck': {
                'name': 'FactCheck.org',
                'country': 'USA',
                'language': 'en',
                'credibility_score': 94,
                'specialties': ['politics', 'advertising', 'media']
            }
        }


# International fact-checkers (simplified implementations)

class SnopesClient(FactCheckerClient):
    """Client for Snopes.com."""

    def __init__(self):
        super().__init__("https://www.snopes.com", "Snopes", "en")

    def search_claims(self, query: str, max_results: int = 5) -> List[FactCheckResult]:
        """Search Snopes for fact-checks."""
        try:
            search_url = f"{self.base_url}/"
            params = {'s': query}

            response = self.http_client.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            results = self._parse_snopes_results(response.text, query, max_results)
            return results

        except Exception as e:
            self.logger.error(f"Snopes search failed: {e}")
            return []

    def _parse_snopes_results(self, html: str, query: str, max_results: int) -> List[FactCheckResult]:
        """Parse Snopes search results."""
        results = []
        soup = BeautifulSoup(html, 'html.parser')

        # Snopes search results structure
        article_selectors = [
            'article',
            '.article',
            '.search-result',
            '.media'
        ]

        articles = []
        for selector in article_selectors:
            articles = soup.select(selector)
            if articles:
                break

        for article in articles[:max_results]:
            try:
                # Extract title
                title_elem = article.select_one('h2, h3, .article-title, .entry-title')
                if not title_elem:
                    continue
                title = title_elem.get_text(strip=True)

                # Extract URL
                link_elem = article.select_one('a[href]') or title_elem.select_one('a[href]')
                if not link_elem:
                    continue
                url = link_elem.get('href')
                if not url.startswith('http'):
                    url = f"{self.base_url}{url}"

                # Extract excerpt/description
                excerpt_elem = article.select_one('.article-description, .entry-excerpt, .summary, p')
                excerpt = excerpt_elem.get_text(strip=True) if excerpt_elem else ""

                # Extract date
                date_elem = article.select_one('time, .article-date, .entry-date')
                pub_date = None
                if date_elem:
                    date_str = date_elem.get('datetime') or date_elem.get_text(strip=True)
                    try:
                        pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        try:
                            pub_date = datetime.strptime(date_str, '%B %d, %Y')
                        except:
                            pub_date = None

                # Extract rating/verdict from Snopes-specific elements
                rating_elem = article.select_one('.rating, .fact-check-rating, .verdict')
                verdict = 'unverified'
                confidence = 0.5

                if rating_elem:
                    rating_text = rating_elem.get_text(strip=True).lower()
                    if 'true' in rating_text:
                        verdict = 'true'
                        confidence = 0.9
                    elif 'false' in rating_text:
                        verdict = 'false'
                        confidence = 0.9
                    elif 'misleading' in rating_text or 'mixture' in rating_text:
                        verdict = 'misleading'
                        confidence = 0.7

                # Fallback: analyze content for verdict indicators
                if verdict == 'unverified':
                    verdict, confidence = self._analyze_snopes_content(title, excerpt)

                if verdict != 'unverified':  # Only include if we found a verdict
                    results.append(FactCheckResult(
                        claim=title,
                        verdict=verdict,
                        confidence=confidence,
                        explanation=excerpt[:200] + "..." if len(excerpt) > 200 else excerpt,
                        source_name="Snopes",
                        source_url=url,
                        publication_date=pub_date,
                        tags=self._extract_snopes_tags(title, excerpt)
                    ))

            except Exception as e:
                self.logger.warning(f"Error parsing Snopes article: {e}")
                continue

        return results[:max_results]

    def _analyze_snopes_content(self, title: str, excerpt: str) -> Tuple[str, float]:
        """Analyze content to determine verdict and confidence."""
        content = (title + " " + excerpt).lower()

        # Look for verdict indicators in English
        if any(word in content for word in ['true', 'confirmed', 'accurate', 'correct']):
            return 'true', 0.85
        elif any(word in content for word in ['false', 'lie', 'debuted', 'incorrect']):
            return 'false', 0.85
        elif any(word in content for word in ['misleading', 'mixture', 'partially', 'mixed']):
            return 'misleading', 0.7
        else:
            return 'unverified', 0.3

    def _extract_snopes_tags(self, title: str, excerpt: str) -> List[str]:
        """Extract relevant tags from content."""
        content = (title + " " + excerpt).lower()
        tags = []

        if 'pandemia' in content or 'covid' in content or 'coronavirus' in content or 'vaccine' in content:
            tags.extend(['pandemia', 'covid', 'salud'])
        if 'economy' in content or 'gdp' in content or 'economic' in content:
            tags.extend(['economía', 'pib', 'crecimiento'])
        if 'population' in content or 'million' in content or 'demographics' in content:
            tags.extend(['población', 'demografía', 'estadísticas'])
        if 'politics' in content or 'government' in content or 'political' in content:
            tags.append('política')
        if 'social media' in content or 'facebook' in content or 'twitter' in content:
            tags.extend(['redes sociales', 'internet'])

        # Return unique tags
        return list(set(tags))


class PolitiFactClient(FactCheckerClient):
    """Client for PolitiFact.com."""

    def __init__(self):
        super().__init__("https://www.politifact.com", "PolitiFact", "en")

    def search_claims(self, query: str, max_results: int = 5) -> List[FactCheckResult]:
        """Search PolitiFact for fact-checks."""
        # Simplified implementation
        return []


class FactCheckOrgClient(FactCheckerClient):
    """Client for FactCheck.org."""

    def __init__(self):
        super().__init__("https://www.factcheck.org", "FactCheck.org", "en")

    def search_claims(self, query: str, max_results: int = 5) -> List[FactCheckResult]:
        """Search FactCheck.org for fact-checks."""
        # Simplified implementation
        return []


# Convenience functions
def search_fact_checks(query: str, sources: List[str] = None) -> List[FactCheckResult]:
    """
    Convenience function to search fact-checking sites.

    Args:
        query: Search query
        sources: List of source names to search (optional)

    Returns:
        List of fact-check results
    """
    manager = FactCheckManager()

    if sources:
        # Filter to specific sources
        results = []
        for source in sources:
            if source in manager.clients:
                try:
                    source_results = manager.clients[source].search_claims(query, max_results=3)
                    results.extend(source_results)
                except Exception as e:
                    logging.error(f"Error searching {source}: {e}")
        return results
    else:
        return manager.search_all_sources(query)


def verify_claim_fact_check(claim: str, language: str = "es") -> Optional[FactCheckResult]:
    """
    Convenience function to verify a claim using fact-checkers.

    Args:
        claim: Claim to verify
        language: Language preference

    Returns:
        Fact-check result or None
    """
    manager = FactCheckManager()
    return manager.verify_claim(claim, language)