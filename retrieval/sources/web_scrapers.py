"""
Web scraping functionality for gathering evidence from news and government sources.
Provides access to additional sources beyond fact-checkers and statistical APIs.
"""

import requests
import re
import time
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from urllib.parse import urljoin, urlparse
import json
import feedparser
from bs4 import BeautifulSoup

from .http_client import HttpClient, create_http_client


@dataclass
class ScrapedContent:
    """Content scraped from a web source."""
    title: str
    content: str
    url: str
    source_name: str
    publication_date: Optional[datetime] = None
    credibility_score: float = 50.0
    tags: List[str] = None
    relevance_score: float = 0.0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class NewsScraper:
    """
    Base class for news website scraping.
    Provides common functionality for different news sources.
    """

    def __init__(self, base_url: str, name: str, language: str = "es"):
        self.base_url = base_url
        self.name = name
        self.language = language
        self.http_client = create_http_client()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def search_articles(self, query: str, max_results: int = 5) -> List[ScrapedContent]:
        """
        Search for articles related to a query.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of scraped content
        """
        raise NotImplementedError("Subclasses must implement search_articles")

    def _extract_article_content(self, url: str) -> Optional[str]:
        """Extract main content from an article URL."""
        try:
            response = self.http_client.get(url, timeout=10)
            response.raise_for_status()

            # This is a simplified content extraction
            # In production, would use libraries like newspaper3k or trafilatura
            content = self._simple_content_extraction(response.text)
            return content

        except Exception as e:
            self.logger.error(f"Content extraction failed for {url}: {e}")
            return None

    def _simple_content_extraction(self, html: str) -> str:
        """Simple content extraction from HTML."""
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)

        # Extract text from paragraph tags
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL)
        content = ' '.join(paragraphs)

        # Clean up HTML entities and extra whitespace
        content = re.sub(r'&[^;]+;', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()

        return content[:2000]  # Limit content length


class RSSNewsScraper(NewsScraper):
    """News scraper that uses RSS feeds."""

    def __init__(self, base_url: str, name: str, rss_url: str, language: str = "es"):
        super().__init__(base_url, name, language)
        self.rss_url = rss_url

    def search_articles(self, query: str, max_results: int = 5) -> List[ScrapedContent]:
        """Search articles using RSS feed."""
        try:
            # Parse RSS feed with better error handling
            feed = feedparser.parse(self.rss_url, agent='Mozilla/5.0 (compatible; EvidenceVerifier/1.0)')
            
            # Check for parsing errors
            if feed.bozo:
                self.logger.warning(f"RSS parse error for {self.rss_url}: {feed.bozo_exception}")
                # Try to continue anyway if we have some entries
                if not feed.entries:
                    return []

            articles = []
            query_lower = query.lower()

            for entry in feed.entries[:30]:  # Check more entries to find relevant ones
                title = entry.get('title', '').strip()
                description = entry.get('description', '').strip()

                # Skip empty entries
                if not title and not description:
                    continue

                # Check if article is relevant to query (more flexible matching)
                combined_text = f"{title} {description}".lower()
                query_words = query_lower.split()
                
                # Check if any query word appears in the combined text
                is_relevant = any(word in combined_text for word in query_words)
                
                # Also check for partial matches and related terms
                if not is_relevant:
                    # For population claims, also match demographic terms
                    if any(word in query_lower for word in ['población', 'habitantes', 'personas', 'millones']):
                        is_relevant = any(word in combined_text for word in ['población', 'habitantes', 'personas', 'millones', 'demografía', 'censo'])
                    # For GDP claims, also match economic terms
                    elif any(word in query_lower for word in ['pib', 'gdp', 'crecimiento', 'economía']):
                        is_relevant = any(word in combined_text for word in ['pib', 'gdp', 'crecimiento', 'economía', 'bruto', 'producto'])
                    # For COVID claims, also match pandemic terms
                    elif any(word in query_lower for word in ['covid', 'pandemia', 'coronavirus']):
                        is_relevant = any(word in combined_text for word in ['covid', 'pandemia', 'coronavirus', 'vacuna', 'contagio'])

                if is_relevant:
                    # Extract content if possible
                    content = description
                    if hasattr(entry, 'link') and entry.link:
                        full_content = self._extract_article_content(entry.link)
                        if full_content and len(full_content) > len(description):
                            content = full_content

                    # Calculate relevance score based on how many query words match
                    relevance_score = sum(1 for word in query_words if word in combined_text) / len(query_words)

                    articles.append(ScrapedContent(
                        title=title,
                        content=content,
                        url=entry.get('link', ''),
                        source_name=self.name,
                        publication_date=self._parse_date(entry),
                        credibility_score=self._get_credibility_score(),
                        relevance_score=min(relevance_score, 1.0)
                    ))

                    if len(articles) >= max_results:
                        break

            return articles

        except Exception as e:
            self.logger.error(f"RSS scraping failed for {self.name}: {e}")
            return []

    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse publication date from RSS entry."""
        date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
        for field in date_fields:
            if hasattr(entry, field) and entry[field]:
                try:
                    return datetime(*entry[field][:6])
                except:
                    continue
        return None

    def _get_credibility_score(self) -> float:
        """Get credibility score for this source."""
        # Base scores for known sources
        credibility_map = {
            'El País': 85,
            'El Mundo': 75,
            'ABC': 80,
            'La Vanguardia': 78,
            'Google News': 70,
            'Bing News': 65
        }
        return credibility_map.get(self.name, 60)

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score for article to query."""
        query_words = set(query.split())
        text_words = set(text.split())

        # Simple word overlap score
        overlap = len(query_words.intersection(text_words))
        return min(overlap / len(query_words), 1.0) if query_words else 0.0


class ElPaisScraper(RSSNewsScraper):
    """Scraper for El País newspaper."""

    def __init__(self):
        super().__init__(
            "https://elpais.com",
            "El País",
            "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",
            "es"
        )


class ElMundoScraper(RSSNewsScraper):
    """Scraper for El Mundo newspaper."""

    def __init__(self):
        super().__init__(
            "https://www.elmundo.es",
            "El Mundo",
            "https://www.elmundo.es/rss/portada.xml",
            "es"
        )


class ABCScraper(RSSNewsScraper):
    """Scraper for ABC newspaper."""

    def __init__(self):
        super().__init__(
            "https://www.abc.es",
            "ABC",
            "https://www.abc.es/rss/feeds/abc_portada.xml",
            "es"
        )


class LaVanguardiaScraper(RSSNewsScraper):
    """Scraper for La Vanguardia newspaper."""

    def __init__(self):
        super().__init__(
            "https://www.lavanguardia.com",
            "La Vanguardia",
            "https://www.lavanguardia.com/rss/home.xml",
            "es"
        )


class GoogleNewsScraper(RSSNewsScraper):
    """Scraper for Google News RSS feeds."""

    def __init__(self, topic: str = "general"):
        # Use a more reliable Google News RSS URL
        rss_url = "https://news.google.com/rss?hl=es&gl=ES&ceid=ES:es"
        if topic != "general":
            rss_url += f"&topic={topic}"

        super().__init__(
            "https://news.google.com",
            "Google News",
            rss_url,
            "es"
        )


class WikipediaScraper(NewsScraper):
    """Scraper for Wikipedia articles."""

    def __init__(self, language: str = "es"):
        base_url = f"https://{language}.wikipedia.org"
        super().__init__(base_url, f"Wikipedia ({language})", language)

    def search_articles(self, query: str, max_results: int = 5) -> List[ScrapedContent]:
        """Search Wikipedia articles."""
        try:
            # Use Wikipedia's search API
            search_url = f"{self.base_url}/w/api.php"
            params = {
                'action': 'query',
                'list': 'search',
                'srsearch': query,
                'format': 'json',
                'srlimit': max_results * 2,  # Get more results to filter
                'srprop': 'title|snippet|timestamp'
            }

            response = self.http_client.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for result in data.get('query', {}).get('search', [])[:max_results]:
                title = result['title']
                snippet = result.get('snippet', '')

                # Create Wikipedia URL
                title_encoded = title.replace(' ', '_')
                url = f"{self.base_url}/wiki/{title_encoded}"

                # Extract content from the article
                full_content = self._extract_article_content(url)
                content = full_content if full_content else snippet

                articles.append(ScrapedContent(
                    title=title,
                    content=content,
                    url=url,
                    source_name=self.name,
                    publication_date=self._parse_wiki_date(result),
                    credibility_score=95,  # Wikipedia is highly credible for facts
                    relevance_score=self._calculate_relevance(query, f"{title} {snippet}")
                ))

            return articles

        except Exception as e:
            self.logger.error(f"Wikipedia search failed: {e}")
            return []

    def _parse_wiki_date(self, result: Dict[str, Any]) -> Optional[datetime]:
        """Parse publication date from Wikipedia search result."""
        timestamp = result.get('timestamp')
        if timestamp:
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                pass
        return None

    def _extract_article_content(self, url: str) -> Optional[str]:
        """Extract main content from Wikipedia article."""
        try:
            response = self.http_client.get(url, timeout=10)
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup.select('script, style, .navbox, .vertical-navbox, .infobox, .metadata'):
                element.decompose()

            # Extract content from main article body
            content_div = soup.select_one('#mw-content-text')
            if content_div:
                # Get text from paragraphs
                paragraphs = content_div.select('p')
                content = ' '.join([p.get_text() for p in paragraphs[:10]])  # First 10 paragraphs
                return content[:2000] if content else None

            return None

        except Exception as e:
            self.logger.error(f"Wikipedia content extraction failed for {url}: {e}")
            return None

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score for article to query."""
        query_words = set(query.split())
        text_words = set(text.split())

        # Simple word overlap score
        overlap = len(query_words.intersection(text_words))
        return min(overlap / len(query_words), 1.0) if query_words else 0.0


class WebScraperManager:
    """
    Manages multiple web scrapers and provides unified search interface.
    """

    def __init__(self):
        self.scrapers = {
            'elpais': ElPaisScraper(),
            'elmundo': ElMundoScraper(),
            'abc': ABCScraper(),
            'lavanguardia': LaVanguardiaScraper(),
            'googlenews': GoogleNewsScraper(),
            'wikipedia': WikipediaScraper(),
        }

        self.logger = logging.getLogger(__name__)

    def search_all_sources(self, query: str, max_per_source: int = 3) -> List[ScrapedContent]:
        """
        Search across all web sources.

        Args:
            query: Search query
            max_per_source: Maximum results per source

        Returns:
            Combined list of scraped content
        """
        all_results = []

        for scraper_name, scraper in self.scrapers.items():
            try:
                results = scraper.search_articles(query, max_per_source)
                all_results.extend(results)
                self.logger.debug(f"{scraper_name}: {len(results)} results")
            except Exception as e:
                self.logger.error(f"Error scraping {scraper_name}: {e}")

        # Sort by relevance and credibility
        all_results.sort(key=lambda x: (x.relevance_score, x.credibility_score), reverse=True)

        return all_results

    def search_specific_sources(self, query: str, sources: List[str], max_per_source: int = 3) -> List[ScrapedContent]:
        """
        Search specific sources only.

        Args:
            query: Search query
            sources: List of source names to search
            max_per_source: Maximum results per source

        Returns:
            Combined list of scraped content
        """
        all_results = []

        for source_name in sources:
            if source_name in self.scrapers:
                try:
                    results = self.scrapers[source_name].search_articles(query, max_per_source)
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Error scraping {source_name}: {e}")

        return all_results

    def get_source_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available web scraping sources."""
        return {
            'elpais': {
                'name': 'El País',
                'country': 'Spain',
                'language': 'es',
                'type': 'newspaper',
                'credibility_score': 85,
                'update_frequency': 'hourly'
            },
            'elmundo': {
                'name': 'El Mundo',
                'country': 'Spain',
                'language': 'es',
                'type': 'newspaper',
                'credibility_score': 75,
                'update_frequency': 'hourly'
            },
            'abc': {
                'name': 'ABC',
                'country': 'Spain',
                'language': 'es',
                'type': 'newspaper',
                'credibility_score': 80,
                'update_frequency': 'hourly'
            },
            'lavanguardia': {
                'name': 'La Vanguardia',
                'country': 'Spain',
                'language': 'es',
                'type': 'newspaper',
                'credibility_score': 78,
                'update_frequency': 'hourly'
            },
            'googlenews': {
                'name': 'Google News',
                'country': 'Global',
                'language': 'es',
                'type': 'aggregator',
                'credibility_score': 70,
                'update_frequency': 'real-time'
            },
            'wikipedia': {
                'name': 'Wikipedia',
                'country': 'Global',
                'language': 'es',
                'type': 'encyclopedia',
                'credibility_score': 95,
                'update_frequency': 'continuous'
            }
        }


# Convenience functions
def search_web_sources(query: str, sources: List[str] = None, max_per_source: int = 3) -> List[ScrapedContent]:
    """
    Convenience function to search web sources.

    Args:
        query: Search query
        sources: List of source names to search (optional)
        max_per_source: Maximum results per source

    Returns:
        List of scraped content
    """
    manager = WebScraperManager()

    if sources:
        return manager.search_specific_sources(query, sources, max_per_source)
    else:
        return manager.search_all_sources(query, max_per_source)


def scrape_article_content(url: str) -> Optional[str]:
    """
    Convenience function to scrape content from a single article URL.

    Args:
        url: Article URL to scrape

    Returns:
        Extracted content or None
    """
    # Determine which scraper to use based on domain
    domain = urlparse(url).netloc.lower()

    scrapers = {
        'elpais.com': ElPaisScraper(),
        'elmundo.es': ElMundoScraper(),
        'abc.es': ABCScraper(),
        'lavanguardia.com': LaVanguardiaScraper(),
        'wikipedia.org': WikipediaScraper(),
    }

    for domain_pattern, scraper in scrapers.items():
        if domain_pattern in domain:
            return scraper._extract_article_content(url)

    # Fallback to generic extraction
    http_client = create_http_client()
    try:
        response = http_client.get(url, timeout=10)
        response.raise_for_status()

        # Simple content extraction
        content = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', response.text, flags=re.DOTALL)
        paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL)
        text = ' '.join(paragraphs)
        text = re.sub(r'&[^;]+;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text[:2000] if text else None

    except Exception as e:
        logging.error(f"Article scraping failed for {url}: {e}")
        return None