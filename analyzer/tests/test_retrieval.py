"""
Tests for analyzer/retrieval.py
Comprehensive test coverage for evidence retrieval functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from analyzer.retrieval import (
    extract_query_terms,
    build_search_urls,
    fetch_links_from_search_url,
    retrieve_evidence_for_post,
    format_evidence,
    SPANISH_STOPWORDS,
    SOURCE_TEMPLATES,
    FACT_CHECKERS,
    DEBUNK_KEYWORDS
)


class TestExtractQueryTerms:
    """Test extract_query_terms function."""

    def test_extract_query_terms_basic(self):
        """Test basic query term extraction."""
        text = "Este es un texto de prueba con muchas palabras importantes"
        result = extract_query_terms(text, max_terms=3)
        assert isinstance(result, str)
        assert len(result.split()) <= 3

    def test_extract_query_terms_with_stopwords(self):
        """Test query term extraction with Spanish stopwords."""
        text = "que de la el y a en un ser se no haber por con"
        result = extract_query_terms(text)
        # Should return original text since all words are stopwords
        assert "que de la" in result

    def test_extract_query_terms_remove_urls(self):
        """Test that URLs are removed from query terms."""
        text = "Check this link https://example.com and this word important"
        result = extract_query_terms(text)
        assert "https" not in result
        assert "important" in result

    def test_extract_query_terms_frequency_ordering(self):
        """Test that terms are ordered by frequency."""
        text = "cuatro cinco seis cuatro cinco cuatro"  # cuatro appears 3 times, cinco 2 times, seis 1 time (all >=4 chars)
        result = extract_query_terms(text, max_terms=3)
        terms = result.split()
        assert terms[0] == "cuatro"
        assert terms[1] == "cinco"
        assert terms[2] == "seis"

    def test_extract_query_terms_empty_text(self):
        """Test query term extraction with empty text."""
        result = extract_query_terms("")
        assert result == ""

    def test_extract_query_terms_short_words_filtered(self):
        """Test that short words (less than 4 chars) are filtered."""
        text = "cuatro cinco tres si no"  # tres is 4 chars, si/no are short and stopwords
        result = extract_query_terms(text)
        terms = result.split()
        assert "cuatro" in terms
        assert "cinco" in terms
        assert "tres" in terms  # exactly 4 chars, should be kept
        assert "si" not in terms  # too short
        assert "no" not in terms  # stopword


class TestBuildSearchUrls:
    """Test build_search_urls function."""

    def test_build_search_urls_default_sources(self):
        """Test building search URLs with default sources."""
        query = "test query"
        result = build_search_urls(query)
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "source" in item
            assert "label" in item
            assert "url" in item
            assert query.replace(" ", "+") in item["url"]

    def test_build_search_urls_specific_sources(self):
        """Test building search URLs with specific sources."""
        query = "test query"
        sources = ["maldita", "newtral"]
        result = build_search_urls(query, sources)
        assert len(result) == 2
        source_names = [item["source"] for item in result]
        assert "maldita" in source_names
        assert "newtral" in source_names

    def test_build_search_urls_invalid_source(self):
        """Test building search URLs with invalid source."""
        query = "test query"
        sources = ["invalid_source"]
        result = build_search_urls(query, sources)
        assert len(result) == 0


class TestFetchLinksFromSearchUrl:
    """Test fetch_links_from_search_url function."""

    @patch('analyzer.retrieval.requests.get')
    def test_fetch_links_rss_success(self, mock_get):
        """Test fetching links from RSS feed."""
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/rss+xml'}
        mock_response.text = '''<?xml version="1.0"?>
        <rss>
            <channel>
                <item>
                    <title>Test Article 1</title>
                    <link>https://example.com/1</link>
                </item>
                <item>
                    <title>Test Article 2</title>
                    <link>https://example.com/2</link>
                </item>
            </channel>
        </rss>'''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_links_from_search_url("https://example.com/rss", max_links=2)
        assert len(result) == 2
        assert result[0]["title"] == "Test Article 1"
        assert result[0]["url"] == "https://example.com/1"

    @patch('analyzer.retrieval.requests.get')
    def test_fetch_links_html_success(self, mock_get):
        """Test fetching links from HTML page."""
        mock_response = Mock()
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.text = '''
        <html>
            <body>
                <a href="https://example.com/1">Article 1</a>
                <a href="https://example.com/2">Article 2</a>
                <a href="#anchor">Skip this</a>
                <a href="javascript:void(0)">Skip this too</a>
            </body>
        </html>'''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_links_from_search_url("https://example.com/search", max_links=2)
        assert len(result) == 2
        assert result[0]["title"] == "Article 1"
        assert result[0]["url"] == "https://example.com/1"

    @patch('analyzer.retrieval.requests.get')
    def test_fetch_links_with_debunk_check(self, mock_get):
        """Test fetching links with debunk keyword detection."""
        # Mock search page response
        mock_search_response = Mock()
        mock_search_response.headers = {'content-type': 'text/html'}
        mock_search_response.text = '''
        <html>
            <body>
                <a href="https://example.com/article1">Article 1</a>
            </body>
        </html>'''
        mock_search_response.raise_for_status.return_value = None

        # Mock article page response with debunk keywords
        mock_article_response = Mock()
        mock_article_response.text = '''
        <html>
            <body>
                Este artículo contiene información falsa y ha sido desmentido.
                La afirmación no es cierta y esto es una corrección.
            </body>
        </html>'''
        mock_article_response.raise_for_status.return_value = None

        mock_get.side_effect = [mock_search_response, mock_article_response]

        result = fetch_links_from_search_url("https://example.com/search", max_links=1)
        assert len(result) == 1
        assert result[0]["verdict"] == "debunk"

    @patch('analyzer.retrieval.requests.get')
    def test_fetch_links_request_failure(self, mock_get):
        """Test handling of request failures."""
        mock_get.side_effect = Exception("Network error")
        result = fetch_links_from_search_url("https://example.com/search")
        assert result == []

    @patch('analyzer.retrieval.requests.get')
    def test_fetch_links_timeout(self, mock_get):
        """Test handling of timeouts."""
        mock_get.side_effect = TimeoutError("Request timed out")
        result = fetch_links_from_search_url("https://example.com/search")
        assert result == []


class TestRetrieveEvidenceForPost:
    """Test retrieve_evidence_for_post function."""

    @patch('analyzer.retrieval.fetch_links_from_search_url')
    @patch('analyzer.retrieval.time.sleep')
    def test_retrieve_evidence_success(self, mock_sleep, mock_fetch):
        """Test successful evidence retrieval."""
        mock_fetch.return_value = [
            {"title": "Test Article", "url": "https://example.com/article", "verdict": "debunk", "snippet": "test"}
        ]

        text = "This is a test post with important information"
        result = retrieve_evidence_for_post(text, max_per_source=1)

        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert "source" in item
            assert "label" in item
            assert "search_url" in item
            assert "results" in item

    @patch('analyzer.retrieval.fetch_links_from_search_url')
    @patch('analyzer.retrieval.time.sleep')
    def test_retrieve_evidence_no_results(self, mock_sleep, mock_fetch):
        """Test evidence retrieval when no results found."""
        mock_fetch.return_value = []

        text = "Test post"
        result = retrieve_evidence_for_post(text, max_per_source=1)

        assert isinstance(result, list)
        # Should still return sources with empty results
        assert len(result) > 0
        for item in result:
            assert item["results"] == []

    @patch('analyzer.retrieval.fetch_links_from_search_url')
    @patch('analyzer.retrieval.time.sleep')
    def test_retrieve_evidence_specific_sources(self, mock_sleep, mock_fetch):
        """Test evidence retrieval with specific sources."""
        mock_fetch.return_value = [
            {"title": "Test Article", "url": "https://example.com/article", "verdict": "unclear", "snippet": "test"}
        ]

        text = "Test post"
        sources = ["maldita", "newtral"]
        result = retrieve_evidence_for_post(text, sources=sources, max_per_source=1)

        assert len(result) == 2
        source_names = [item["source"] for item in result]
        assert "maldita" in source_names
        assert "newtral" in source_names


class TestFormatEvidence:
    """Test format_evidence function."""

    def test_format_evidence_basic(self):
        """Test basic evidence formatting."""
        evidence = [
            {
                "label": "Maldita.es",
                "search_url": "https://www.maldita.es/buscador/?s=test",
                "results": [
                    {"title": "Test Article 1", "url": "https://example.com/1"},
                    {"title": "Test Article 2", "url": "https://example.com/2"}
                ]
            }
        ]
        result = format_evidence(evidence)
        assert isinstance(result, str)
        assert "Maldita.es" in result
        assert "Test Article 1" in result
        assert "https://example.com/1" in result

    def test_format_evidence_empty_results(self):
        """Test formatting evidence with empty results."""
        evidence = [
            {
                "label": "Maldita.es",
                "search_url": "https://www.maldita.es/buscador/?s=test",
                "results": []
            }
        ]
        result = format_evidence(evidence)
        assert isinstance(result, str)
        assert "Maldita.es" in result
        assert "https://www.maldita.es/buscador/?s=test" in result

    def test_format_evidence_multiple_sources(self):
        """Test formatting evidence from multiple sources."""
        evidence = [
            {
                "label": "Source 1",
                "search_url": "https://source1.com/search?q=test",
                "results": [{"title": "Article 1", "url": "https://source1.com/1"}]
            },
            {
                "label": "Source 2",
                "search_url": "https://source2.com/search?q=test",
                "results": [{"title": "Article 2", "url": "https://source2.com/2"}]
            }
        ]
        result = format_evidence(evidence)
        assert "Source 1" in result
        assert "Source 2" in result
        assert "Article 1" in result
        assert "Article 2" in result


class TestConstants:
    """Test module constants."""

    def test_spanish_stopwords(self):
        """Test Spanish stopwords constant."""
        assert isinstance(SPANISH_STOPWORDS, set)
        assert len(SPANISH_STOPWORDS) > 0
        assert "que" in SPANISH_STOPWORDS
        assert "de" in SPANISH_STOPWORDS

    def test_source_templates(self):
        """Test source templates constant."""
        assert isinstance(SOURCE_TEMPLATES, dict)
        assert len(SOURCE_TEMPLATES) > 0
        assert "maldita" in SOURCE_TEMPLATES
        assert "newtral" in SOURCE_TEMPLATES

        # Check template structure
        maldita = SOURCE_TEMPLATES["maldita"]
        assert "label" in maldita
        assert "template" in maldita
        assert "{q}" in maldita["template"]

    def test_fact_checkers(self):
        """Test fact checkers constant."""
        assert isinstance(FACT_CHECKERS, set)
        assert len(FACT_CHECKERS) > 0
        assert "maldita" in FACT_CHECKERS
        assert "newtral" in FACT_CHECKERS

    def test_debunk_keywords(self):
        """Test debunk keywords constant."""
        assert isinstance(DEBUNK_KEYWORDS, list)
        assert len(DEBUNK_KEYWORDS) > 0
        assert "falso" in DEBUNK_KEYWORDS
        assert "desmiente" in DEBUNK_KEYWORDS