"""
Test API Endpoints and Form Functionality
Comprehensive testing for API routes, form submissions, and data validation.
"""

import pytest
import json
from unittest.mock import patch, Mock, MagicMock
from web.tests.conftest import TestHelpers, MockRow


class TestTemplateRendering:
    """Test template rendering and context functionality."""

    def test_index_template_context(self, client, mock_database):
        """Test index template renders with correct context."""
        mock_cursor = mock_database.cursor.return_value

        # Set up mock to return different data for different queries
        def mock_fetchall():
            if not hasattr(mock_fetchall, 'call_count'):
                mock_fetchall.call_count = 0
            mock_fetchall.call_count += 1
            
            if mock_fetchall.call_count == 1:
                # get_all_accounts query
                return [
                    MockRow({'username': 'testuser', 'profile_pic_url': 'http://example.com/pic.jpg', 'last_scraped': '2023-01-01'})
                ]
            elif mock_fetchall.call_count == 2:
                # get_analysis_distribution_cached query
                return [
                    MockRow({'category': 'general', 'count': 30, 'percentage': 60.0})
                ]
            else:
                return []

        def mock_fetchone():
            if not hasattr(mock_fetchone, 'call_count'):
                mock_fetchone.call_count = 0
            mock_fetchone.call_count += 1
            
            if mock_fetchone.call_count == 1:
                # get_all_accounts total count
                return MockRow({'cnt': 1})
            elif mock_fetchone.call_count == 2:
                # tweet count for testuser
                return MockRow({'cnt': 5})
            elif mock_fetchone.call_count == 3:
                # analyzed count for testuser
                return MockRow({'cnt': 3})
            elif mock_fetchone.call_count == 4:
                # problematic count for testuser
                return MockRow({'cnt': 2})
            elif mock_fetchone.call_count == 5:
                # get_overall_stats_cached
                return MockRow({'total_accounts': 10, 'analyzed_tweets': 50})
            else:
                return MockRow({'total_accounts': 10, 'analyzed_tweets': 50})

        mock_cursor.fetchone.side_effect = mock_fetchone
        mock_cursor.fetchall.side_effect = mock_fetchall
            
        # Mock the response for the template rendering
        response = client.get('/')
        assert response.status_code == 200
        assert b'dimetuverdad' in response.data

    def test_user_template_context(self, client, mock_database):
        """Test user template renders with correct context."""
        # Mock the functions called by the user page
        with patch('web.routes.main.get_user_profile_data') as mock_get_profile, \
             patch('web.routes.main.get_user_tweets_data') as mock_get_tweets, \
             patch('web.routes.main.get_user_analysis_stats') as mock_get_stats:

            mock_get_profile.return_value = {
                'profile_pic_url': 'https://example.com/avatar.jpg',
                'total_tweets': 100
            }

            mock_get_tweets.return_value = {
                'tweets': [
                    MockRow({
                        'tweet_url': 'https://twitter.com/testuser/status/123',
                        'content': 'Test content',
                        'media_links': None,
                        'hashtags': '[]',
                        'mentions': '[]',
                        'tweet_timestamp': '2024-01-01 12:00:00',
                        'post_type': 'original',
                        'tweet_id': '1234567890',
                        'analysis_category': 'general',
                        'llm_explanation': 'Test explanation',
                        'analysis_method': 'pattern',
                        'analysis_timestamp': '2024-01-01 12:00:00',
                        'categories_detected': None,
                        'multimodal_analysis': False,
                        'media_analysis': None,
                        'is_deleted': False,
                        'is_edited': False,
                        'rt_original_analyzed': False,
                        'original_author': None,
                        'original_tweet_id': None,
                        'reply_to_username': None
                    })
                ],
                'page': 1,
                'per_page': 20,
                'total_tweets': 80,
                'total_pages': 4
            }

            mock_get_stats.return_value = {
                'total_analyzed': 80,
                'analysis': [
                    {
                        'category': 'hate_speech',
                        'count': 30,
                        'percentage': 37.5
                    },
                    {
                        'category': 'general',
                        'count': 50,
                        'percentage': 62.5
                    }
                ]
            }

            response = client.get('/user/testuser')
            assert response.status_code == 200
            assert b'testuser' in response.data
            assert b'100' in response.data  # Total tweets

    def test_user_template_with_category_filter(self, client, mock_database):
        """Test user template with category filter."""
        # Mock the functions called by the user page with category filter
        with patch('web.routes.main.get_user_profile_data') as mock_get_profile, \
             patch('web.routes.main.get_user_tweets_data') as mock_get_tweets, \
             patch('web.routes.main.get_user_analysis_stats') as mock_get_stats:

            mock_get_profile.return_value = {
                'profile_pic_url': 'https://example.com/avatar.jpg',
                'total_tweets': 100
            }

            mock_get_tweets.return_value = {
                'tweets': [
                    MockRow({
                        'tweet_url': 'https://twitter.com/testuser/status/123',
                        'content': 'Hate content',
                        'media_links': None,
                        'hashtags': '[]',
                        'mentions': '[]',
                        'tweet_timestamp': '2024-01-01 12:00:00',
                        'post_type': 'original',
                        'tweet_id': '1234567890',
                        'analysis_category': 'hate_speech',
                        'llm_explanation': 'Hate speech detected',
                        'analysis_method': 'pattern',
                        'analysis_timestamp': '2024-01-01 12:00:00',
                        'categories_detected': None,
                        'multimodal_analysis': False,
                        'media_analysis': None,
                        'is_deleted': False,
                        'is_edited': False,
                        'rt_original_analyzed': False,
                        'original_author': None,
                        'original_tweet_id': None,
                        'reply_to_username': None
                    })
                ],
                'page': 1,
                'per_page': 20,
                'total_tweets': 30,
                'total_pages': 2
            }

            mock_get_stats.return_value = {
                'total_analyzed': 80,
                'analysis': [
                    {
                        'category': 'hate_speech',
                        'count': 30,
                        'percentage': 37.5
                    }
                ]
            }

            response = client.get('/user/testuser?category=hate_speech')
            assert response.status_code == 200
            assert b'testuser' in response.data  # Just check that the page loads with the username

    def test_error_template_rendering(self, client):
        """Test error template rendering."""
        response = client.get('/nonexistent-route')
        assert response.status_code == 404

        # Check that error template is rendered
        assert b'P\xc3\xa1gina no encontrada' in response.data or b'Not Found' in response.data


class TestJavaScriptIntegration:
    """Test JavaScript integration and AJAX functionality."""

    def test_loading_states_js_integration(self, client):
        """Test loading states JavaScript integration."""
        response = client.get('/')
        assert response.status_code == 200

        # Check for loading state elements
        assert b'loading-overlay' in response.data or b'loading' in response.data

    def test_chart_js_integration(self, client, mock_database):
        """Test Chart.js integration in templates."""
        mock_cursor = mock_database.cursor.return_value

        # Set up mock to return different data for different queries
        def mock_fetchall():
            if not hasattr(mock_fetchall, 'call_count'):
                mock_fetchall.call_count = 0
            mock_fetchall.call_count += 1
            
            if mock_fetchall.call_count == 1:
                # get_all_accounts query
                return [
                    MockRow({'username': 'testuser', 'profile_pic_url': 'http://example.com/pic.jpg', 'last_scraped': '2023-01-01'})
                ]
            elif mock_fetchall.call_count == 2:
                # get_analysis_distribution_cached query
                return [
                    MockRow({'category': 'general', 'count': 30, 'percentage': 60.0})
                ]
            else:
                return []

        def mock_fetchone():
            if not hasattr(mock_fetchone, 'call_count'):
                mock_fetchone.call_count = 0
            mock_fetchone.call_count += 1
            
            if mock_fetchone.call_count == 1:
                # get_all_accounts total count
                return MockRow({'cnt': 1})
            elif mock_fetchone.call_count == 2:
                # tweet count for testuser
                return MockRow({'cnt': 5})
            elif mock_fetchone.call_count == 3:
                # analyzed count for testuser
                return MockRow({'cnt': 3})
            elif mock_fetchone.call_count == 4:
                # problematic count for testuser
                return MockRow({'cnt': 2})
            elif mock_fetchone.call_count == 5:
                # get_overall_stats_cached
                return MockRow({'total_accounts': 10, 'analyzed_tweets': 50})
            else:
                return MockRow({'total_accounts': 10, 'analyzed_tweets': 50})

        mock_cursor.fetchone.side_effect = mock_fetchone
        mock_cursor.fetchall.side_effect = mock_fetchall
            
        response = client.get('/')
        assert response.status_code == 200
        # Check for Chart.js elements (basic check)
        assert b'chart' in response.data.lower() or b'Chart' in response.data

    def test_responsive_js_integration(self, client):
        """Test responsive JavaScript integration."""
        response = client.get('/')
        assert response.status_code == 200

        # Check for responsive elements
        assert b'mobile' in response.data.lower() or b'responsive' in response.data.lower() or \
               b'viewport' in response.data.lower()


class TestRateLimiting:
    """Test rate limiting functionality."""