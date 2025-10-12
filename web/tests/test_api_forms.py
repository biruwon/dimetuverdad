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
        # Mock the database connection used by the local functions in the index route
        with patch('web.utils.helpers.get_db_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            
            # Mock the database queries used by get_all_accounts
            mock_conn.execute.return_value.fetchall.return_value = [
                {'username': 'testuser', 'profile_pic_url': 'http://example.com/pic.jpg', 'last_scraped': '2023-01-01'}
            ]
            mock_conn.execute.return_value.fetchone.return_value = (1,)  # Total count
            
            # Mock the response for the template rendering
            response = client.get('/')
            assert response.status_code == 200
            assert b'dimetuverdad' in response.data

    def test_user_template_context(self, client, mock_database):
        """Test user template renders with correct context."""
        # Mock the functions called by the user page
        with patch('web.app.get_user_profile_data') as mock_get_profile, \
             patch('web.app.get_user_tweets_data') as mock_get_tweets, \
             patch('web.app.get_user_analysis_stats') as mock_get_stats:

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
        with patch('web.app.get_user_profile_data') as mock_get_profile, \
             patch('web.app.get_user_tweets_data') as mock_get_tweets, \
             patch('web.app.get_user_analysis_stats') as mock_get_stats:

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
        # Mock the database connection used by the local functions in the index route
        with patch('web.utils.helpers.get_db_connection') as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn
            
            # Mock the database queries used by get_all_accounts
            mock_conn.execute.return_value.fetchall.return_value = [
                {'username': 'testuser', 'profile_pic_url': 'http://example.com/pic.jpg', 'last_scraped': '2023-01-01'}
            ]
            mock_conn.execute.return_value.fetchone.return_value = (1,)  # Total count
            
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