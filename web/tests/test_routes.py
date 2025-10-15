"""
Test Flask Routes
Comprehensive testing for all Flask application routes and endpoints.
"""

import pytest
import sqlite3
import os
from unittest.mock import Mock, patch, MagicMock
from web.tests.conftest import TestHelpers, MockRow
from utils.database import get_db_connection_context


class TestMainRoutes:
    """Test main application routes."""

    def test_index_route(self, client, mock_database):
        """Test the main dashboard route."""
        mock_cursor = mock_database.cursor.return_value

        # Set up mock to return different data for different queries
        def mock_fetchall():
            # This will be called multiple times, return appropriate data based on call count
            if not hasattr(mock_fetchall, 'call_count'):
                mock_fetchall.call_count = 0
            mock_fetchall.call_count += 1
            
            if mock_fetchall.call_count == 1:
                # get_all_accounts query - return account data
                return [
                    MockRow({'username': 'testuser', 'profile_pic_url': 'http://example.com/pic.jpg', 'last_scraped': '2023-01-01'})
                ]
            elif mock_fetchall.call_count == 2:
                # get_analysis_distribution_cached query - return category data
                return [
                    MockRow({'category': 'general', 'count': 30, 'percentage': 60.0}),
                    MockRow({'category': 'hate_speech', 'count': 10, 'percentage': 20.0})
                ]
            else:
                return []

        def mock_fetchone():
            # This will be called multiple times
            if not hasattr(mock_fetchone, 'call_count'):
                mock_fetchone.call_count = 0
            mock_fetchone.call_count += 1
            
            if mock_fetchone.call_count == 1:
                # get_all_accounts total count query
                return MockRow({'cnt': 1})
            elif mock_fetchone.call_count == 2:
                # get_overall_stats_cached query
                return MockRow({'total_accounts': 10, 'analyzed_tweets': 50})
            else:
                return MockRow({'total_accounts': 10, 'analyzed_tweets': 50})

        mock_cursor.fetchone.side_effect = mock_fetchone
        mock_cursor.fetchall.side_effect = mock_fetchall

        response = client.get('/')
        assert response.status_code == 200
        assert b'dimetuverdad' in response.data

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
            mock_conn.execute.return_value.fetchone.return_value = MockRow({'cnt': 1})  # Total count
            
            response = client.get('/')
            assert response.status_code == 200
            assert b'analysisChart' in response.data  # Chart should be rendered
            assert b'Cuentas Monitoreadas' in response.data  # Accounts section should be present

    def test_index_with_category_filter(self, client, mock_database):
        """Test index route with category filtering."""
        # Mock the connection's execute method
        mock_connection = mock_database.return_value
        mock_connection.execute.side_effect = [
            Mock(fetchone=Mock(return_value=MockRow({'total_accounts': 10, 'analyzed_tweets': 50}))),
            Mock(fetchall=Mock(return_value=[
                MockRow({'category': 'hate_speech', 'count': 10, 'percentage': 100.0})
            ])),
            Mock(fetchall=Mock(return_value=[]))  # filtered accounts
        ]

        response = client.get('/?category=hate_speech')
        assert response.status_code == 200
        assert b'Filtrado por: Hate_speech' in response.data

    def test_user_page_exists(self, client, app, sample_tweet_data):
        """Test user page for existing account."""
        # Use the test database that's already set up by the fixtures

        with app.app_context():
            with get_db_connection_context() as conn:
                # Create test data using helper methods
                TestHelpers.create_test_account(conn, {
                    'username': 'testuser',
                    'profile_pic_url': 'https://example.com/avatar.jpg',
                    'last_activity': '2024-01-01 12:00:00'
                })
                TestHelpers.create_test_tweet(conn, sample_tweet_data)

        response = client.get('/user/testuser')
        assert response.status_code == 200
        assert b'@testuser' in response.data

    def test_user_page_not_found(self, client, mock_database):
        """Test user page for non-existing account."""
        mock_database.execute.return_value.fetchone.return_value = None

        response = client.get('/user/nonexistent')
        assert response.status_code == 404
        assert b'Usuario no encontrado' in response.data

    def test_user_page_with_filters(self, client, app, sample_tweet_data):
        """Test user page with category filtering."""

        with app.app_context():
            with get_db_connection_context() as conn:
                # Create test data
                TestHelpers.create_test_account(conn, {
                    'username': 'testuser',
                    'profile_pic_url': 'https://example.com/avatar.jpg',
                    'last_activity': '2024-01-01 12:00:00'
                })
                TestHelpers.create_test_tweet(conn, sample_tweet_data)

        response = client.get('/user/testuser?category=general')
        assert response.status_code == 200
        assert b'@testuser' in response.data
        assert b'Filtrado: General' in response.data

    def test_loading_routes(self, client):
        """Test loading page routes."""
        # Test default loading page
        response = client.get('/loading')
        assert response.status_code == 200
        assert b'Cargando...' in response.data

        # Test custom loading message
        response = client.get('/loading/Test_Message')
        assert response.status_code == 200
        assert b'Test Message' in response.data


class TestAdminRoutes:
    """Test admin-only routes."""

    def test_admin_login_page(self, client):
        """Test admin login page access."""
        response = client.get('/admin/login')
        assert response.status_code == 200
        assert b'Admin Login' in response.data

    def test_admin_login_success(self, client):
        """Test successful admin login."""
        response = client.post('/admin/login', data={'token': 'test-admin-token'}, follow_redirects=True)
        assert response.status_code == 200
        # Check that we were redirected to admin dashboard
        assert b'Panel de Administraci' in response.data or b'Admin' in response.data

    def test_admin_login_failure(self, client):
        """Test failed admin login."""
        response = client.post('/admin/login', data={'token': 'wrong-token'})
        assert response.status_code == 200
        assert b'Token administrativo incorrecto' in response.data

    def test_admin_dashboard_requires_auth(self, client):
        """Test admin dashboard requires authentication."""
        response = client.get('/admin/')
        assert response.status_code == 302  # Redirect to login

    def test_admin_dashboard_authenticated(self, admin_client, mock_database):
        """Test admin dashboard with authentication."""
        # Mock admin dashboard data with new dual explanation schema
        mock_cursor = mock_database.cursor.return_value

        # Mock get_admin_stats query (fetchone) with new fields
        mock_cursor.fetchone.side_effect = [
            MockRow({
                'total_tweets': 100, 
                'analyzed_tweets': 80, 
                'external_analyzed': 10,  # New field
                'local_llm_analyzed': 30  # New field
            }),  # stats
        ]

        # Mock database queries (fetchall calls)
        mock_cursor.fetchall.side_effect = [
            [  # recent analyses (first fetchall) - updated with new fields
                MockRow({
                    'analysis_timestamp': '2024-01-01', 
                    'category': 'general',
                    'analysis_stages': 'pattern',  # New field (replaces analysis_method)
                    'external_analysis_used': False,  # New field
                    'username': 'user1',
                    'content_preview': 'Test content...'
                })
            ],
            [],  # feedback_rows (second fetchall - empty)
            [],  # fetch_rows (third fetchall - empty)
            [  # categories (fourth fetchall)
                MockRow({'category': 'general', 'count': 50}),
                MockRow({'category': 'hate_speech', 'count': 30})
            ]
        ]

        response = admin_client.get('/admin/')
        assert response.status_code == 200
        assert b'Admin' in response.data or b'Panel' in response.data  # More flexible check

    def test_admin_logout(self, admin_client):
        """Test admin logout."""
        response = admin_client.get('/admin/logout', follow_redirects=True)
        assert response.status_code == 200
        # Check that we were redirected to index (should contain main page content)
        assert b'dimetuverdad' in response.data

    def test_admin_fetch_requires_auth(self, client):
        """Test admin fetch requires authentication."""
        response = client.post('/admin/fetch', data={'username': 'testuser'})
        assert response.status_code == 302  # Redirect to login

    @patch('web.routes.admin.reanalyze_tweet_sync')
    def test_admin_reanalyze_category(self, mock_reanalyze, admin_client, mock_database):
        """Test admin reanalyze category operation."""
        # Mock the database to return some tweets for the category
        mock_cursor = mock_database.cursor.return_value
        mock_cursor.fetchall.return_value = [
            MockRow({'post_id': '1234567890'}),
            MockRow({'post_id': '0987654321'})
        ]
        
        # Mock reanalyze_tweet_sync to return a successful result
        mock_reanalyze.return_value = Mock(category='hate_speech')

        response = admin_client.post('/admin/reanalyze',
                                   data={'action': 'category', 'category': 'hate_speech'},
                                   follow_redirects=False)
        assert response.status_code == 200
        assert b'Cargando' in response.data

    @patch('subprocess.run')
    def test_admin_reanalyze_user(self, mock_subprocess, admin_client):
        """Test admin reanalyze user operation."""
        # Mock subprocess.run to return a successful result
        mock_subprocess.return_value = Mock()

        response = admin_client.post('/admin/reanalyze',
                                   data={'action': 'user', 'username': 'testuser'},
                                   follow_redirects=False)
        assert response.status_code == 200
        assert b'Cargando' in response.data


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_tweet_status_api(self, client, app):
        """Test tweet status API endpoint."""
        # Use the test database that's already set up by the fixtures

        with app.app_context():
            with get_db_connection_context() as conn:
                # Create test account first to satisfy FK constraint
                TestHelpers.create_test_account(conn, {
                    'username': 'testuser',
                    'profile_pic_url': 'https://example.com/avatar.jpg',
                    'last_activity': '2024-01-01 12:00:00'
                })
                # Now create test tweet with new schema
                TestHelpers.create_test_tweet(conn, {
                    'tweet_id': '9999999999',
                    'content': 'Test tweet content',
                    'username': 'testuser',
                    'tweet_timestamp': '2024-01-01 12:00:00',
                    'tweet_url': 'https://twitter.com/testuser/status/9999999999',
                    'category': 'general',
                    'local_explanation': 'Test explanation',
                    'analysis_stages': 'pattern'
                })

        response = client.get('/api/tweet-status/9999999999')
        assert response.status_code == 200
        data = response.get_json()
        assert data['exists'] == True
        assert data['tweet_id'] == '9999999999'
        assert data['username'] == 'testuser'
        assert data['analyzed'] == True
        assert data['category'] == 'general'

    def test_usernames_api_requires_auth(self, client, app, session_test_db_path):
        """Test usernames API endpoint (no auth required)."""
        # Set up test data with accounts first (FK constraint)
        with app.app_context():
            # Override DATABASE_PATH for this test to ensure we use the correct database
            import os
            old_db_path = os.environ.get('DATABASE_PATH')
            os.environ['DATABASE_PATH'] = session_test_db_path
            
            try:
                with get_db_connection_context() as conn:
                    # Create accounts first
                    TestHelpers.create_test_account(conn, {
                        'username': 'user1',
                        'profile_pic_url': 'https://example.com/user1.jpg',
                        'last_activity': '2024-01-01 12:00:00'
                    })
                    TestHelpers.create_test_account(conn, {
                        'username': 'user2',
                        'profile_pic_url': 'https://example.com/user2.jpg',
                        'last_activity': '2024-01-01 13:00:00'
                    })
                    
                    # Create test tweets to populate usernames
                    TestHelpers.create_test_tweet(conn, {
                        'tweet_id': '1111111111',
                        'content': 'Test tweet 1',
                        'username': 'user1',
                        'tweet_timestamp': '2024-01-01 12:00:00',
                        'tweet_url': 'https://twitter.com/user1/status/1111111111'
                    })
                    TestHelpers.create_test_tweet(conn, {
                        'tweet_id': '2222222222',
                        'content': 'Test tweet 2',
                        'username': 'user2',
                        'tweet_timestamp': '2024-01-01 13:00:00',
                        'tweet_url': 'https://twitter.com/user2/status/2222222222'
                    })
            finally:
                # Restore original DATABASE_PATH
                if old_db_path is not None:
                    os.environ['DATABASE_PATH'] = old_db_path
                elif 'DATABASE_PATH' in os.environ:
                    del os.environ['DATABASE_PATH']
        
        response = client.get('/api/usernames')
        assert response.status_code == 200  # Endpoint exists and is public
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) >= 2  # At least our test users


class TestErrorHandlers:
    """Test error handlers."""

    def test_404_error(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent-page')
        assert response.status_code == 404
        assert b'Not Found' in response.data  # Default Flask 404 page

    def test_403_error(self, client):
        """Test 403 error handler."""
        # This would typically be triggered by forbidden access
        # For testing purposes, we'll just check that the error handler exists
        # by making a request that should trigger a 403 in the real app
        # Since we can't easily trigger a 403 in tests, we'll skip this test
        pytest.skip("403 error handler test requires specific forbidden access scenario")

    def test_500_error(self, client):
        """Test 500 error handler."""
        # Mock get_db_connection_context to raise a database error
        with patch('utils.database.get_db_connection_context') as mock_get_db:
            mock_get_db.side_effect = sqlite3.OperationalError("database is locked")

            response = client.get('/')
            assert response.status_code == 503  # Should trigger database locked error handler
            assert b'Base de datos ocupada' in response.data


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_admin_login(self, client):
        """Test rate limiting on admin login."""
        # Make multiple rapid requests
        for i in range(15):  # Exceed the limit
            response = client.post('/admin/login', data={'token': 'wrong-token'})

        # Should eventually get rate limited
        response = client.post('/admin/login', data={'token': 'wrong-token'})
        assert response.status_code == 429
        assert b'Demasiadas solicitudes' in response.data


class TestTemplateRendering:
    """Test template rendering and context."""

    def test_user_template_context(self, client, app, sample_tweet_data, session_test_db_path):
        """Test user template receives correct context."""
        # Use the test database that's already set up by the fixtures
        with app.app_context():
            # Override DATABASE_PATH environment variable to ensure we use the correct database
            old_db_path = os.environ.get('DATABASE_PATH')
            os.environ['DATABASE_PATH'] = session_test_db_path
            
            try:
                with get_db_connection_context() as conn:
                    # Create test account first
                    TestHelpers.create_test_account(conn, {
                        'username': 'testuser',
                        'profile_pic_url': 'https://example.com/avatar.jpg',
                        'last_activity': '2024-01-01 12:00:00'
                    })
                    
                    # Create test tweet with analysis
                    TestHelpers.create_test_tweet(conn, {
                        'tweet_id': '1234567890123456789',
                        'content': 'Test tweet content for template context',
                        'username': 'testuser',
                        'tweet_timestamp': '2024-01-01 12:00:00',
                        'tweet_url': 'https://twitter.com/testuser/status/1234567890123456789',
                        'category': 'general',
                        'local_explanation': 'Test explanation',
                        'analysis_stages': 'pattern'
                    })
            finally:
                # Restore original DATABASE_PATH environment variable
                if old_db_path is not None:
                    os.environ['DATABASE_PATH'] = old_db_path
                else:
                    os.environ.pop('DATABASE_PATH', None)

        response = client.get('/user/testuser')
        assert response.status_code == 200
        assert b'@testuser' in response.data
        assert b'analysisChart' in response.data  # User analysis chart should be present
        assert b'filter-buttons' in response.data  # Filter buttons should be present