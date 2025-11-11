"""
Test Admin Functionality
Comprehensive testing for admin-specific features and operations.
"""

from unittest.mock import patch, Mock
from web.tests.conftest import MockRow


class TestAdminEditAnalysis:
    """Test admin edit analysis functionality."""

    def test_edit_analysis_requires_auth(self, client):
        """Test edit analysis requires admin authentication."""
        response = client.get('/admin/edit-analysis/1234567890')
        assert response.status_code == 302  # Redirect to login

    def test_edit_analysis_get_existing_tweet(self, admin_client, mock_database, sample_tweet_data):
        """Test GET request for editing existing tweet."""
        # Mock the database query result
        mock_row = MockRow({
            'content': sample_tweet_data['content'],
            'username': sample_tweet_data['username'],
            'tweet_timestamp': sample_tweet_data['tweet_timestamp'],
            'is_deleted': False,
            'media_links': '',
            'category': sample_tweet_data['category'],
            'local_explanation': sample_tweet_data.get('local_explanation', ''),
            'external_explanation': sample_tweet_data.get('external_explanation', ''),
            'tweet_url': sample_tweet_data['tweet_url'],
            'original_content': sample_tweet_data['content'],
            'verification_data': None,
            'verification_confidence': 0.0
        })
        
        # Override the mock to return our row
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None
            
            mock_database.execute.return_value.fetchone.return_value = mock_row

            response = admin_client.get('/admin/edit-analysis/1234567890', follow_redirects=True)
            assert response.status_code == 200
            assert sample_tweet_data['tweet_url'].encode('utf-8') in response.data
            assert sample_tweet_data['username'].encode('utf-8') in response.data

    def test_edit_analysis_get_nonexistent_tweet(self, admin_client, mock_database):
        """Test GET request for editing non-existent tweet."""
        mock_database.execute.return_value.fetchone.return_value = None

        response = admin_client.get('/admin/edit-analysis/9999999999')
        assert response.status_code == 302  # Redirect to dashboard

    def test_edit_analysis_manual_update(self, admin_client, mock_database):
        """Test manual update of tweet analysis."""
        # Mock the repository calls used by handle_manual_update_action
        with patch('web.utils.helpers.get_content_analysis_repository') as mock_content_repo, \
             patch('web.utils.helpers.get_tweet_repository') as mock_tweet_repo:

            mock_content_repo.return_value.get_analysis_by_post_id.return_value = None  # No existing analysis
            mock_tweet_repo.return_value.get_tweet_by_id.return_value = {
                'username': 'testuser',
                'content': 'Test content',
                'tweet_url': 'https://twitter.com/test/status/123'
            }
            mock_content_repo.return_value.save_analysis.return_value = True

            response = admin_client.post('/admin/edit-analysis/1234567890',
                                       data={
                                           'action': 'update',
                                           'category': 'hate_speech',
                                           'explanation': 'Updated analysis'
                                       },
                                       follow_redirects=False)

            assert response.status_code == 302  # Redirect after successful update
            # Verify repository operations were called
            mock_content_repo.return_value.get_analysis_by_post_id.assert_called_once_with('1234567890')
            mock_tweet_repo.return_value.get_tweet_by_id.assert_called_once_with('1234567890')
            mock_content_repo.return_value.save_analysis.assert_called_once()

    def test_edit_analysis_reanalyze_action(self, admin_client, mock_database, sample_tweet_data):
        """Test reanalyze action."""
        with patch('web.utils.helpers.get_tweet_data') as mock_get_tweet, \
             patch('web.utils.helpers.reanalyze_tweet') as mock_reanalyze:

            mock_get_tweet.return_value = sample_tweet_data
            mock_reanalyze.return_value = Mock(category='disinformation', explanation='Reanalyzed content')

            response = admin_client.post('/admin/edit-analysis/1234567890',
                                       data={'action': 'reanalyze'},
                                       follow_redirects=False)

            assert response.status_code == 302  # Redirect after action
            mock_reanalyze.assert_called_once_with('1234567890', verbose=True)

    def test_edit_analysis_refresh_action(self, admin_client, mock_database, sample_tweet_data):
        """Test refresh action."""
        with patch('web.utils.helpers.get_tweet_data') as mock_get_tweet, \
             patch('web.utils.helpers.refetch_tweet') as mock_refetch:

            mock_get_tweet.return_value = sample_tweet_data
            mock_refetch.return_value = True

            response = admin_client.post('/admin/edit-analysis/1234567890',
                                       data={'action': 'refresh'},
                                       follow_redirects=False)

            assert response.status_code == 302
            mock_refetch.assert_called_once_with('1234567890')

    def test_edit_analysis_refresh_and_reanalyze(self, admin_client, mock_database, sample_tweet_data):
        """Test refresh and reanalyze combined action."""
        with patch('web.utils.helpers.get_tweet_data') as mock_get_tweet, \
             patch('web.utils.helpers.refetch_tweet') as mock_refetch, \
             patch('web.utils.helpers.reanalyze_tweet') as mock_reanalyze:

            mock_get_tweet.return_value = sample_tweet_data
            mock_refetch.return_value = True
            mock_reanalyze.return_value = Mock(category='conspiracy_theory')

            response = admin_client.post('/admin/edit-analysis/1234567890',
                                       data={'action': 'refresh_and_reanalyze'},
                                       follow_redirects=False)

            assert response.status_code == 302
            mock_refetch.assert_called_once_with('1234567890')
            mock_reanalyze.assert_called_once_with('1234567890', verbose=True)


class TestAdminCategoryViews:
    """Test admin category view functionality."""

    def test_category_view_requires_auth(self, client):
        """Test category view requires admin authentication."""
        response = client.get('/admin/category/hate_speech')
        assert response.status_code == 302

    def test_category_view_existing_category(self, admin_client, mock_database):
        """Test viewing existing category."""
        # Mock the database queries
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None
            
            mock_database.execute.side_effect = [
                Mock(fetchone=Mock(return_value=MockRow({'cnt': 1}))),  # Category exists
                Mock(fetchone=Mock(return_value=MockRow({'cnt': 100}))),  # Total count
                Mock(fetchall=Mock(return_value=[MockRow({
                    'tweet_url': 'https://twitter.com/user1/status/123',
                    'content': 'Test content',
                    'username': 'user1',
                    'tweet_timestamp': '2024-01-01 12:00:00',
                    'tweet_id': '123',
                    'category': 'hate_speech',
                    'local_explanation': 'Test explanation',
                    'external_explanation': '',
                    'analysis_stages': 'pattern',
                    'external_analysis_used': False,
                    'analysis_timestamp': '2024-01-01 12:00:00',
                    'is_deleted': False,
                    'is_edited': False,
                    'post_type': 'original'
                })])),  # Recent analyses
                Mock(fetchone=Mock(return_value=MockRow({
                    'local_llm_count': 50, 
                    'external_count': 10, 
                    'unique_users': 30
                }))),  # Category stats
                Mock(fetchall=Mock(return_value=[
                    MockRow({'username': 'user1', 'tweet_count': 25}),
                    MockRow({'username': 'user2', 'tweet_count': 15})
                ]))  # Top users
            ]

            response = admin_client.get('/admin/category/hate_speech', follow_redirects=True)
            assert response.status_code == 200
            assert 'hate_speech'.encode('utf-8') in response.data

    def test_category_view_nonexistent_category(self, admin_client, mock_database):
        """Test viewing non-existent category."""
        mock_database.execute.return_value.fetchone.return_value = [0]  # No tweets in category

        response = admin_client.get('/admin/category/nonexistent')
        assert response.status_code == 302  # Redirect to dashboard

    def test_category_view_pagination(self, admin_client, mock_database):
        """Test category view pagination."""
        mock_database.execute.side_effect = [
            Mock(fetchone=Mock(return_value=MockRow({'cnt': 1}))),  # Category exists (mapping)
            Mock(fetchone=Mock(return_value=MockRow({'cnt': 100}))),  # Total count (mapping)
            Mock(fetchall=Mock(return_value=[])),  # Empty results for page 2
            Mock(fetchone=Mock(return_value=MockRow({'local_llm_count': 50, 'external_count': 10, 'unique_users': 30}))),  # Category stats (mapping)
            Mock(fetchall=Mock(return_value=[]))  # Top users
        ]

        response = admin_client.get('/admin/category/hate_speech?page=2', follow_redirects=True)
        assert response.status_code == 200


class TestAdminQuickEdit:
    """Test admin quick edit functionality."""

    def test_quick_edit_requires_auth(self, client):
        """Test quick edit requires admin authentication."""
        response = client.post('/admin/quick-edit-category/1234567890',
                             data={'category': 'general'})
        assert response.status_code == 302

    def test_quick_edit_success(self, admin_client, mock_database):
        """Test successful quick category edit."""
        # Mock existing analysis
        mock_database.execute.side_effect = [
            Mock(fetchone=Mock(return_value=MockRow({'post_id': '1234567890'}))),  # Existing analysis found (mapping)
            Mock(rowcount=1),  # Update successful
        ]

        response = admin_client.post('/admin/quick-edit-category/1234567890',
                                   data={'category': 'disinformation'},
                                   follow_redirects=False)

        assert response.status_code == 302  # Redirect after successful edit

    def test_quick_edit_create_new_analysis(self, admin_client, mock_database):
        """Test quick edit creates new analysis if none exists."""
        # Mock no existing analysis, then tweet data, then successful insert
        mock_database.execute.side_effect = [
            Mock(fetchone=Mock(return_value=None)),  # No existing analysis
            Mock(fetchone=Mock(return_value=MockRow({'username': 'testuser', 'content': 'Test content', 'tweet_url': 'https://twitter.com/test/status/123'}))),  # Tweet data for insert
            Mock(rowcount=1),  # Insert successful
        ]

        response = admin_client.post('/admin/quick-edit-category/1234567890',
                                   data={'category': 'conspiracy_theory'},
                                   follow_redirects=False)

        assert response.status_code == 302  # Redirect after successful creation

    def test_quick_edit_missing_category(self, admin_client):
        """Test quick edit with missing category."""
        response = admin_client.post('/admin/quick-edit-category/1234567890',
                                   data={},  # No category provided
                                   follow_redirects=False)

        assert response.status_code == 302  # Redirect with error


class TestAdminExport:
    """Test admin export functionality."""

    def test_export_csv_requires_auth(self, client):
        """Test CSV export requires admin authentication."""
        response = client.get('/admin/export/csv')
        assert response.status_code == 302

    def test_export_csv_success(self, admin_client, mock_database):
        """Test successful CSV export."""
        # Mock export data
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            MockRow({
                'post_id': '1234567890',
                'author_username': 'testuser',
                'category': 'general',
                'local_explanation': 'Test explanation',
                'external_explanation': '',
                'analysis_stages': 'pattern',
                'external_analysis_used': False,
                'analysis_timestamp': '2024-01-01 12:00:00',
                'tweet_content': 'Test content',
                'tweet_url': 'https://twitter.com/test/status/123',
                'tweet_timestamp': '2024-01-01 12:00:00'
            })
        ]
        mock_database.cursor.return_value = mock_cursor

        response = admin_client.get('/admin/export/csv')
        assert response.status_code == 200
        assert 'text/csv' in response.content_type
        assert 'attachment' in response.headers.get('Content-Disposition', '')
        assert b'Post ID,Author Username,Category' in response.data

    def test_export_json_success(self, admin_client, mock_database):
        """Test successful JSON export."""
        # Mock export data
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            MockRow({
                'post_id': '1234567890',
                'author_username': 'testuser',
                'category': 'general',
                'local_explanation': 'Test explanation',
                'external_explanation': '',
                'analysis_stages': 'pattern',
                'external_analysis_used': False,
                'analysis_timestamp': '2024-01-01 12:00:00',
                'post_content': 'Test content',
                'post_url': 'https://twitter.com/test/status/123',
                'categories_detected': None,
                'verification_data': None,
                'verification_confidence': 0.0,
                'tweet_content': 'Test tweet content',
                'tweet_url': 'https://twitter.com/test/status/123',
                'tweet_timestamp': '2024-01-01 12:00:00'
            })
        ]
        mock_database.cursor.return_value = mock_cursor

        # Patch get_db_connection_context directly
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None

            response = admin_client.get('/admin/export/json')
            assert response.status_code == 200
            assert 'application/json' in response.content_type
            assert 'attachment' in response.headers.get('Content-Disposition', '')

            data = response.get_json()
            assert 'data' in data
            assert len(data['data']) == 1
            assert data['data'][0]['post_id'] == '1234567890'


class TestAdminReanalyzeSingle:
    """Test single tweet reanalysis functionality."""

    def test_reanalyze_single_requires_auth(self, client):
        """Test single reanalysis requires admin authentication."""
        response = client.post('/admin/reanalyze-single/1234567890')
        assert response.status_code == 302

    def test_reanalyze_single_success(self, admin_client, mock_database, sample_tweet_data):
        """Test successful single tweet reanalysis."""
        with patch('web.utils.helpers.get_tweet_data') as mock_get_tweet, \
             patch('web.utils.helpers.reanalyze_tweet') as mock_reanalyze:

            mock_get_tweet.return_value = sample_tweet_data
            mock_reanalyze.return_value = Mock(category='hate_speech', explanation='Updated analysis')

            response = admin_client.post('/admin/reanalyze-single/1234567890',
                                       follow_redirects=False)

            assert response.status_code == 302  # Redirect to user page

    def test_reanalyze_single_no_tweet(self, admin_client):
        """Test reanalysis of non-existent tweet."""
        with patch('web.utils.helpers.get_tweet_data') as mock_get_tweet:
            mock_get_tweet.return_value = None

            response = admin_client.post('/admin/reanalyze-single/9999999999',
                                       follow_redirects=False)

            assert response.status_code == 302  # Redirect with error

    def test_reanalyze_single_analysis_failure(self, admin_client, mock_database, sample_tweet_data):
        """Test reanalysis when analysis fails."""
        with patch('web.app.get_tweet_data') as mock_get_tweet, \
             patch('web.app.reanalyze_tweet') as mock_reanalyze:

            mock_get_tweet.return_value = sample_tweet_data
            mock_reanalyze.return_value = None  # Analysis failed

            response = admin_client.post('/admin/reanalyze-single/1234567890',
                                       follow_redirects=False)

            assert response.status_code == 302  # Redirect with error


class TestAdminUserCategory:
    """Test admin user category view functionality."""

    def test_user_category_redirect(self, admin_client):
        """Test user category view redirects to user page."""
        response = admin_client.get('/admin/user-category/testuser/hate_speech')
        assert response.status_code == 302
        assert '/user/testuser' in response.headers.get('Location', '')
        assert 'category=hate_speech' in response.headers.get('Location', '')


class TestAdminFetch:
    """Test admin fetch functionality."""

    def test_fetch_requires_auth(self, client):
        """Test fetch requires admin authentication."""
        response = client.post('/admin/fetch', data={'username': 'testuser'})
        assert response.status_code == 302

    @patch('subprocess.run')
    @patch('threading.Thread')
    def test_fetch_only_action(self, mock_thread, mock_subprocess, admin_client, mock_database):
        """Test fetch_only action initiates data fetch without analysis."""
        # Mock database check for user existence
        mock_database.execute.return_value.fetchone.return_value = MockRow({'cnt': 0})  # User doesn't exist
        
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None
            
            response = admin_client.post('/admin/fetch', 
                                       data={'username': 'testuser', 'action': 'fetch_only'},
                                       follow_redirects=False)  # Don't follow redirects to check flash messages
            
            assert response.status_code == 200  # Returns loading page
            assert b'testuser' in response.data  # Username in loading page
            mock_thread.assert_called_once()

    @patch('subprocess.run')
    @patch('threading.Thread')
    def test_fetch_and_analyze_action(self, mock_thread, mock_subprocess, admin_client, mock_database):
        """Test fetch_and_analyze action initiates both fetch and analysis."""
        # Mock database check for user existence
        mock_database.execute.return_value.fetchone.return_value = MockRow({'cnt': 10})  # User exists
        
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None
            
            response = admin_client.post('/admin/fetch', 
                                       data={'username': 'existinguser', 'action': 'fetch_and_analyze'},
                                       follow_redirects=False)
            
            assert response.status_code == 200
            assert b'existinguser' in response.data
            mock_thread.assert_called_once()

    @patch('subprocess.run')
    @patch('threading.Thread')
    def test_refetch_all_action(self, mock_thread, mock_subprocess, admin_client, mock_database):
        """Test refetch_all action initiates complete refetch with data deletion."""
        # Mock database check for user existence
        mock_database.execute.return_value.fetchone.return_value = MockRow({'cnt': 50})  # User exists
        
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None
            
            response = admin_client.post('/admin/fetch', 
                                       data={'username': 'existinguser', 'action': 'refetch_all'},
                                       follow_redirects=False)
            
            assert response.status_code == 200
            assert b'existinguser' in response.data
            mock_thread.assert_called_once()

    @patch('subprocess.run')
    @patch('threading.Thread')
    def test_fetch_with_max_parameter(self, mock_thread, mock_subprocess, admin_client, mock_database):
        """Test fetch with max tweets parameter."""
        mock_database.execute.return_value.fetchone.return_value = MockRow({'cnt': 0})
        
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None
            
            response = admin_client.post('/admin/fetch',
                                       data={'username': 'testuser', 'action': 'fetch_only', 'max': '100'},
                                       follow_redirects=True)
            
            assert response.status_code == 200
            mock_thread.assert_called_once()

    def test_fetch_missing_username(self, admin_client):
        """Test fetch with missing username."""
        response = admin_client.post('/admin/fetch',
                                   data={'action': 'fetch_only'},
                                   follow_redirects=True)
        
        assert response.status_code == 200
        # Flash message is in session, not in HTML content after redirect
        assert b'admin' in response.data or b'dashboard' in response.data.lower()
class TestAdminViewFeedback:
    """Test admin feedback view functionality."""

    def test_view_feedback_requires_auth(self, client):
        """Test feedback view requires admin authentication."""
        response = client.get('/admin/feedback')
        assert response.status_code == 302

    def test_view_feedback_success(self, admin_client, mock_database):
        """Test successful feedback view with data."""
        # Mock database calls for feedback view
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.return_value = mock_database
            mock_context.return_value.__exit__.return_value = None
            
            mock_database.execute.side_effect = [
                Mock(fetchone=Mock(return_value=MockRow({'cnt': 50}))),  # Total count
                Mock(fetchall=Mock(return_value=[
                    MockRow({
                        'id': 1,
                        'post_id': '1234567890',
                        'feedback_type': 'correction',
                        'original_category': 'general',
                        'corrected_category': 'hate_speech',
                        'user_comment': 'This should be hate speech',
                        'user_ip': '192.168.1.1',
                        'submitted_at': '2024-01-01 12:00:00',
                        'username': 'testuser',
                        'content': 'Test tweet content',
                        'tweet_url': 'https://twitter.com/test/status/123'
                    })
                ])),  # Feedback submissions
                Mock(fetchone=Mock(return_value=MockRow({
                    'total_feedback': 50,
                    'corrections': 30,
                    'improvements': 15,
                    'bug_reports': 5,
                    'unique_posts': 45
                })))  # Feedback stats
            ]

            response = admin_client.get('/admin/feedback')
            assert response.status_code == 200
            assert b'feedback' in response.data.lower()
            assert b'correction' in response.data.lower()

    def test_view_feedback_pagination(self, admin_client, mock_database):
        """Test feedback view pagination."""
        mock_database.execute.side_effect = [
            Mock(fetchone=Mock(return_value=MockRow({'cnt': 100}))),  # Total count
            Mock(fetchall=Mock(return_value=[])),  # Empty results for page 2
            Mock(fetchone=Mock(return_value=MockRow({
                'total_feedback': 100,
                'corrections': 60,
                'improvements': 30,
                'bug_reports': 10,
                'unique_posts': 90
            })))  # Feedback stats
        ]

        response = admin_client.get('/admin/feedback?page=2')
        assert response.status_code == 200

    def test_view_feedback_empty(self, admin_client, mock_database):
        """Test feedback view with no feedback submissions."""
        mock_database.execute.side_effect = [
            Mock(fetchone=Mock(return_value=MockRow({'cnt': 0}))),  # No feedback
            Mock(fetchall=Mock(return_value=[])),  # Empty results
            Mock(fetchone=Mock(return_value=MockRow({
                'total_feedback': 0,
                'corrections': 0,
                'improvements': 0,
                'bug_reports': 0,
                'unique_posts': 0
            })))  # Empty stats
        ]

        response = admin_client.get('/admin/feedback')
        assert response.status_code == 200

    def test_view_feedback_database_error(self, admin_client, mock_database):
        """Test feedback view handles database errors gracefully."""
        with patch('database.get_db_connection_context') as mock_context:
            mock_context.return_value.__enter__.side_effect = Exception("Database connection failed")
            mock_context.return_value.__exit__.return_value = None

            response = admin_client.get('/admin/feedback')
            assert response.status_code == 302  # Redirect to dashboard on error