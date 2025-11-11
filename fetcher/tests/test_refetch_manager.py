"""
Unit tests for RefetchManager retry logic and refetch operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fetcher.refetch_manager import RefetchManager


class TestRefetchManagerRetryLogic:
    """Test retry logic for handling intermittent failures."""
    
    @pytest.fixture
    def refetch_manager(self):
        """Create RefetchManager instance for testing."""
        return RefetchManager()
    
    @pytest.fixture
    def mock_tweet_info(self):
        """Mock tweet info from database."""
        return ("vitoquiles", "https://x.com/vitoquiles/status/1983635249646203199")
    
    def test_refetch_single_tweet_success_first_attempt(self, refetch_manager, mock_tweet_info):
        """Test successful refetch on first attempt."""
        with patch.object(refetch_manager, 'get_tweet_info_from_db', return_value=mock_tweet_info):
            with patch('fetcher.refetch_manager.sync_playwright') as mock_playwright:
                with patch('fetcher.refetch_manager.fetcher_parsers.extract_tweet_with_media_monitoring') as mock_extract:
                    with patch('fetcher.refetch_manager.fetcher_db.update_tweet_in_database', return_value=True):
                        # Setup mocks
                        mock_page = Mock()
                        mock_context = Mock()
                        mock_browser = Mock()
                        mock_playwright.return_value.__enter__.return_value = Mock()
                        refetch_manager.session_manager.create_browser_context = Mock(
                            return_value=(mock_browser, mock_context, mock_page)
                        )
                        refetch_manager.session_manager.cleanup_session = Mock()
                        refetch_manager.scroller.delay = Mock()
                        
                        mock_extract.return_value = {'tweet_id': '1983635249646203199', 'content': 'Test'}
                        
                        # Execute
                        result = refetch_manager.refetch_single_tweet('1983635249646203199')
                        
                        # Verify
                        assert result is True
                        mock_page.goto.assert_called_once()
                        refetch_manager.session_manager.cleanup_session.assert_called_once()
    
    def test_refetch_single_tweet_retry_on_page_load_failure(self, refetch_manager, mock_tweet_info):
        """Test retry logic when page load fails initially."""
        with patch.object(refetch_manager, 'get_tweet_info_from_db', return_value=mock_tweet_info):
            with patch('fetcher.refetch_manager.sync_playwright') as mock_playwright:
                with patch('fetcher.refetch_manager.fetcher_parsers.extract_tweet_with_media_monitoring') as mock_extract:
                    with patch('fetcher.refetch_manager.fetcher_db.update_tweet_in_database', return_value=True):
                        with patch('time.sleep'):  # Mock sleep to speed up test
                            # Setup mocks
                            mock_page = Mock()
                            mock_context = Mock()
                            mock_browser = Mock()
                            mock_playwright.return_value.__enter__.return_value = Mock()
                            
                            # First attempt fails, second succeeds
                            call_count = {'count': 0}
                            
                            def create_context_side_effect(*args):
                                call_count['count'] += 1
                                if call_count['count'] == 1:
                                    # First attempt - page.goto will fail
                                    page = Mock()
                                    page.goto.side_effect = Exception("ERR_NETWORK_IO_SUSPENDED")
                                    return (mock_browser, mock_context, page)
                                else:
                                    # Second attempt - succeeds
                                    page = Mock()
                                    page.goto.return_value = None
                                    return (mock_browser, mock_context, page)
                            
                            refetch_manager.session_manager.create_browser_context = Mock(
                                side_effect=create_context_side_effect
                            )
                            refetch_manager.session_manager.cleanup_session = Mock()
                            refetch_manager.scroller.delay = Mock()
                            
                            mock_extract.return_value = {'tweet_id': '1983635249646203199', 'content': 'Test'}
                            
                            # Execute
                            result = refetch_manager.refetch_single_tweet('1983635249646203199', max_retries=2)
                            
                            # Verify - should succeed after retry
                            assert result is True
                            assert call_count['count'] == 2  # Two attempts made
                            assert refetch_manager.session_manager.cleanup_session.call_count == 2
    
    def test_refetch_single_tweet_retry_on_extraction_failure(self, refetch_manager, mock_tweet_info):
        """Test retry logic when tweet extraction fails initially."""
        with patch.object(refetch_manager, 'get_tweet_info_from_db', return_value=mock_tweet_info):
            with patch('fetcher.refetch_manager.sync_playwright') as mock_playwright:
                with patch('fetcher.refetch_manager.fetcher_parsers.extract_tweet_with_media_monitoring') as mock_extract:
                    with patch('fetcher.refetch_manager.fetcher_db.update_tweet_in_database', return_value=True):
                        with patch('time.sleep'):  # Mock sleep to speed up test
                            # Setup mocks
                            mock_page = Mock()
                            mock_context = Mock()
                            mock_browser = Mock()
                            mock_playwright.return_value.__enter__.return_value = Mock()
                            refetch_manager.session_manager.create_browser_context = Mock(
                                return_value=(mock_browser, mock_context, mock_page)
                            )
                            refetch_manager.session_manager.cleanup_session = Mock()
                            refetch_manager.scroller.delay = Mock()
                            
                            # First attempt returns None (extraction failed), second succeeds
                            mock_extract.side_effect = [
                                None,  # First attempt fails
                                {'tweet_id': '1983635249646203199', 'content': 'Test'}  # Second succeeds
                            ]
                            
                            # Execute
                            result = refetch_manager.refetch_single_tweet('1983635249646203199', max_retries=2)
                            
                            # Verify - should succeed after retry
                            assert result is True
                            assert mock_extract.call_count == 2
                            assert refetch_manager.session_manager.cleanup_session.call_count == 2
    
    def test_refetch_single_tweet_max_retries_exceeded(self, refetch_manager, mock_tweet_info):
        """Test that refetch fails after max retries exceeded."""
        with patch.object(refetch_manager, 'get_tweet_info_from_db', return_value=mock_tweet_info):
            with patch('fetcher.refetch_manager.sync_playwright') as mock_playwright:
                with patch('fetcher.refetch_manager.fetcher_parsers.extract_tweet_with_media_monitoring') as mock_extract:
                    with patch('time.sleep'):  # Mock sleep to speed up test
                        # Setup mocks
                        mock_page = Mock()
                        mock_context = Mock()
                        mock_browser = Mock()
                        mock_playwright.return_value.__enter__.return_value = Mock()
                        refetch_manager.session_manager.create_browser_context = Mock(
                            return_value=(mock_browser, mock_context, mock_page)
                        )
                        refetch_manager.session_manager.cleanup_session = Mock()
                        refetch_manager.scroller.delay = Mock()
                        
                        # All attempts return None (extraction always fails)
                        mock_extract.return_value = None
                        
                        # Execute with only 2 retries
                        result = refetch_manager.refetch_single_tweet('1983635249646203199', max_retries=2)
                        
                        # Verify - should fail after max retries
                        assert result is False
                        assert mock_extract.call_count == 2
                        assert refetch_manager.session_manager.cleanup_session.call_count == 2
    
    def test_refetch_single_tweet_not_found_in_db(self, refetch_manager):
        """Test behavior when tweet not found in database."""
        with patch.object(refetch_manager, 'get_tweet_info_from_db', return_value=(None, None)):
            result = refetch_manager.refetch_single_tweet('9999999999999999999')
            assert result is False
    
    def test_refetch_single_tweet_database_error(self, refetch_manager):
        """Test behavior when database error occurs."""
        with patch.object(refetch_manager, 'get_tweet_info_from_db', side_effect=Exception("Database error")):
            result = refetch_manager.refetch_single_tweet('1983635249646203199')
            assert result is False
    
    def test_exponential_backoff_timing(self, refetch_manager, mock_tweet_info):
        """Test that exponential backoff uses correct delays."""
        with patch.object(refetch_manager, 'get_tweet_info_from_db', return_value=mock_tweet_info):
            with patch('fetcher.refetch_manager.sync_playwright') as mock_playwright:
                with patch('fetcher.refetch_manager.fetcher_parsers.extract_tweet_with_media_monitoring') as mock_extract:
                    with patch('time.sleep') as mock_sleep:
                        # Setup mocks
                        mock_page = Mock()
                        mock_context = Mock()
                        mock_browser = Mock()
                        mock_playwright.return_value.__enter__.return_value = Mock()
                        refetch_manager.session_manager.create_browser_context = Mock(
                            return_value=(mock_browser, mock_context, mock_page)
                        )
                        refetch_manager.session_manager.cleanup_session = Mock()
                        refetch_manager.scroller.delay = Mock()
                        
                        # All attempts fail
                        mock_extract.return_value = None
                        
                        # Execute with 3 retries
                        result = refetch_manager.refetch_single_tweet('1983635249646203199', max_retries=3)
                        
                        # Verify exponential backoff: 2^1=2, 2^2=4 seconds (no sleep after last attempt)
                        assert result is False
                        assert mock_sleep.call_count == 2  # Only 2 sleeps for 3 attempts (no sleep after last)
                        mock_sleep.assert_any_call(2)  # First retry delay
                        mock_sleep.assert_any_call(4)  # Second retry delay


class TestRefetchAccountAll:
    """Test account refetch functionality."""
    
    @pytest.fixture
    def refetch_manager(self):
        """Create RefetchManager instance for testing."""
        return RefetchManager()
    
    def test_refetch_account_all_success(self, refetch_manager):
        """Test successful account refetch."""
        with patch('fetcher.refetch_manager.fetcher_db.delete_account_data') as mock_delete:
            with patch('fetcher.refetch_manager.sync_playwright') as mock_playwright:
                with patch('fetcher.fetch_tweets.run_fetch_session', return_value=(10, 1)):
                    mock_delete.return_value = {'tweets': 5, 'analyses': 3}
                    
                    result = refetch_manager.refetch_account_all('vitoquiles', max_tweets=10)
                    
                    assert result is True
                    mock_delete.assert_called_once_with('vitoquiles')
    
    def test_refetch_account_all_no_tweets_fetched(self, refetch_manager):
        """Test account refetch when no tweets are fetched."""
        with patch('fetcher.refetch_manager.fetcher_db.delete_account_data') as mock_delete:
            with patch('fetcher.refetch_manager.sync_playwright'):
                with patch('fetcher.fetch_tweets.run_fetch_session', return_value=(0, 1)):
                    mock_delete.return_value = {'tweets': 5, 'analyses': 3}
                    
                    result = refetch_manager.refetch_account_all('vitoquiles')
                    
                    assert result is False
    
    def test_refetch_account_all_removes_at_symbol(self, refetch_manager):
        """Test that @ symbol is removed from username."""
        with patch('fetcher.refetch_manager.fetcher_db.delete_account_data') as mock_delete:
            with patch('fetcher.refetch_manager.sync_playwright'):
                with patch('fetcher.fetch_tweets.run_fetch_session', return_value=(10, 1)):
                    mock_delete.return_value = {'tweets': 5, 'analyses': 3}
                    
                    result = refetch_manager.refetch_account_all('@vitoquiles', max_tweets=10)
                    
                    assert result is True
                    mock_delete.assert_called_once_with('vitoquiles')  # @ removed
