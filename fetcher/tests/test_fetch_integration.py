#!/usr/bin/env python3
"""
Fetch Integration Tests
Tests that require external dependencies like Twitter/X API access.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fetcher import fetch_tweets
from playwright.sync_api import sync_playwright


class TestFetchIntegration:
    """Integration tests for fetch functionality requiring external services."""

    def run_live_fetch_test(self) -> dict:
        """Run refetch integration test (requires Twitter credentials)."""
        print("🌐 TESTING LIVE FETCH INTEGRATION")
        print("=" * 60)
        print("⚡ Running live fetch test (requires Twitter/X credentials)...")

        # Check if we have Twitter credentials
        twitter_username = os.getenv('X_USERNAME')
        twitter_password = os.getenv('X_PASSWORD')

        if not twitter_username or not twitter_password:
            print("⚠️  Skipping live fetch test - Twitter credentials not found")
            print("   Set X_USERNAME and X_PASSWORD environment variables to enable")
            print("   Optional: Set X_EMAIL_OR_PHONE for additional login verification")
            return {
                'passed': 0,
                'failed': 0,
                'skipped': 1,
                'results': [{
                    'test_id': 'live_fetch',
                    'success': None,  # Skipped
                    'description': 'Skipped - Twitter credentials not configured'
                }]
            }

        try:
            # Use a specific tweet ID for deterministic refetch testing
            test_tweet_id = '1977734268571791494'  # Known tweet from vox_es

            # Use subprocess to run the fetch_tweets.py script with --refetch flag
            import subprocess
            import sys

            print(f"🔄 Testing refetch of tweet: {test_tweet_id}")

            # Run the refetch command
            result = subprocess.run([
                sys.executable, '/Users/antonio/projects/bulos/dimetuverdad/fetcher/fetch_tweets.py',
                '--refetch', test_tweet_id
            ], capture_output=True, text=True, cwd='/Users/antonio/projects/bulos/dimetuverdad')

            # Check if the command succeeded
            if result.returncode == 0:
                print(f"✅ Refetch test passed - successfully refetched tweet {test_tweet_id}")
                return {
                    'passed': 1,
                    'failed': 0,
                    'results': [{
                        'test_id': 'refetch_test',
                        'success': True,
                        'description': f'Successfully refetched tweet {test_tweet_id}'
                    }]
                }
            else:
                print(f"❌ Refetch test failed - command exited with code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {
                    'passed': 0,
                    'failed': 1,
                    'results': [{
                        'test_id': 'refetch_test',
                        'success': False,
                        'description': f'Failed to refetch tweet {test_tweet_id}: {result.stderr.strip()}'
                    }]
                }

        except Exception as e:
            print(f"❌ Refetch test failed: {str(e)}")
            return {
                'passed': 0,
                'failed': 1,
                'results': [{
                    'test_id': 'refetch_test',
                    'success': False,
                    'description': f'Refetch error: {str(e)}'
                }]
            }


def main():
    parser = argparse.ArgumentParser(description='Fetch Integration Tests')
    parser.add_argument('--user', help='Override the default test user (default: vox_es) - not used in refetch test')

    args = parser.parse_args()

    test_suite = TestFetchIntegration()

    # Run the refetch test directly (uses hardcoded tweet ID for deterministic testing)
    results = test_suite.run_live_fetch_test()

    # Print summary
    total_passed = results.get('passed', 0)
    total_failed = results.get('failed', 0)
    total_skipped = results.get('skipped', 0)
    total_tests = total_passed + total_failed + total_skipped
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"⏭️  Skipped: {total_skipped}")
    print(f"📈 Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")

    if total_failed > 0:
        print(f"⚠️  {total_failed} tests failed")
    elif total_skipped > 0:
        print(f"ℹ️  {total_skipped} tests skipped (credentials not configured)")
    else:
        print("🎉 All tests passed!")


if __name__ == "__main__":
    main()