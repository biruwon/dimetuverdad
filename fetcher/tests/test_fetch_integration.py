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
        """Run live fetch integration test (requires Twitter credentials)."""
        print("🌐 TESTING LIVE FETCH INTEGRATION")
        print("=" * 60)
        print("⚡ Running live fetch test (requires Twitter/X credentials)...")

        try:
            user = os.getenv('LIVE_FETCH_USER', 'vox_es')

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                results = fetch_tweets.fetch_latest_tweets(page, user, max_tweets=2)
                browser.close()

                if isinstance(results, list) and len(results) <= 2:
                    print("✅ Live fetch test passed")
                    return {
                        'passed': 1,
                        'failed': 0,
                        'results': [{
                            'test_id': 'live_fetch',
                            'success': True,
                            'description': f'Successfully fetched {len(results)} posts from {user}'
                        }]
                    }
                else:
                    print("❌ Live fetch test failed - unexpected results")
                    return {
                        'passed': 0,
                        'failed': 1,
                        'results': [{
                            'test_id': 'live_fetch',
                            'success': False,
                            'description': 'Live fetch returned unexpected results'
                        }]
                    }

        except Exception as e:
            print(f"❌ Live fetch test failed: {str(e)}")
            return {
                'passed': 0,
                'failed': 1,
                'results': [{
                    'test_id': 'live_fetch',
                    'success': False,
                    'description': f'Live fetch error: {str(e)}'
                }]
            }


def main():
    parser = argparse.ArgumentParser(description='Fetch Integration Tests')
    parser.add_argument('--live-fetch', action='store_true', help='Run live fetch integration test (requires Twitter credentials)')

    args = parser.parse_args()

    test_suite = TestFetchIntegration()

    if args.live_fetch:
        results = test_suite.run_live_fetch_test()
    else:
        print("No test specified. Use --live-fetch to run live fetch integration test.")
        return

    # Print summary
    total_passed = results['passed']
    total_failed = results['failed']
    total_tests = total_passed + total_failed
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"📈 Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")

    if total_failed > 0:
        print(f"⚠️  {total_failed} tests failed")
    else:
        print("🎉 All tests passed!")


if __name__ == "__main__":
    main()