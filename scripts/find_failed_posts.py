#!/usr/bin/env python3
"""
Script to identify and display posts that failed analysis (timed out or had errors).

Usage:
    python scripts/find_failed_posts.py [--username USERNAME] [--limit LIMIT]
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from analyzer.repository import ContentAnalysisRepository

def find_failed_posts(username=None, limit=50):
    """Find and display posts that failed analysis."""
    repo = ContentAnalysisRepository()

    print("🔍 Finding posts that failed analysis...")
    print("=" * 60)

    # Get failed analyses
    failed_posts = repo.get_analyses_by_category("ERROR", limit=limit)

    if not failed_posts:
        print("✅ No failed posts found!")
        return

    print(f"❌ Found {len(failed_posts)} failed posts:")
    print()

    for i, post in enumerate(failed_posts, 1):
        print(f"{i:2d}. Tweet ID: {post.post_id}")
        print(f"    👤 User: @{post.author_username}")
        print(f"    📅 Time: {post.analysis_timestamp}")
        print(f"    💥 Error: {post.local_explanation}")

        # Show content preview (first 100 chars)
        content_preview = post.post_content[:100] + "..." if len(post.post_content) > 100 else post.post_content
        print(f"    📝 Content: {content_preview}")

        if post.media_urls:
            print(f"    🖼️  Media: {len(post.media_urls)} files")

        print()

    print("=" * 60)
    print(f"💡 To reanalyze failed posts, run:")
    print(f"   ./run_in_venv.sh analyze-twitter --force-reanalyze --username {username or 'USERNAME'}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find posts that failed analysis")
    parser.add_argument('--username', '-u', help='Filter by username')
    parser.add_argument('--limit', '-l', type=int, default=50, help='Maximum number of failed posts to show')

    args = parser.parse_args()

    find_failed_posts(username=args.username, limit=args.limit)