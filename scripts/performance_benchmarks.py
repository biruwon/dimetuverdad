#!/usrimport time
import os
import sqlite3
from datetime import datetime
import json
import sys
from pathlib import Path

# Import utility modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import paths
"""
Performance Benchmarks Script for dimetuverdad
Establishes baseline performance metrics for the analysis system.
"""

import time
import os
import sqlite3
from datetime import datetime
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from analyzer.analyze_twitter import Analyzer

class PerformanceBenchmarks:
    """Run performance benchmarks for the dimetuverdad system."""

    def __init__(self, db_path="accounts.db"):
        self.db_path = db_path
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'benchmarks': {}
        }

    def get_system_info(self):
        """Get basic system information."""
        return {
            'python_version': sys.version,
            'platform': sys.platform
        }

    def get_database_stats(self):
        """Get current database statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get table counts
            stats = {}
            tables = ['tweets', 'content_analyses', 'accounts']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f'{table}_count'] = cursor.fetchone()[0]

            # Get database file size
            stats['db_file_size_mb'] = round(os.path.getsize(self.db_path) / (1024**2), 2)

            conn.close()
            return stats
        except Exception as e:
            return {'error': str(e)}

    def benchmark_database_performance(self):
        """Benchmark database query performance."""
        print("📊 Benchmarking database performance...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Test common queries
            benchmarks = {}

            # Query 1: Get recent analyses
            start_time = time.time()
            cursor.execute("""
                SELECT ca.tweet_id, ca.category, ca.analysis_timestamp, t.username
                FROM content_analyses ca
                JOIN tweets t ON ca.tweet_id = t.tweet_id
                ORDER BY ca.analysis_timestamp DESC
                LIMIT 100
            """)
            results = cursor.fetchall()
            benchmarks['recent_analyses_query_ms'] = round((time.time() - start_time) * 1000, 2)

            # Query 2: Category distribution
            start_time = time.time()
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM content_analyses
                GROUP BY category
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            benchmarks['category_distribution_query_ms'] = round((time.time() - start_time) * 1000, 2)

            # Query 3: User-specific analysis
            start_time = time.time()
            cursor.execute("""
                SELECT username, COUNT(*) as analysis_count
                FROM content_analyses
                GROUP BY username
                ORDER BY analysis_count DESC
                LIMIT 10
            """)
            results = cursor.fetchall()
            benchmarks['user_analysis_query_ms'] = round((time.time() - start_time) * 1000, 2)

            # Query 4: Tweet search performance
            start_time = time.time()
            cursor.execute("""
                SELECT tweet_id, content, username
                FROM tweets
                WHERE content LIKE '%disinformation%'
                LIMIT 50
            """)
            results = cursor.fetchall()
            benchmarks['content_search_query_ms'] = round((time.time() - start_time) * 1000, 2)

            conn.close()

            return benchmarks

        except Exception as e:
            return {'error': str(e)}

    def benchmark_analysis_throughput(self, sample_size=5):
        """Benchmark analysis throughput using existing tweets."""
        print(f"📊 Benchmarking analysis throughput with {sample_size} tweets...")

        try:
            # Get sample tweets from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tweet_id, content, username
                FROM tweets
                WHERE content IS NOT NULL AND length(content) > 10
                ORDER BY RANDOM()
                LIMIT ?
            """, (sample_size,))

            sample_tweets = cursor.fetchall()
            conn.close()

            if not sample_tweets:
                return {'error': 'No tweets found for benchmarking'}

            # Initialize analyzer
            analyzer = Analyzer(model_priority='fast')  # Use fast mode for benchmarking

            # Run benchmark
            start_time = time.time()

            results = []
            for tweet_id, content, username in sample_tweets:
                tweet_start = time.time()

                # Analyze tweet - need to provide all required parameters
                tweet_url = f"https://twitter.com/{username}/status/{tweet_id}"
                analysis_result = analyzer.analyze_content(tweet_id, tweet_url, username, content)

                tweet_time = time.time() - tweet_start
                results.append({
                    'tweet_id': tweet_id,
                    'analysis_time_seconds': round(tweet_time, 3),
                    'category': analysis_result.category if analysis_result else 'error'
                })

            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_tweet = total_time / len(results)

            return {
                'sample_size': len(results),
                'total_time_seconds': round(total_time, 2),
                'avg_time_per_tweet_seconds': round(avg_time_per_tweet, 3),
                'throughput_tweets_per_minute': round(60 / avg_time_per_tweet, 2),
                'results': results
            }

        except Exception as e:
            return {'error': str(e)}

    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("🚀 Running dimetuverdad Performance Benchmarks")
        print("=" * 50)

        # Database stats
        self.results['database_stats'] = self.get_database_stats()

        # Analysis throughput
        self.results['benchmarks']['analysis_throughput'] = self.benchmark_analysis_throughput()

        # Database performance
        self.results['benchmarks']['database_performance'] = self.benchmark_database_performance()

        return self.results

    def save_results(self, output_file=None):
        """Save benchmark results to file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_benchmarks_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"✅ Benchmark results saved to: {output_file}")
        return output_file

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n📊 Performance Benchmark Summary")
        print("=" * 40)

        # Database stats
        db_stats = self.results.get('database_stats', {})
        print(f"Database: {db_stats.get('tweets_count', 0)} tweets, {db_stats.get('content_analyses_count', 0)} analyses")
        print(f"Database Size: {db_stats.get('db_file_size_mb', 0)} MB")

        # Analysis throughput
        throughput = self.results.get('benchmarks', {}).get('analysis_throughput', {})
        if 'avg_time_per_tweet_seconds' in throughput:
            print(f"Analysis Speed: {throughput['avg_time_per_tweet_seconds']}s per tweet ({throughput['throughput_tweets_per_minute']} tweets/min)")

        # Database performance
        db_perf = self.results.get('benchmarks', {}).get('database_performance', {})
        if 'recent_analyses_query_ms' in db_perf:
            print(f"DB Query Performance:")
            print(f"  Recent analyses: {db_perf['recent_analyses_query_ms']}ms")
            print(f"  Category distribution: {db_perf['category_distribution_query_ms']}ms")
            print(f"  User analysis: {db_perf['user_analysis_query_ms']}ms")

def main():
    """Main function to run benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description='Run dimetuverdad performance benchmarks')
    parser.add_argument('--output', '-o', help='Output file for results (default: auto-generated)')
    parser.add_argument('--sample-size', '-s', type=int, default=5, help='Number of tweets to sample for analysis benchmark')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode, only save results')

    args = parser.parse_args()

    # Run benchmarks
    benchmarks = PerformanceBenchmarks()
    results = benchmarks.run_all_benchmarks()

    # Save results
    output_file = benchmarks.save_results(args.output)

    # Print summary unless quiet
    if not args.quiet:
        benchmarks.print_summary()

    print(f"\n✅ Benchmarks completed! Results saved to {output_file}")

if __name__ == "__main__":
    main()