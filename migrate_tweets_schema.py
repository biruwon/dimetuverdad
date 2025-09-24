#!/usr/bin/env python3
"""
Enhanced tweet database schema migration script.
Adds comprehensive fields for better tweet handling and analysis.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "accounts.db"

def migrate_tweets_schema():
    """Migrate the tweets table to support enhanced tweet data."""
    
    print("🔄 Starting tweet database schema migration...")
    
    # Backup current database
    backup_path = f"accounts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
    if os.path.exists(DB_PATH):
        print(f"📋 Creating backup: {backup_path}")
        os.system(f"cp {DB_PATH} {backup_path}")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check if the enhanced schema already exists
    c.execute("PRAGMA table_info(tweets)")
    existing_columns = [row[1] for row in c.fetchall()]
    
    # Define new columns we want to add
    new_columns = [
        ("post_type", "TEXT DEFAULT 'original'"), # original, repost_own, repost_other, reply, quote, thread
        ("is_pinned", "INTEGER DEFAULT 0"),
        ("original_author", "TEXT"),  # For reposts, the original author
        ("original_tweet_id", "TEXT"),  # For reposts, the original tweet ID
        ("original_content", "TEXT"),  # For reposts, the original tweet content
        ("reply_to_username", "TEXT"),  # For replies, who we're replying to
        ("reply_to_tweet_id", "TEXT"),  # For replies, which tweet we're replying to
        ("reply_to_content", "TEXT"),  # For replies, the content being replied to
        ("thread_position", "INTEGER DEFAULT 0"),  # Position in thread (0 = main tweet)
        ("thread_root_id", "TEXT"),  # Root tweet ID for threads
        ("engagement_retweets", "INTEGER DEFAULT 0"),
        ("engagement_likes", "INTEGER DEFAULT 0"), 
        ("engagement_replies", "INTEGER DEFAULT 0"),
        ("engagement_views", "INTEGER DEFAULT 0"),
        ("media_count", "INTEGER DEFAULT 0"),
        ("media_types", "TEXT"),  # JSON array of media types: ["image", "video", "gif"]
        ("has_external_link", "INTEGER DEFAULT 0"),
        ("external_links", "TEXT"),  # JSON array of external links
        ("hashtags", "TEXT"),  # JSON array of hashtags
        ("mentions", "TEXT"),  # JSON array of user mentions
        ("tweet_timestamp", "TEXT"),  # Original tweet timestamp from X
        ("scrape_timestamp", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),  # When we scraped it
        ("last_updated", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),  # Last time we updated this record
    ]
    
    # Add missing columns
    columns_added = 0
    for col_name, col_definition in new_columns:
        if col_name not in existing_columns:
            try:
                c.execute(f"ALTER TABLE tweets ADD COLUMN {col_name} {col_definition}")
                print(f"  ✅ Added column: {col_name}")
                columns_added += 1
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    print(f"  ⚠️ Error adding {col_name}: {e}")
    
    # Update existing records to use new schema defaults
    if columns_added > 0:
        print("🔄 Updating existing records with default values...")
        
        # Set post_type based on existing flags
        c.execute("""
            UPDATE tweets 
            SET post_type = CASE
                WHEN is_repost = 1 THEN 'repost_other'
                WHEN is_comment = 1 THEN 'reply'  
                WHEN parent_tweet_id IS NOT NULL THEN 'quote'
                ELSE 'original'
            END
            WHERE post_type = 'original'
        """)
        
        # Set scrape_timestamp to created_at for existing records
        c.execute("""
            UPDATE tweets 
            SET scrape_timestamp = created_at,
                last_updated = CURRENT_TIMESTAMP
            WHERE scrape_timestamp IS NULL
        """)
        
        # Count media links for existing records
        c.execute("""
            UPDATE tweets 
            SET media_count = CASE
                WHEN media_links IS NULL OR media_links = '' THEN 0
                ELSE (LENGTH(media_links) - LENGTH(REPLACE(media_links, ',', '')) + 1)
            END
            WHERE media_count = 0 AND media_links IS NOT NULL
        """)
    
    # Create indexes for better performance
    indexes = [
        ("idx_tweets_post_type", "CREATE INDEX IF NOT EXISTS idx_tweets_post_type ON tweets(post_type)"),
        ("idx_tweets_username", "CREATE INDEX IF NOT EXISTS idx_tweets_username ON tweets(username)"),
        ("idx_tweets_timestamp", "CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets(scrape_timestamp)"),
        ("idx_tweets_pinned", "CREATE INDEX IF NOT EXISTS idx_tweets_pinned ON tweets(is_pinned)"),
        ("idx_tweets_engagement", "CREATE INDEX IF NOT EXISTS idx_tweets_engagement ON tweets(engagement_likes, engagement_retweets)"),
    ]
    
    for idx_name, idx_sql in indexes:
        try:
            c.execute(idx_sql)
            print(f"  ✅ Created index: {idx_name}")
        except sqlite3.OperationalError:
            pass  # Index already exists
    
    conn.commit()
    
    # Show final schema
    print("\n📊 Final tweets table schema:")
    c.execute("PRAGMA table_info(tweets)")
    for row in c.fetchall():
        print(f"  {row[1]} {row[2]} {row[3] if row[3] else ''} {row[4] if row[4] else ''}")
    
    # Show statistics
    c.execute("SELECT COUNT(*) FROM tweets")
    total_tweets = c.fetchone()[0]
    
    c.execute("SELECT post_type, COUNT(*) FROM tweets GROUP BY post_type")
    post_type_counts = c.fetchall()
    
    print(f"\n📈 Database statistics:")
    print(f"  Total tweets: {total_tweets}")
    print("  Post type distribution:")
    for post_type, count in post_type_counts:
        print(f"    {post_type}: {count}")
    
    conn.close()
    print(f"\n✅ Schema migration completed! Added {columns_added} new columns.")
    print(f"💾 Backup saved as: {backup_path}")

def reset_tweets_table():
    """Reset the tweets table completely - use with caution!"""
    
    response = input("⚠️ This will DELETE ALL TWEETS from the database. Are you sure? (type 'DELETE' to confirm): ")
    if response != "DELETE":
        print("❌ Operation cancelled.")
        return
    
    print("🗑️ Resetting tweets table...")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Drop the table entirely
    c.execute("DROP TABLE IF EXISTS tweets")
    
    # Recreate with new enhanced schema
    c.execute('''
        CREATE TABLE tweets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tweet_id TEXT UNIQUE,
            tweet_url TEXT,
            username TEXT,
            content TEXT,
            
            -- Media and links
            media_links TEXT,  -- Comma-separated URLs (legacy)
            media_count INTEGER DEFAULT 0,
            media_types TEXT,  -- JSON array of media types
            has_external_link INTEGER DEFAULT 0,
            external_links TEXT,  -- JSON array of external links
            
            -- Content analysis
            hashtags TEXT,  -- JSON array of hashtags
            mentions TEXT,  -- JSON array of user mentions
            
            -- Post classification
            post_type TEXT DEFAULT 'original',  -- original, repost_own, repost_other, reply, quote, thread
            is_pinned INTEGER DEFAULT 0,
            
            -- Legacy flags (kept for compatibility)
            is_repost INTEGER DEFAULT 0,
            is_like INTEGER DEFAULT 0, 
            is_comment INTEGER DEFAULT 0,
            parent_tweet_id TEXT,
            
            -- Enhanced relationship data
            original_author TEXT,  -- For reposts
            original_tweet_id TEXT,  -- For reposts
            original_content TEXT,  -- For reposts
            reply_to_username TEXT,  -- For replies
            reply_to_tweet_id TEXT,  -- For replies  
            reply_to_content TEXT,  -- For replies
            thread_position INTEGER DEFAULT 0,  -- Position in thread
            thread_root_id TEXT,  -- Root tweet ID for threads
            
            -- Engagement metrics
            engagement_retweets INTEGER DEFAULT 0,
            engagement_likes INTEGER DEFAULT 0,
            engagement_replies INTEGER DEFAULT 0, 
            engagement_views INTEGER DEFAULT 0,
            
            -- Timestamps
            tweet_timestamp TEXT,  -- Original tweet timestamp
            scrape_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Legacy
        )
    ''')
    
    # Create indexes
    indexes = [
        "CREATE INDEX idx_tweets_post_type ON tweets(post_type)",
        "CREATE INDEX idx_tweets_username ON tweets(username)", 
        "CREATE INDEX idx_tweets_timestamp ON tweets(scrape_timestamp)",
        "CREATE INDEX idx_tweets_pinned ON tweets(is_pinned)",
        "CREATE INDEX idx_tweets_engagement ON tweets(engagement_likes, engagement_retweets)",
    ]
    
    for idx_sql in indexes:
        c.execute(idx_sql)
    
    conn.commit()
    conn.close()
    
    print("✅ Tweets table reset with enhanced schema!")

if __name__ == "__main__":
    import sys
    
    # Default behavior: reset the table with new schema
    reset_tweets_table()