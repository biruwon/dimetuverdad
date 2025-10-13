#!/usr/bin/env python3
"""
Database initialization script - Creates clean database from scratch.
Run this to set up a fresh dimetuverdad database with proper schema.
"""

import sqlite3
import os
import sys
from datetime import datetime
from pathlib import Path

# Import utility modules
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import paths
from utils.database import get_db_connection

DB_PATH = paths.get_db_path()

def drop_existing_database():
    """Remove existing database file if it exists."""
    if os.path.exists(DB_PATH):
        print(f"🗑️  Removing existing database: {DB_PATH}")
        os.remove(DB_PATH)
    else:
        print(f"📋 No existing database found")

def create_fresh_database_schema(db_path: str = None):
    """Create a clean database schema for the specified path."""
    target_path = db_path or DB_PATH

    print(f"🏗️  Creating fresh database schema at {target_path}...")

    # Create connection directly to the target database
    conn = sqlite3.connect(target_path, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    try:
        # Core accounts table (multi-platform support)
        print("  📝 Creating accounts table...")
        c.execute('''
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                platform TEXT DEFAULT 'twitter',  -- Multi-platform support
                profile_pic_url TEXT,
                profile_pic_updated TIMESTAMP,
                last_scraped TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Core tweets table (simplified)
        print("  📝 Creating tweets table...")
        c.execute('''
            CREATE TABLE tweets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tweet_id TEXT UNIQUE NOT NULL,
                tweet_url TEXT NOT NULL,
                username TEXT NOT NULL,
                content TEXT NOT NULL,
                
                -- Post classification (simplified)
                post_type TEXT DEFAULT 'original', -- original, repost_own, repost_other, repost_reply, thread
                is_pinned INTEGER DEFAULT 0,
                
                -- RT / embedded/referenced content data (only when needed)
                original_author TEXT,     -- For reposts or referenced tweets
                original_tweet_id TEXT,   -- For reposts or referenced tweets
                original_content TEXT,    -- For reposts or referenced tweets (if different)
                reply_to_username TEXT,   -- For replies
                
                -- Media and content
                media_links TEXT,         -- Comma-separated URLs
                media_count INTEGER DEFAULT 0,
                hashtags TEXT,           -- JSON array
                mentions TEXT,           -- JSON array
                external_links TEXT,     -- JSON array
                
                -- Basic engagement (optional)
                engagement_likes INTEGER DEFAULT 0,
                engagement_retweets INTEGER DEFAULT 0,
                engagement_replies INTEGER DEFAULT 0,
                
                -- Essential status tracking
                is_deleted INTEGER DEFAULT 0,
                is_edited INTEGER DEFAULT 0,
                
                -- RT optimization
                rt_original_analyzed INTEGER DEFAULT 0, -- Avoid duplicate analysis
                
                -- Timestamps (minimal)
                tweet_timestamp TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (username) REFERENCES accounts (username)
            )
        ''')
        
        # Content analysis results (platform-agnostic)
        print("  📝 Creating content_analyses table...")
        c.execute('''
            CREATE TABLE content_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,           -- Platform-agnostic post identifier
                post_url TEXT,                   -- Platform-agnostic post URL
                author_username TEXT,            -- Platform-agnostic author identifier
                platform TEXT DEFAULT 'twitter', -- Multi-platform support
                post_content TEXT,               -- Platform-agnostic content
                category TEXT,                   -- Primary category (backward compatibility)
                categories_detected TEXT,        -- JSON array of all detected categories
                llm_explanation TEXT,
                analysis_method TEXT DEFAULT "pattern", -- "pattern", "llm", or "gemini"
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_json TEXT,
                
                -- Multi-category support
                pattern_matches TEXT,            -- JSON array of pattern matches
                topic_classification TEXT,       -- JSON topic classification data
                
                -- Media analysis (multimodal support)
                media_urls TEXT,                 -- JSON array of media URLs
                media_analysis TEXT,             -- Gemini multimodal analysis result
                media_type TEXT,                 -- "image", "video", or ""
                multimodal_analysis BOOLEAN DEFAULT FALSE, -- Whether media was analyzed
                
                -- Evidence retrieval and verification
                verification_data TEXT,          -- JSON verification data from retrieval system
                verification_confidence REAL DEFAULT 0.0, -- Confidence score for verification
                
                FOREIGN KEY (post_id) REFERENCES tweets (tweet_id),  -- Keep FK for now, will be updated
                FOREIGN KEY (author_username) REFERENCES accounts (username),  -- Keep FK for now, will be updated
                UNIQUE(post_id) -- One analysis per post
            )
        ''')
        
        # Post edits detection (renamed for clarity - tracks post content changes)
        print("  📝 Creating post_edits table...")
        c.execute('''
            CREATE TABLE post_edits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                previous_content TEXT NOT NULL,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (post_id) REFERENCES tweets (tweet_id),  -- Keep FK for now, will be updated
                UNIQUE(post_id, version_number)
            )
        ''')
        
        # User feedback table for model improvement (platform-agnostic)
        print("  📝 Creating user_feedback table...")
        c.execute('''
            CREATE TABLE user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT NOT NULL,          -- Platform-agnostic post identifier
                feedback_type TEXT NOT NULL,    -- 'correction', 'flag', 'improvement'
                original_category TEXT,
                corrected_category TEXT,
                user_comment TEXT,
                user_ip TEXT,                   -- For rate limiting and analytics
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (post_id) REFERENCES tweets (tweet_id)  -- Keep FK for now, will be updated
            )
        ''')
        
        # Platforms table for multi-platform support (hierarchical)
        print("  📝 Creating platforms table...")
        c.execute('''
            CREATE TABLE platforms (
                platform_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,  -- 'social_media', 'messenger', 'news', etc.
                name TEXT NOT NULL,
                display_name TEXT NOT NULL,
                description TEXT,
                api_base_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Populate platforms table with defaults
        print("  📝 Populating platforms table...")
        platforms_data = [
            ('twitter', 'social_media', 'Twitter', 'Twitter/X', 'Social media platform for short-form content', 'https://twitter.com'),
            ('telegram', 'messenger', 'Telegram', 'Telegram', 'Messaging platform with channels and groups', 'https://telegram.org'),
            ('news', 'news', 'News', 'News Sources', 'Newspaper and news website sources', None),
        ]
        
        c.executemany('''
            INSERT INTO platforms (platform_id, category, name, display_name, description, api_base_url)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', platforms_data)
        
        # Performance indexes
        print("  📝 Creating indexes...")
        indexes = [
            ('idx_tweets_username', 'tweets', 'username'),
            ('idx_tweets_post_type', 'tweets', 'post_type'),
            ('idx_tweets_timestamp', 'tweets', 'scraped_at'),
            ('idx_tweets_tweet_timestamp', 'tweets', 'tweet_timestamp'),
            ('idx_tweets_deleted', 'tweets', 'is_deleted'),
            ('idx_tweets_edited', 'tweets', 'is_edited'),
            ('idx_analyses_post', 'content_analyses', 'post_id'),
            ('idx_analyses_category', 'content_analyses', 'category'),
            ('idx_analyses_author', 'content_analyses', 'author_username'),
            ('idx_analyses_platform', 'content_analyses', 'platform'),
            ('idx_content_analyses_timestamp', 'content_analyses', 'analysis_timestamp'),
            ('idx_content_analyses_method', 'content_analyses', 'analysis_method'),
            ('idx_content_analyses_multimodal', 'content_analyses', 'multimodal_analysis'),
            ('idx_post_edits_post', 'post_edits', 'post_id'),
            ('idx_user_feedback_post', 'user_feedback', 'post_id'),
            ('idx_user_feedback_type', 'user_feedback', 'feedback_type'),
            ('idx_user_feedback_submitted', 'user_feedback', 'submitted_at'),
            ('idx_platforms_name', 'platforms', 'name')
        ]
        
        for idx_name, table, columns in indexes:
            c.execute(f'CREATE INDEX {idx_name} ON {table}({columns})')
            
        print(f"    ✅ Created {len(indexes)} performance indexes")
        
        conn.commit()
        print("✅ Clean database schema created successfully!")
        
        # Show summary
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row['name'] for row in c.fetchall()]
        print(f"📊 Created {len(tables)} tables: {', '.join(tables)}")

    except Exception as e:
        conn.rollback()
        print(f"❌ Database creation failed: {e}")
        raise
    finally:
        conn.close()


def create_fresh_database():
    """Create a clean database with optimized schema (legacy wrapper)."""
    return create_fresh_database_schema()


def verify_schema():
    """Verify the database schema is correct."""
    print("🔍 Verifying database schema...")
    
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        # Check tables exist
        c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row['name'] for row in c.fetchall()]
        expected_tables = ['accounts', 'content_analyses', 'platforms', 'post_edits', 'tweets', 'user_feedback']
        
        print(f"  📋 Tables found: {tables}")
        for table in expected_tables:
            if table in tables:
                print(f"    ✅ {table}")
            else:
                print(f"    ❌ {table} MISSING!")
                return False
        
        # Check key fields exist in content_analyses table
        c.execute("PRAGMA table_info(content_analyses)")
        ca_columns = [row['name'] for row in c.fetchall()]
        essential_ca_fields = [
            'post_id', 'author_username', 'platform', 'category', 'analysis_method', 'analysis_timestamp',
            'categories_detected', 'media_urls', 'media_analysis', 'media_type', 'multimodal_analysis',
            'verification_data', 'verification_confidence'
        ]
        
        print(f"  📋 Content analyses table columns: {len(ca_columns)} total")
        for field in essential_ca_fields:
            if field in ca_columns:
                print(f"    ✅ {field}")
            else:
                print(f"    ❌ {field} MISSING!")
                return False
        
        print("✅ Database schema verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ Schema verification failed: {e}")
        return False
    finally:
        conn.close()

def show_usage():
    """Show how to use this script."""
    print("""
🎯 dimetuverdad Database Initialization

This script creates a clean database from scratch with an optimized schema.

Usage:
    python init_database.py [--force]
    
Options:
    --force    Force recreation even if database exists
    
What it does:
    1. Removes existing database (if --force or no database exists)
    2. Creates clean schema with essential fields only
    3. Sets up proper indexes for performance
    4. Verifies schema correctness

After running this script, you can:
    1. Run fetch_tweets.py to collect data
    2. Run analysis scripts to analyze content
    3. Start the web interface to view results
    """)

if __name__ == "__main__":
    import sys
    
    force_recreate = "--force" in sys.argv
    show_help = "--help" in sys.argv or "-h" in sys.argv
    
    if show_help:
        show_usage()
        sys.exit(0)
    
    print("🚀 dimetuverdad Database Initialization")
    print("=" * 50)
    
    # Check if database exists
    db_exists = os.path.exists(DB_PATH)
    
    if db_exists and not force_recreate:
        print(f"⚠️  Database {DB_PATH} already exists!")
        print("Use --force to recreate it, or --help for more options.")
        print("\nTo recreate: python init_database.py --force")
        sys.exit(1)
    
    try:
        # Remove existing database
        if db_exists:
            drop_existing_database()
        
        # Create fresh schema
        create_fresh_database_schema()
        
        # Verify it worked
        if verify_schema():
            print("\n🎉 Database initialization completed successfully!")
            print("\nNext steps:")
            print("  1. Run: python fetch_tweets.py --max 10")
            print("  2. Run: python -m analyzer.analyze_twitter") 
            print("  3. Start web interface: cd web && python app.py")
        else:
            print("\n❌ Database verification failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Initialization failed: {e}")
        sys.exit(1)