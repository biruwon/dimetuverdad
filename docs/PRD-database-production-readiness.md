# Development Plan: Database Multi-Source Support & Optimization

**Development Plan**  
**Version:** 1.0  
**Date:** October 11, 2025  
**Status:** Development Phase  

---

## Executive Summary

This development plan outlines the steps to optimize the dimetuverdad database for multi-source support while maintaining SQLite. The focus is on enabling content analysis from multiple social media platforms while keeping the current development workflow.

## Current State & Goals

### Current Architecture
- **Database**: SQLite (accounts.db)
- **Schema**: Twitter-specific tables (tweets, content_analyses, edit_history)
- **Environment**: Single development database
- **Backup**: None

### Development Goals
1. **Multi-Platform Schema**: Support Twitter + future platforms (Facebook, Instagram, etc.)
2. **Platform-Agnostic Analysis**: Unified content analysis across platforms
3. **Development Workflow**: Maintain fast iteration with proper test isolation
4. **Local Backup**: Simple backup system for development safety

## Implementation Plan

### Phase 1: Schema Evolution & Multi-Platform Foundation

#### 1.1 Backup Before Changes
Run `./scripts/backup_db.py` to create a backup before making schema changes.

#### 1.2 Schema Redesign

**Modify accounts table (multi-platform):**
```sql
-- One account per platform (accounts get scraped at different times per platform)
accounts:
  id, platform TEXT DEFAULT 'twitter', handle, display_name, profile_url,
  last_scraped, created_at
-- Example: @Santi_ABASCAL on twitter, @Santi_ABASCAL on facebook, etc.
```

**Modify content_analyses (platform-agnostic):**
```sql
-- Remove Twitter-specific fields (exist in tweets table)
content_analyses:
  id, post_id TEXT NOT NULL,  -- References tweets.id or future posts.id
  platform TEXT DEFAULT 'twitter',
  category, llm_explanation, analysis_method,
  categories_detected, pattern_matches, topic_classification,
  media_urls, media_analysis, media_type, multimodal_analysis,
  analysis_timestamp, analysis_json
```

**Rename edit_history:**
```sql
-- Make Twitter-specific
twitter_edit_history:
  id, tweet_id, version_number, previous_content, detected_at
```

#### 1.3 Platform Registry & Sources
Add platform support tables:
```sql
platforms:
  id, name, display_name, base_url, api_endpoint
-- Pre-populate with: twitter, facebook, instagram, tiktok, youtube

sources:  -- For future multi-platform accounts/feeds
  id, platform_id, handle, display_name, profile_url, last_scraped, is_active
```

#### 1.4 Migration Script
Create `scripts/migrate_schema_v2.py`:
- Add `platform` column to `accounts` and `content_analyses`
- Rename `edit_history` to `twitter_edit_history`
- Update `content_analyses.post_id` to reference `tweets.id` instead of `tweet_id`
- Create platform/sources tables
- Migrate existing data
- Update all code references

#### 1.5 Schema Optimization
**Field Usage Audit & Cleanup:**
```sql
-- Run these queries to check utilization:
SELECT COUNT(*) FROM tweets WHERE original_tweet_id IS NOT NULL;
SELECT COUNT(*) FROM content_analyses WHERE multimodal_analysis = 1;
SELECT COUNT(*) FROM tweets WHERE hashtags IS NOT NULL;
SELECT COUNT(*) FROM tweets WHERE external_links IS NOT NULL;
```

**Index Optimization:**
```sql
CREATE INDEX idx_accounts_platform_handle ON accounts(platform, handle);
CREATE INDEX idx_content_analyses_platform ON content_analyses(platform);
CREATE INDEX idx_content_analyses_post_id ON content_analyses(post_id);
CREATE INDEX idx_content_analyses_multimodal ON content_analyses(multimodal_analysis);
CREATE INDEX idx_tweets_media_count ON tweets(media_count) WHERE media_count > 0;
```

### Phase 2: Environment Isolation

#### 2.1 Test Database Management
- **Development**: `./accounts.db` (current)
- **Testing**: `/tmp/dimetuverdad_test_*.db` (ephemeral)
- **Backup**: `./backups/` (gitignored)

#### 2.2 Database Path Configuration
Update `utils/paths.py`:
```python
def get_db_path(test_mode=False):
    if test_mode or os.environ.get('PYTEST_CURRENT_TEST'):
        return f'/tmp/dimetuverdad_test_{os.getpid()}.db'
    return './accounts.db'

def get_backup_dir():
    return './backups'
```

#### 2.3 Test Detection
Automatic test database creation:
- Detect pytest environment
- Use ephemeral databases for all tests
- Clean up after test completion

### Phase 3: Backup System

#### 3.1 Local Backup Implementation
Create `scripts/backup_db.py`:
```bash
#!/usr/bin/env python3
# Create timestamped backup of accounts.db
cp accounts.db backups/accounts_$(date +%Y%m%d_%H%M%S).db
```

#### 3.2 Backup Directory
- Add `backups/` to `.gitignore`
- Create directory structure: `./backups/`
- Manual backup command: `./scripts/backup_db.py`

## Code Changes Required

### Files to Update:
1. `scripts/init_database.py` - Add new tables, modify existing schema
2. `utils/paths.py` - Add test database detection
3. `analyzer/repository.py` - Update queries for new schema
4. `web/app.py` - Handle platform-specific displays
5. `fetch_tweets.py` - Set platform='twitter' in analyses

### New Files:
1. `scripts/migrate_schema_v2.py` - Schema migration
2. `scripts/backup_db.py` - Backup utility
3. `backups/` directory (gitignored)

## Development Workflow

### Daily Development
1. **Work on main database**: `./accounts.db`
2. **Run tests**: Automatic ephemeral databases
3. **Manual backup**: `./scripts/backup_db.py` when needed
4. **Schema changes**: Update migration scripts

### Testing Strategy
- **Unit tests**: Use ephemeral test databases
- **Integration tests**: Isolated database instances
- **No production data**: Never touch main database in tests

## Success Criteria

### Technical Milestones
- ✅ Schema supports multiple platforms
- ✅ Test databases are ephemeral and isolated
- ✅ Manual backup system functional
- ✅ No breaking changes to existing functionality
- ✅ Performance maintained with new indexes

### Development Benefits
- 🚀 Fast iteration with proper test isolation
- 🔧 Easy multi-platform extension
- 🛡️ Development safety with local backups
- 📊 Better query performance
- 🧹 Cleaner, optimized schema

---

**Document Owner**: Development Team  
**Status**: Development Plan (not PRD)  
**Next Review**: After Phase 1 completion