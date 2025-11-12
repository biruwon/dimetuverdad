#!/bin/bash

# Script to re-analyze users with 20 or fewer posts
# Excludes test_user and handles timeouts with investigation

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOG_FILE="logs/reanalyze_small_accounts_$(date +%Y%m%d_%H%M%S).log"
TIMEOUT_LOG="logs/timeout_investigation_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $*" | tee -a "$LOG_FILE"
}

# Error logging function
error_log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# Timeout investigation function
investigate_timeout() {
    local username="$1"
    local tweet_id="$2"

    log "🔍 Investigating timeout for user $username, tweet $tweet_id"

    # Get tweet content and media info
    local tweet_info=$(sqlite3 accounts.db "
        SELECT content, media_links, LENGTH(content) as content_length
        FROM tweets
        WHERE username = '$username' AND tweet_id = '$tweet_id';
    ")

    local content=$(echo "$tweet_info" | cut -d'|' -f1)
    local media_links=$(echo "$tweet_info" | cut -d'|' -f2)
    local content_length=$(echo "$tweet_info" | cut -d'|' -f3)

    echo "=== TIMEOUT INVESTIGATION ===" >> "$TIMEOUT_LOG"
    echo "User: $username" >> "$TIMEOUT_LOG"
    echo "Tweet ID: $tweet_id" >> "$TIMEOUT_LOG"
    echo "Content Length: $content_length characters" >> "$TIMEOUT_LOG"
    echo "Media Links: ${media_links:-None}" >> "$TIMEOUT_LOG"
    echo "Content Preview: ${content:0:200}..." >> "$TIMEOUT_LOG"
    echo "Timestamp: $(date)" >> "$TIMEOUT_LOG"
    echo "" >> "$TIMEOUT_LOG"

    log "📝 Timeout details logged to $TIMEOUT_LOG"
}

# Timeout investigation from analysis output
investigate_timeout_from_output() {
    local username="$1"
    local analysis_output="$2"

    log "🔍 Investigating timeout from analysis output for $username"

    echo "=== TIMEOUT INVESTIGATION FROM OUTPUT ===" >> "$TIMEOUT_LOG"
    echo "User: $username" >> "$TIMEOUT_LOG"
    echo "Timestamp: $(date)" >> "$TIMEOUT_LOG"
    echo "Analysis Output (last 50 lines):" >> "$TIMEOUT_LOG"
    echo "$analysis_output" | tail -50 >> "$TIMEOUT_LOG"
    echo "" >> "$TIMEOUT_LOG"

    log "📝 Timeout details from output logged to $TIMEOUT_LOG"
}

# Function to analyze a single user
analyze_user() {
    local username="$1"
    local post_count="$2"

    log "🔄 Starting analysis for $username ($post_count posts)"

    # Run analysis without external timeout (rely on internal analysis timeouts)
    local start_time=$(date +%s)
    local analysis_output

    log "⏰ Starting analysis for $username (no external timeout)"

    # Run the analysis
    if analysis_output=$(./run_in_venv.sh analyze-twitter --username "$username" --force-reanalyze --verbose 2>&1); then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        # Check if analysis completed successfully
        if echo "$analysis_output" | grep -q "Analysis Complete"; then
            log "✅ $username analysis completed successfully in ${duration}s"
            echo "$analysis_output" | grep -E "(Analysis Complete|Tweets processed|Successful analyses|Failed analyses)" | while read -r line; do
                log "   $line"
            done
            return 0
        else
            log "⚠️  $username analysis finished but may have issues in ${duration}s"
            return 1
        fi
    else
        local exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        log "❌ $username analysis failed with exit code $exit_code after ${duration}s"

        # Log the last part of the output for debugging
        echo "=== LAST OUTPUT FOR $username ===" >> "$LOG_FILE"
        echo "$analysis_output" | tail -20 >> "$LOG_FILE"
        echo "=== END OUTPUT ===" >> "$LOG_FILE"

        # Try to identify potential timeout issues
        if echo "$analysis_output" | grep -q "timeout\|Timeout\|timed out"; then
            log "⏰ Analysis appears to have timed out internally"
            investigate_timeout_from_output "$username" "$analysis_output"
        fi

        return 1
    fi
}

# Main script
log "🚀 Starting re-analysis of users with 20 or fewer posts"
log "📁 Log file: $LOG_FILE"
log "🔍 Timeout investigation log: $TIMEOUT_LOG"

# Create logs directory if it doesn't exist
mkdir -p logs

# Get list of users with 20 or fewer posts (excluding test_user)
log "📊 Finding users with 20 or fewer posts..."

USERS_QUERY="SELECT username, COUNT(*) as post_count FROM tweets WHERE username != 'test_user' GROUP BY username HAVING post_count <= 20 ORDER BY post_count DESC;"

USER_LIST=$(sqlite3 accounts.db "$USERS_QUERY")

if [ -z "$USER_LIST" ]; then
    log "❌ No users found with 20 or fewer posts"
    exit 1
fi

log "📋 Found users to analyze:"
echo "$USER_LIST" | while IFS='|' read -r username post_count; do
    log "   $username: $post_count posts"
done

TOTAL_USERS=$(echo "$USER_LIST" | wc -l)
log "🎯 Will analyze $TOTAL_USERS users one by one"

# Analyze each user
SUCCESSFUL=0
FAILED=0

# Process users one by one (avoid subshell issues with while loops)
while IFS='|' read -r username post_count; do
    if [ -z "$username" ]; then
        continue
    fi

    log "🎯 Processing user: $username ($post_count posts)"

    if analyze_user "$username" "$post_count"; then
        ((SUCCESSFUL++))
    else
        ((FAILED++))
    fi

    log "📊 Progress: $((SUCCESSFUL + FAILED))/$TOTAL_USERS users processed"
    echo "" >> "$LOG_FILE"
done <<< "$USER_LIST"

# Final summary
log "🎉 Analysis complete!"
log "📊 Summary:"
log "   ✅ Successful: $SUCCESSFUL"
log "   ❌ Failed: $FAILED"
log "📁 Full logs available at: $LOG_FILE"

echo ""
echo "${GREEN}🎉 Analysis complete!${NC}"
echo "${BLUE}📁 Logs: $LOG_FILE${NC}"</content>
<parameter name="filePath">/Users/antonio/projects/bulos/dimetuverdad/reanalyze_small_accounts.sh