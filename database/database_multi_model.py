"""
Multi-model analysis database operations.
Handles storage and retrieval of individual model analysis results.
"""

import sqlite3
from typing import List, Dict, Optional
from datetime import datetime

def save_model_analysis(
    conn: sqlite3.Connection,
    post_id: str,
    model_name: str,
    category: str,
    explanation: str,
    processing_time: float,
    confidence_score: Optional[float] = None,
    error_message: Optional[str] = None
) -> None:
    """
    Save analysis result from a specific model.
    
    Args:
        conn: Database connection
        post_id: Post/tweet identifier
        model_name: Name of the model used (e.g., "gemma3:4b")
        category: Detected category
        explanation: Model's explanation
        processing_time: Time taken for analysis in seconds
        confidence_score: Optional confidence score
        error_message: Optional error message if analysis failed
    """
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO model_analyses 
        (post_id, model_name, category, explanation, confidence_score, 
         processing_time_seconds, error_message, analysis_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        post_id,
        model_name,
        category,
        explanation,
        confidence_score,
        processing_time,
        error_message,
        datetime.now().isoformat()
    ))
    
    conn.commit()


def get_model_analyses(conn: sqlite3.Connection, post_id: str) -> List[Dict]:
    """
    Get all model analyses for a specific post.
    
    Args:
        conn: Database connection
        post_id: Post/tweet identifier
        
    Returns:
        List of dictionaries containing model analysis results
    """
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            model_name,
            category,
            explanation,
            confidence_score,
            processing_time_seconds,
            analysis_timestamp,
            error_message
        FROM model_analyses
        WHERE post_id = ?
        ORDER BY analysis_timestamp DESC
    ''', (post_id,))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'model_name': row['model_name'],
            'category': row['category'],
            'explanation': row['explanation'],
            'confidence_score': row['confidence_score'],
            'processing_time': row['processing_time_seconds'],
            'timestamp': row['analysis_timestamp'],
            'error': row['error_message']
        })
    
    return results


def get_model_consensus(conn: sqlite3.Connection, post_id: str) -> Optional[Dict]:
    """
    Calculate consensus category from all model analyses.
    Uses majority voting - returns category agreed upon by most models.
    
    Args:
        conn: Database connection
        post_id: Post/tweet identifier
        
    Returns:
        Dictionary with consensus info or None if no analyses exist:
        {
            'category': str,           # Consensus category
            'agreement_score': float,  # Percentage agreement (0-1)
            'model_votes': dict,       # Category -> count mapping
            'total_models': int        # Total models analyzed
        }
    """
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT category, COUNT(*) as count
        FROM model_analyses
        WHERE post_id = ? AND error_message IS NULL
        GROUP BY category
        ORDER BY count DESC
    ''', (post_id,))
    
    rows = cursor.fetchall()
    
    if not rows:
        return None
    
    # Build vote counts
    votes = {row['category']: row['count'] for row in rows}
    total_models = sum(votes.values())
    
    # Get consensus (most voted category)
    consensus_category = rows[0]['category']
    consensus_count = rows[0]['count']
    
    agreement_score = consensus_count / total_models if total_models > 0 else 0.0
    
    return {
        'category': consensus_category,
        'agreement_score': agreement_score,
        'model_votes': votes,
        'total_models': total_models
    }


def get_model_performance_stats(conn: sqlite3.Connection, model_name: Optional[str] = None) -> Dict:
    """
    Get performance statistics for models.
    
    Args:
        conn: Database connection
        model_name: Optional specific model to analyze (None = all models)
        
    Returns:
        Dictionary with performance metrics:
        {
            'model_name': {
                'total_analyses': int,
                'successful': int,
                'failed': int,
                'avg_processing_time': float,
                'category_distribution': dict,
                'error_rate': float
            }
        }
    """
    cursor = conn.cursor()
    
    # Base query for all models or specific model
    where_clause = "WHERE model_name = ?" if model_name else ""
    params = (model_name,) if model_name else ()
    
    cursor.execute(f'''
        SELECT 
            model_name,
            COUNT(*) as total,
            SUM(CASE WHEN error_message IS NULL THEN 1 ELSE 0 END) as successful,
            SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as failed,
            AVG(CASE WHEN error_message IS NULL THEN processing_time_seconds ELSE NULL END) as avg_time
        FROM model_analyses
        {where_clause}
        GROUP BY model_name
    ''', params)
    
    stats = {}
    
    for row in cursor.fetchall():
        model = row['model_name']
        total = row['total']
        successful = row['successful']
        failed = row['failed']
        
        # Get category distribution for this model
        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM model_analyses
            WHERE model_name = ? AND error_message IS NULL
            GROUP BY category
        ''', (model,))
        
        category_dist = {r['category']: r['count'] for r in cursor.fetchall()}
        
        stats[model] = {
            'total_analyses': total,
            'successful': successful,
            'failed': failed,
            'avg_processing_time': row['avg_time'] or 0.0,
            'category_distribution': category_dist,
            'error_rate': failed / total if total > 0 else 0.0
        }
    
    return stats


def get_model_agreement_stats(conn: sqlite3.Connection) -> Dict:
    """
    Calculate inter-model agreement statistics across all posts.
    
    Returns:
        Dictionary with agreement metrics:
        {
            'total_posts_analyzed': int,
            'full_agreement_count': int,      # All models agree
            'partial_agreement_count': int,   # 2+ models agree
            'no_agreement_count': int,        # All models disagree
            'avg_agreement_score': float,
            'agreement_by_category': dict
        }
    """
    cursor = conn.cursor()
    
    # Get all posts with multiple model analyses
    cursor.execute('''
        SELECT post_id, COUNT(DISTINCT model_name) as model_count
        FROM model_analyses
        WHERE error_message IS NULL
        GROUP BY post_id
        HAVING model_count > 1
    ''')
    
    posts = cursor.fetchall()
    
    if not posts:
        return {
            'total_posts_analyzed': 0,
            'full_agreement_count': 0,
            'partial_agreement_count': 0,
            'no_agreement_count': 0,
            'avg_agreement_score': 0.0,
            'agreement_by_category': {}
        }
    
    full_agreement = 0
    partial_agreement = 0
    no_agreement = 0
    agreement_scores = []
    category_agreements = {}
    
    for post_row in posts:
        post_id = post_row['post_id']
        consensus = get_model_consensus(conn, post_id)
        
        if consensus:
            agreement_score = consensus['agreement_score']
            agreement_scores.append(agreement_score)
            
            if agreement_score == 1.0:
                full_agreement += 1
            elif agreement_score >= 0.5:
                partial_agreement += 1
            else:
                no_agreement += 1
            
            # Track agreement by category
            category = consensus['category']
            if category not in category_agreements:
                category_agreements[category] = []
            category_agreements[category].append(agreement_score)
    
    # Calculate average agreement per category
    category_avg_agreement = {
        cat: sum(scores) / len(scores) if scores else 0.0
        for cat, scores in category_agreements.items()
    }
    
    return {
        'total_posts_analyzed': len(posts),
        'full_agreement_count': full_agreement,
        'partial_agreement_count': partial_agreement,
        'no_agreement_count': no_agreement,
        'avg_agreement_score': sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0,
        'agreement_by_category': category_avg_agreement
    }


def update_consensus_in_content_analyses(conn: sqlite3.Connection, post_id: str) -> bool:
    """
    Update the content_analyses table with multi-model consensus results.
    Creates the row if it doesn't exist.
    
    Args:
        conn: Database connection
        post_id: Post/tweet identifier
        
    Returns:
        True if update/insert was successful, False otherwise
    """
    consensus = get_model_consensus(conn, post_id)
    
    if not consensus:
        return False
    
    cursor = conn.cursor()
    
    # Get tweet data for the insert
    cursor.execute('''
        SELECT username, content, tweet_timestamp
        FROM tweets
        WHERE tweet_id = ?
    ''', (post_id,))
    
    tweet_row = cursor.fetchone()
    if not tweet_row:
        return False
    
    # Insert or replace the content analysis record
    cursor.execute('''
        INSERT OR REPLACE INTO content_analyses (
            post_id,
            author_username,
            category,
            local_explanation,
            analysis_timestamp,
            multi_model_analysis,
            model_consensus_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        post_id,
        tweet_row['username'],
        consensus['category'],  # Use consensus as the main category
        f"Multi-model consensus analysis: {consensus['category']} (agreement: {consensus['agreement_score']:.2f})",
        datetime.now().isoformat(),
        1,  # multi_model_analysis = True
        consensus['category']
    ))
    
    conn.commit()
    
    return True


def get_posts_for_multi_model_analysis(
    conn: sqlite3.Connection,
    limit: Optional[int] = None,
    username: Optional[str] = None
) -> List[Dict]:
    """
    Get posts that need multi-model analysis or reanalysis.
    
    Args:
        conn: Database connection
        limit: Maximum number of posts to return
        username: Optional filter by username
        
    Returns:
        List of post dictionaries with post_id, content, and media_urls
    """
    cursor = conn.cursor()
    
    # Build query
    where_clauses = []
    params = []
    
    if username:
        where_clauses.append("t.username = ?")
        params.append(username)
    
    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    limit_sql = f"LIMIT {limit}" if limit else ""
    
    query = f'''
        SELECT 
            t.tweet_id as post_id,
            t.content,
            t.media_links as media_urls,
            ca.multi_model_analysis
        FROM tweets t
        LEFT JOIN content_analyses ca ON t.tweet_id = ca.post_id
        {where_sql}
        ORDER BY t.tweet_timestamp DESC
        {limit_sql}
    '''
    
    cursor.execute(query, params)
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'post_id': row['post_id'],
            'content': row['content'],
            'media_urls': row['media_urls'].split(',') if row['media_urls'] else [],
            'already_analyzed': bool(row['multi_model_analysis'])
        })
    
    return results
