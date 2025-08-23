import sqlite3
import argparse

def print_relationships(conn, username=None):
    c = conn.cursor()
    base_query = "SELECT tweet_id, tweet_url, username, content, media_links, is_repost, is_like, is_comment, parent_tweet_id, created_at FROM tweets"
    if username:
        base_query += " WHERE username = ?"
        c.execute(base_query, (username,))
    else:
        c.execute(base_query)
    rows = c.fetchall()
    for row in rows:
        tweet_id, tweet_url, user, content, media, repost, like, comment, parent, created = row
        print(f"Tweet ID: {tweet_id}\nUser: {user}\nTime: {created}")
        print(f"Content: {content}")
        print(f"URL: {tweet_url}")
        if media:
            print(f"Media: {media}")
        if repost:
            print("Type: Repost")
        if like:
            print("Type: Like")
        if comment:
            print(f"Type: Comment/Quote (parent tweet: {parent})")
        print("-"*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query and display tweets and their relationships from the DB.")
    parser.add_argument('--user', help='Filter by username')
    args = parser.parse_args()
    conn = sqlite3.connect("accounts.db")
    print_relationships(conn, args.user)
    conn.close()