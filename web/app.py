from flask import Flask, render_template, request
import sqlite3
import os
import json

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'accounts.db')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user/<username>')
def user_tweets(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT tweet_url, content, media_links, is_repost, is_like, is_comment, parent_tweet_id, created_at FROM tweets WHERE username = ? ORDER BY created_at DESC", (username,))
    tweets = c.fetchall()
    # load analyses keyed by tweet_url
    c.execute("SELECT tweet_url, analysis_json FROM analyses")
    rows = c.fetchall()
    # parse JSON for easier rendering
    analysis_map = {}
    for r in rows:
        try:
            analysis_map[r[0]] = json.loads(r[1])
        except Exception:
            analysis_map[r[0]] = r[1]
    conn.close()
    return render_template('user.html', username=username, tweets=tweets, analysis_map=analysis_map)

if __name__ == '__main__':
    app.run(debug=True)
