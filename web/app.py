from flask import Flask, render_template, request
import sqlite3
import os

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
    conn.close()
    return render_template('user.html', username=username, tweets=tweets)

if __name__ == '__main__':
    app.run(debug=True)
