import os
import sqlite3
import time
import argparse
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError


# Load credentials from .env
load_dotenv()
USERNAME = os.getenv("X_USERNAME")
PASSWORD = os.getenv("X_PASSWORD")
EMAIL_OR_PHONE = os.getenv("X_EMAIL_OR_PHONE")

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"

DB_PATH = "accounts.db"

def login_and_save_session(page, username, password):

    # Use stealth techniques to avoid detection
    page.route("**/*", lambda route, request: route.continue_() if "webdriver" not in request.url else route.abort())
    page.goto("https://x.com/login")


    # Step 1: Enter username
    try:
        page.wait_for_selector('input[name="text"]', timeout=10000)
        page.fill('input[name="text"]', username)
        page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
    except TimeoutError:
        print("No apareció el campo para el usuario/email, puede que ya esté logueado o haya otro paso.")

    # Step 2: Handle possible 'unusual activity' or confirmation prompt for email/phone
    try:
        page.wait_for_selector('input[name="text"]', timeout=4000)
        unusual_label = None
        try:
            unusual_label = page.query_selector('div:has-text("unusual activity")')
        except Exception:
            pass
        if unusual_label:
            print("Unusual activity detected, entering email/phone...")
            page.fill('input[name="text"]', EMAIL_OR_PHONE or username)
        else:
            print("Se pide confirmar usuario/telefono, rellenando nuevamente...")
            page.fill('input[name="text"]', username)
        page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
    except TimeoutError:
        pass

    # Step 3: Enter password
    try:
        page.wait_for_selector('input[name="password"]', timeout=10000)
        page.fill('input[name="password"]', password)
        page.click('div[data-testid="LoginForm_Login_Button"], button:has-text("Iniciar sesión"), button:has-text("Log in")')
    except TimeoutError:
        print("No apareció el campo de contraseña, puede que ya esté logueado o haya otro paso.")

    try:
        page.wait_for_url("https://x.com/home", timeout=15000)
        print("Login exitoso.")
    except TimeoutError:
        print("No se pudo confirmar el login, verifica si hay captcha o 2FA.")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tweets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tweet_id TEXT UNIQUE,
        tweet_url TEXT,
        username TEXT,
        content TEXT,
        media_links TEXT,
        is_repost INTEGER DEFAULT 0,
        is_like INTEGER DEFAULT 0,
        is_comment INTEGER DEFAULT 0,
        parent_tweet_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    # Add tweet_url column if upgrading from old DB
    try:
        c.execute('ALTER TABLE tweets ADD COLUMN tweet_url TEXT')
    except Exception:
        pass
    conn.commit()
    return conn

def save_tweet(conn, tweet_id, tweet_url, username, content, media_links=None, is_repost=0, is_like=0, is_comment=0, parent_tweet_id=None):
    c = conn.cursor()
    try:
        c.execute("SELECT 1 FROM tweets WHERE tweet_id = ?", (tweet_id,))
        if c.fetchone():
            # Already saved
            return
        c.execute("""
            INSERT INTO tweets (tweet_id, tweet_url, username, content, media_links, is_repost, is_like, is_comment, parent_tweet_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tweet_id, tweet_url, username, content, media_links, is_repost, is_like, is_comment, parent_tweet_id))
        conn.commit()
    except Exception as e:
        print(f"Error saving tweet: {e}")

def fetch_full_tweets(page, username, max_tweets=100):
    url = f"https://x.com/{username}"
    page.goto(url)
    page.wait_for_selector('article div[lang]', timeout=15000)

    tweet_ids = set()
    tweets_data = []
    last_height = 0
    scroll_attempts = 0
    max_scroll_attempts = 30

    def extract_media_links(article):
        media_links = []
        # Images
        for img in article.query_selector_all('img[src*="twimg.com/media/"]'):
            src = img.get_attribute('src')
            if src:
                media_links.append(src)
        # Videos (poster images)
        for video in article.query_selector_all('video'):
            poster = video.get_attribute('poster')
            if poster:
                media_links.append(poster)
        return media_links

    def detect_relationship(article):
        # More robust detection for repost, like, comment/quote
        is_repost = 0
        is_like = 0
        is_comment = 0
        parent_tweet_id = None

        # Repost: look for retweet/repost icon or label
        repost_icon = article.query_selector('svg[aria-label="Repost"]')
        repost_label = article.query_selector('span:has-text("Reposted")')
        if repost_icon or repost_label:
            is_repost = 1

        # Like: look for like icon filled (aria-label="Like") or label
        like_icon = article.query_selector('svg[aria-label="Like"][fill]')
        like_label = article.query_selector('span:has-text("Liked")')
        if like_icon or like_label:
            is_like = 1

        # Comment/Quote: look for quote tweet structure (quoted tweet inside article)
        quote_tweet = article.query_selector('div[data-testid="tweetText"] ~ div[role="group"] ~ div[tabindex] article')
        if quote_tweet:
            is_comment = 1
            # Try to get parent tweet id from quoted tweet link
            parent = quote_tweet.query_selector('a[href*="/status/"]')
            if parent:
                href = parent.get_attribute('href')
                if href:
                    parent_tweet_id = href.split("/")[-1]
        return is_repost, is_like, is_comment, parent_tweet_id

    while len(tweet_ids) < max_tweets and scroll_attempts < max_scroll_attempts:
        articles = page.query_selector_all('article')
        for article in articles:
            try:
                tweet_url_elem = article.query_selector('a[href*="/status/"]')
                if not tweet_url_elem:
                    continue
                href = tweet_url_elem.get_attribute('href')
                tweet_id = href.split("/")[-1]
                tweet_url = f"https://x.com{href}"
                if tweet_id in tweet_ids:
                    continue
                tweet_text_elem = article.query_selector('div[lang]')
                if not tweet_text_elem:
                    continue
                # Get full text, including all spans
                content = " ".join([span.inner_text() for span in tweet_text_elem.query_selector_all('span')])
                if not content.strip():
                    content = tweet_text_elem.inner_text()
                media_links = extract_media_links(article)
                is_repost, is_like, is_comment, parent_tweet_id = detect_relationship(article)
                tweets_data.append({
                    'tweet_id': tweet_id,
                    'tweet_url': tweet_url,
                    'username': username,
                    'content': content,
                    'media_links': ",".join(media_links) if media_links else None,
                    'is_repost': is_repost,
                    'is_like': is_like,
                    'is_comment': is_comment,
                    'parent_tweet_id': parent_tweet_id
                })
                tweet_ids.add(tweet_id)
                if len(tweet_ids) >= max_tweets:
                    break
            except Exception as e:
                print(f"Error parsing tweet: {e}")
        # Scroll down
        page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        time.sleep(1.5)
        new_height = page.evaluate("document.body.scrollHeight")
        if new_height == last_height:
            scroll_attempts += 1
        else:
            scroll_attempts = 0
        last_height = new_height
    return tweets_data

def main():
    parser = argparse.ArgumentParser(description="Fetch tweets from a given X (Twitter) user.")
    parser.add_argument("username", help="The username to fetch tweets from (without @)")
    parser.add_argument("--max", type=int, default=100, help="Maximum number of tweets to fetch (default: 100)")
    args = parser.parse_args()

    username_to_fetch = args.username
    max_tweets = args.max

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720},
            locale="en-US",
            timezone_id="America/New_York",
            color_scheme="light",
            java_script_enabled=True,
        )

        page = context.new_page()
        # Stealth: Remove webdriver flag and spoof more properties
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.navigator.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)

        login_and_save_session(page, USERNAME, PASSWORD)
        context.storage_state(path="x_session.json")
        print("Sesión guardada en x_session.json")

        conn = init_db()
        tweets = fetch_full_tweets(page, username_to_fetch, max_tweets=max_tweets)
        print(f"Fetched {len(tweets)} tweets.")
        for i, tweet in enumerate(tweets, 1):
            print(f"Tweet #{i}:\n{tweet['content']}\nURL: {tweet['tweet_url']}\nMedia: {tweet['media_links']}\nRepost: {tweet['is_repost']} Like: {tweet['is_like']} Comment: {tweet['is_comment']} Parent: {tweet['parent_tweet_id']}\n{'-'*40}")
            save_tweet(
                conn,
                tweet['tweet_id'],
                tweet['tweet_url'],
                tweet['username'],
                tweet['content'],
                tweet['media_links'],
                tweet['is_repost'],
                tweet['is_like'],
                tweet['is_comment'],
                tweet['parent_tweet_id']
            )
        conn.close()
        browser.close()

if __name__ == "__main__":
    main()
