import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError

# Load credentials from .env
load_dotenv()
USERNAME = os.getenv("X_USERNAME")
PASSWORD = os.getenv("X_PASSWORD")

def login_and_save_session(page, username, password):
    page.goto("https://x.com/login")

    try:
        page.wait_for_selector('input[name="text"]', timeout=10000)
        page.fill('input[name="text"]', username)
        page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
    except TimeoutError:
        print("No apareció el campo para el usuario/email, puede que ya esté logueado o haya otro paso.")

    try:
        page.wait_for_selector('input[name="text"]', timeout=4000)
        print("Se pide confirmar usuario/telefono, rellenando nuevamente...")
        page.fill('input[name="text"]', username)
        page.click('div[data-testid="LoginForm_Login_Button"], div[role="button"]:has-text("Siguiente"), button:has-text("Next")')
    except TimeoutError:
        pass

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

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        page = context.new_page()

        login_and_save_session(page, USERNAME, PASSWORD)

        context.storage_state(path="x_session.json")
        print("Sesión guardada en x_session.json")

        page.goto("https://x.com/elonmusk")
        page.wait_for_selector('article div[lang]', timeout=15000)

        tweets = page.query_selector_all('article div[lang]')
        for i, tweet in enumerate(tweets, 1):
            print(f"Tweet #{i}:\n{tweet.inner_text()}\n{'-'*40}")

        browser.close()

if __name__ == "__main__":
    main()
