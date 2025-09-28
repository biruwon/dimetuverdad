import pytest


@pytest.fixture(autouse=True)
def disable_delays(monkeypatch):
    """Make human_delay and time.sleep quick/no-op during tests."""
    import time
    try:
        from fetcher import fetch_tweets
        monkeypatch.setattr(fetch_tweets, 'human_delay', lambda *a, **k: None)
    except Exception:
        pass
    monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)
    yield