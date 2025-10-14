import pytest


@pytest.fixture(autouse=True)
def disable_delays(monkeypatch):
    """Make human_delay and time.sleep quick/no-op during tests."""
    import time
    from fetcher import parsers
    monkeypatch.setattr(parsers, 'human_delay', lambda *a, **k: None)
    monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)
    yield