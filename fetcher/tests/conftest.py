import pytest
import time
from fetcher import parsers


@pytest.fixture(autouse=True)
def disable_delays(monkeypatch):
    """Make human_delay and time.sleep quick/no-op during tests."""
    monkeypatch.setattr(parsers, 'human_delay', lambda *a, **k: None)
    monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)
    yield