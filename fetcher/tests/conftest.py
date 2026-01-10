import pytest
import time
from fetcher import parsers


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires browser)"
    )


@pytest.fixture(autouse=True)
def disable_delays(monkeypatch):
    """Make human_delay and time.sleep quick/no-op during tests."""
    monkeypatch.setattr(parsers, 'human_delay', lambda *a, **k: None)
    monkeypatch.setattr(time, 'sleep', lambda *a, **k: None)
    yield