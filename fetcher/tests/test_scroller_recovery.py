import pytest

from fetcher.scroller import Scroller


class FakeLocator:
    def __init__(self, page, visible=True):
        self.page = page
        self._visible = visible

    def click(self):
        self.page._clicked_retry = True

    def is_visible(self):
        return self._visible


class FakePage:
    def __init__(self, has_retry=True):
        self._clicked_retry = False
        self._reloaded = False
        self._has_retry = has_retry
        self.evaluations = []

    def locator(self, selector):
        if self._has_retry and selector == 'button:has-text("Retry")':
            return FakeLocator(self, visible=True)
        # Return a falsy object if no retry
        return None

    def evaluate(self, js):
        # record evaluation calls
        self.evaluations.append(js)

    def reload(self, **kwargs):
        self._reloaded = True

    def query_selector_all(self, selector):
        # After clicking retry or after reload, pretend articles appear
        if self._clicked_retry or self._reloaded:
            return ["article"]
        return []


def test_try_recovery_clicks_retry_and_does_not_reload():
    page = FakePage(has_retry=True)
    scroller = Scroller()

    recovered = scroller.try_recovery_strategies(page, attempt_number=1)

    assert recovered is True
    assert page._clicked_retry is True
    assert page._reloaded is False


def test_try_recovery_reload_as_last_resort():
    page = FakePage(has_retry=False)
    scroller = Scroller()

    # attempts 1-4 will not find anything; attempt 5 is reload
    recovered = scroller.try_recovery_strategies(page, attempt_number=5)

    assert recovered is True
    assert page._reloaded is True
