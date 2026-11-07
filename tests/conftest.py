import pytest


@pytest.fixture(autouse=True)
def isolated_cache(settings):
    """Keep tests away from the live Redis: they must not disturb a running worker's heartbeat or the
    login throttle, and a running app must not affect test results."""
    settings.CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
    from django.core.cache import cache

    # Local-memory caches at one location share their store, so throttle counts would carry over between tests.
    cache.clear()
