import pytest

from config.env import env

# Tests read the same .env as the app. Without this, every test question would be traced into the real Langfuse
# project, among the real users' questions.
env.LANGFUSE_PUBLIC_KEY = env.LANGFUSE_SECRET_KEY = ""
env.OTEL_EXPORTER_OTLP_ENDPOINT = ""


@pytest.fixture(autouse=True)
def isolated_cache(settings):
    """Keep tests away from the live Redis: they must not disturb a running worker's heartbeat or the
    login throttle, and a running app must not affect test results."""
    settings.CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}
    from django.core.cache import cache

    # Local-memory caches at one location share their store, so throttle counts would carry over between tests.
    cache.clear()
