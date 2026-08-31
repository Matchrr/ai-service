import os

import pytest

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SERPAPI_API_KEY", "test-serpapi-key")
os.environ.setdefault("XANO_API_URL", "https://test.xano.io/api:test")
os.environ.setdefault("XANO_API_KEY", "test-xano-key")
os.environ.setdefault("XANO_MATCH_ENDPOINT", "")


@pytest.fixture
def ai_settings(monkeypatch):
    from app.core.config import settings

    original = {
        "serpapi_api_key": settings.serpapi_api_key,
        "xano_api_url": settings.xano_api_url,
        "harvest_stale_hours": settings.harvest_stale_hours,
        "harvest_cache_hours": settings.harvest_cache_hours,
        "harvest_standing_hours": settings.harvest_standing_hours,
        "harvest_pages": settings.harvest_pages,
        "embedding_dimensions": settings.embedding_dimensions,
    }

    def _patch(**overrides):
        for key, value in overrides.items():
            monkeypatch.setattr(settings, key, value)
        return settings

    yield _patch

    for key, value in original.items():
        monkeypatch.setattr(settings, key, value)
