import pytest
from app.db.mongodb import check_mongo_connection
from app.core.config import get_settings


@pytest.mark.asyncio
async def test_mongo_unconfigured_fallback():
    settings = get_settings()
    # Ensure mongodb_uri is empty
    original_uri = settings.mongodb_uri
    settings.mongodb_uri = ""
    try:
        is_connected, details = await check_mongo_connection()
        assert is_connected is False
        assert "MONGODB_URI" in details or "not configured" in details.lower()
    finally:
        settings.mongodb_uri = original_uri


@pytest.mark.asyncio
async def test_mongo_unreachable_uri_graceful_handling():
    settings = get_settings()
    original_uri = settings.mongodb_uri
    original_timeout = settings.mongodb_connect_timeout_ms
    # Set an unreachable URI with 500ms timeout
    settings.mongodb_uri = "mongodb://127.0.0.1:27099/?connectTimeoutMS=500"
    settings.mongodb_connect_timeout_ms = 500
    try:
        is_connected, details = await check_mongo_connection()
        assert is_connected is False
        assert isinstance(details, str)
        assert len(details) > 0
    finally:
        settings.mongodb_uri = original_uri
        settings.mongodb_connect_timeout_ms = original_timeout
