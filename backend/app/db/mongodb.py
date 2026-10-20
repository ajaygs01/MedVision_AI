import asyncio
import logging
from typing import Optional, Tuple
from app.core.config import get_settings

logger = logging.getLogger("medivision.db")

_mongo_client = None


def get_mongo_client():
    global _mongo_client
    settings = get_settings()
    if not settings.mongodb_uri:
        return None

    if _mongo_client is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            _mongo_client = AsyncIOMotorClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=settings.mongodb_connect_timeout_ms,
            )
        except Exception as exc:
            logger.warning(f"Could not initialize MongoDB client: {exc}")
            return None
    return _mongo_client


async def check_mongo_connection() -> Tuple[bool, str]:
    """
    Checks MongoDB Atlas connectivity with timeout.
    Fails gracefully if unconfigured or unreachable.
    """
    settings = get_settings()
    if not settings.mongodb_uri:
        return False, "MONGODB_URI not configured in .env (waiting for Atlas connection string)"

    client = get_mongo_client()
    if client is None:
        return False, "Driver initialization failed or URI invalid"

    try:
        # Run a quick ping command with a strict timeout
        await asyncio.wait_for(
            client.admin.command("ping"),
            timeout=settings.mongodb_connect_timeout_ms / 1000.0,
        )
        return True, f"Connected to MongoDB Atlas ({settings.mongodb_db_name})"
    except asyncio.TimeoutError:
        return False, "Connection timed out (verify network & Atlas IP whitelist)"
    except Exception as exc:
        return False, f"Connection error: {str(exc)}"


async def close_mongo_connection():
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
