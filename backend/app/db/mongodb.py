import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from app.core.config import get_settings

logger = logging.getLogger("medivision.db")

_mongo_client = None
_current_uri = None
_client_loop = None
_active_engine = "MongoDB Atlas"
LOCAL_MONGO_URI = "mongodb://127.0.0.1:27017"


def get_active_mongo_engine() -> str:
    global _active_engine
    return _active_engine


def is_placeholder_uri(uri: str) -> bool:
    if not uri or not uri.strip():
        return True
    lower = uri.lower()
    return (
        "example.mongodb.net" in lower
        or "<username>" in lower
        or "your-subdomain" in lower
        or "<gsajay18" in lower
    )


def sanitize_mongodb_uri(uri: str) -> str:
    """
    Removes stray placeholder angle brackets around credentials if present.
    Example: mongodb+srv://<user>:<pwd>@host -> mongodb+srv://user:pwd@host
    """
    if not uri:
        return ""
    uri = uri.strip()
    if "@" in uri and "://" in uri:
        prefix, rest = uri.split("://", 1)
        auth_part, host_part = rest.split("@", 1)
        if ":" in auth_part:
            user, pwd = auth_part.split(":", 1)
            user = user.lstrip("<").rstrip(">")
            pwd = pwd.lstrip("<").rstrip(">")
            return f"{prefix}://{user}:{pwd}@{host_part}"
    return uri


def get_mongo_client(force_uri: Optional[str] = None):
    global _mongo_client, _current_uri, _client_loop, _active_engine
    settings = get_settings()

    if force_uri:
        raw_uri = force_uri
    elif _active_engine == "Local MongoDB":
        raw_uri = LOCAL_MONGO_URI
    else:
        raw_uri = settings.mongodb_uri

    if not raw_uri or not raw_uri.strip() or is_placeholder_uri(raw_uri):
        if _active_engine == "Local MongoDB":
            raw_uri = LOCAL_MONGO_URI
        else:
            if _mongo_client is not None:
                try:
                    _mongo_client.close()
                except Exception:
                    pass
                _mongo_client = None
                _current_uri = None
                _client_loop = None
            return None

    clean_uri = sanitize_mongodb_uri(raw_uri)

    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    # Recreate client if uninitialized, URI changed, or event loop was recreated (e.g. across async tasks/tests)
    if (
        _mongo_client is None
        or _current_uri != clean_uri
        or (_client_loop is not None and current_loop is not None and _client_loop != current_loop)
    ):
        if _mongo_client is not None:
            try:
                _mongo_client.close()
            except Exception:
                pass
            _mongo_client = None

        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            try:
                import certifi
                ca_file = certifi.where()
            except Exception:
                ca_file = None

            timeout_ms = 2000 if "127.0.0.1" in clean_uri or "localhost" in clean_uri else settings.mongodb_connect_timeout_ms
            client_kwargs = {"serverSelectionTimeoutMS": timeout_ms}
            if ca_file and ("mongodb+srv://" in clean_uri or "mongodb.net" in clean_uri):
                client_kwargs["tlsCAFile"] = ca_file

            _mongo_client = AsyncIOMotorClient(
                clean_uri,
                **client_kwargs,
            )
            _current_uri = clean_uri
            _client_loop = current_loop
        except Exception as exc:
            logger.warning(f"Could not initialize MongoDB client: {exc}")
            _mongo_client = None
            _current_uri = None
            _client_loop = None
            return None

    return _mongo_client


def get_database():
    client = get_mongo_client()
    if client is None:
        return None
    settings = get_settings()
    return client[settings.mongodb_db_name]


async def check_mongo_connection() -> Tuple[bool, str]:
    """
    Checks MongoDB connectivity with hybrid fallback:
    1. Tests configured MongoDB Atlas cloud connection.
    2. If Atlas is unreachable (e.g. IP whitelist TLS alert), seamlessly falls back
       to local MongoDB running on 127.0.0.1:27017.
    """
    global _active_engine
    settings = get_settings()

    if not settings.mongodb_uri or not settings.mongodb_uri.strip():
        _active_engine = "MongoDB Atlas"
        return False, "MONGODB_URI not configured in .env (waiting for Atlas connection string)"

    clean_uri = sanitize_mongodb_uri(settings.mongodb_uri)
    is_placeholder = is_placeholder_uri(clean_uri)
    is_cloud_uri = "mongodb+srv://" in clean_uri or "mongodb.net" in clean_uri

    atlas_reason = ""
    # 1. Attempt Primary MongoDB Atlas Connection if valid non-placeholder URI provided
    if not is_placeholder:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            try:
                import certifi
                ca_file = certifi.where()
            except Exception:
                ca_file = None

            timeout_ms = min(2000, settings.mongodb_connect_timeout_ms)
            client_kwargs = {
                "serverSelectionTimeoutMS": timeout_ms,
            }
            if ca_file and is_cloud_uri:
                client_kwargs["tlsCAFile"] = ca_file

            atlas_client = AsyncIOMotorClient(clean_uri, **client_kwargs)
            await asyncio.wait_for(atlas_client.admin.command("ping"), timeout=3.0)
            _active_engine = "MongoDB Atlas"
            get_mongo_client(clean_uri)
            return True, f"Connected to MongoDB Atlas ({settings.mongodb_db_name})"
        except Exception as atlas_exc:
            exc_str = str(atlas_exc)
            if "TLSV1_ALERT_INTERNAL_ERROR" in exc_str or "SSL handshake failed" in exc_str:
                logger.warning(f"MongoDB Atlas TLS alert (IP access list restricted): {exc_str}")
                atlas_reason = "Atlas TLS alert: IP not whitelisted in Atlas Network Access"
            else:
                logger.info(f"MongoDB Atlas unreachable ({atlas_exc}); checking local MongoDB fallback...")
                atlas_reason = "Atlas connection timed out"
    else:
        atlas_reason = "Atlas URI contains placeholder domain"

    # 2. Hybrid Fallback: If configured URI was Atlas/cloud, check local MongoDB
    if is_cloud_uri:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            local_client = AsyncIOMotorClient(LOCAL_MONGO_URI, serverSelectionTimeoutMS=1500)
            await asyncio.wait_for(local_client.admin.command("ping"), timeout=1.5)
            _active_engine = "Local MongoDB"
            get_mongo_client(LOCAL_MONGO_URI)
            return True, f"Connected to Local MongoDB (127.0.0.1:27017) - {atlas_reason}"
        except Exception as local_exc:
            logger.warning(f"Local MongoDB fallback failed: {local_exc}")

    _active_engine = "MongoDB Atlas"
    if "TLS" in atlas_reason or "whitelisted" in atlas_reason:
        return False, "Atlas TLS alert: IP not whitelisted (Security -> Network Access in Atlas)"
    if is_placeholder:
        return False, "MONGODB_URI contains placeholder ('example.mongodb.net')"
    return False, "Connection timed out (verify network & Atlas IP whitelist)"


async def save_prediction_record(record: dict) -> Optional[str]:
    """
    Saves an analysis prediction record to the 'analysis_history' collection.
    Operates safely with a strict timeout and never raises exceptions.
    """
    settings = get_settings()
    if not settings.mongodb_uri or not settings.mongodb_uri.strip():
        return None

    try:
        db = get_database()
        if db is None:
            return None

        collection = db["analysis_history"]
        doc = dict(record)
        if "timestamp" not in doc:
            doc["timestamp"] = datetime.now(timezone.utc).isoformat()

        timeout_sec = max(1.0, settings.mongodb_connect_timeout_ms / 1000.0)
        res = await asyncio.wait_for(collection.insert_one(doc), timeout=timeout_sec)
        inserted_id = str(res.inserted_id)
        logger.info(f"Saved prediction history record: {inserted_id} ({_active_engine})")
        return inserted_id
    except asyncio.TimeoutError:
        logger.warning(f"{_active_engine} prediction save timed out")
        return None
    except Exception as exc:
        logger.warning(f"Could not save prediction to {_active_engine}: {exc}")
        return None


async def get_prediction_history(limit: int = 50, skip: int = 0) -> List[dict]:
    """
    Fetches chronological prediction history.
    Returns empty list if database is unconfigured or unreachable.
    """
    settings = get_settings()
    if not settings.mongodb_uri or not settings.mongodb_uri.strip():
        return []

    try:
        db = get_database()
        if db is None:
            return []

        collection = db["analysis_history"]
        timeout_sec = max(1.5, settings.mongodb_connect_timeout_ms / 1000.0)

        async def _query():
            cursor = collection.find({}).sort("timestamp", -1).skip(skip).limit(limit)
            items = []
            async for doc in cursor:
                doc["id"] = str(doc.get("_id", ""))
                doc.pop("_id", None)
                items.append(doc)
            return items

        return await asyncio.wait_for(_query(), timeout=timeout_sec)
    except Exception as exc:
        logger.warning(f"Could not retrieve prediction history from {_active_engine}: {exc}")
        return []


async def close_mongo_connection():
    global _mongo_client, _current_uri, _client_loop
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
        _mongo_client = None
        _current_uri = None
        _client_loop = None
