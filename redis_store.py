"""
redis_store.py — Redis session storage for Slack button payloads

Replaces the pattern of embedding full JSON in Slack button values.
Instead, the payload is stored in Redis under a UUID session_id, and
only the session_id is placed in the button value field.

Usage:
    from redis_store import save_session, load_session

    # When posting a Slack preview card:
    session_id = save_session(pending_dict)
    button["value"] = session_id

    # When parse_input receives the button click:
    data = load_session(session_id)   # returns dict or None if expired
"""

import json
import logging
from uuid import uuid4

import redis

from state import REDIS_URL, SESSION_TTL_SECONDS

logger = logging.getLogger(__name__)

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def save_session(data: dict) -> str:
    """
    Serialize *data* as JSON, store it in Redis with a TTL, and return the
    session_id (UUID string) that was used as the key.
    """
    session_id = str(uuid4())
    try:
        _get_client().setex(session_id, SESSION_TTL_SECONDS, json.dumps(data))
        logger.info("redis_store: saved session %s (ttl=%ds)", session_id, SESSION_TTL_SECONDS)
    except Exception as exc:
        logger.error("redis_store: save_session failed: %s", exc)
        raise
    return session_id


def load_session(session_id: str) -> dict | None:
    """
    Fetch and deserialize the session stored under *session_id*.
    Returns None if the key is missing or expired.
    Deletes the key after a successful load (one-shot semantics).
    """
    try:
        raw = _get_client().getdel(session_id)
    except Exception as exc:
        logger.error("redis_store: load_session failed: %s", exc)
        return None

    if raw is None:
        logger.warning("redis_store: session %s not found or expired", session_id)
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("redis_store: could not decode session %s: %s", session_id, exc)
        return None
