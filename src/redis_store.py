"""
redis_store.py — Redis session storage for Slack button payloads + human feedback

Replaces the pattern of embedding full JSON in Slack button values.
Instead, the payload is stored in Redis under a UUID session_id, and
only the session_id is placed in the button value field.

Usage:
    from redis_store import save_session, load_session, record_feedback

    # When posting a Slack preview card:
    session_id = save_session(pending_dict)
    button["value"] = session_id

    # When parse_input receives the button click:
    data = load_session(session_id)   # returns dict, key is kept alive

    # After the human's decision is actioned:
    record_feedback(session_id, "accepted", {"calendar_link": "..."})
    record_feedback(session_id, "rejected")
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import redis

from state import REDIS_URL, SESSION_TTL_SECONDS

logger = logging.getLogger(__name__)

# Feedback records are kept for 7 days for audit / RLHF use
FEEDBACK_TTL_SECONDS = 7 * 24 * 3600

_client: redis.Redis | None = None


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def save_session(data: dict) -> str:
    """
    Serialize *data* as JSON, store it in Redis with a TTL, and return the
    session_id (UUID string) used as the key.
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
    The key is kept alive — record_feedback() will update it later.
    """
    try:
        raw = _get_client().get(session_id)
    except Exception as exc:
        logger.error("redis_store: load_session failed: %s", exc)
        return None

    if raw is None:
        logger.warning("redis_store: session %s not found or expired", session_id)
        return None

    try:
        data = json.loads(raw)
        logger.info("redis_store: loaded session %s keys=%s", session_id, list(data.keys()))
        return data
    except json.JSONDecodeError as exc:
        logger.error("redis_store: could not decode session %s: %s", session_id, exc)
        return None


def record_feedback(session_id: str, outcome: str, metadata: dict | None = None) -> None:
    """
    Merge human feedback into the existing session record and extend TTL
    to FEEDBACK_TTL_SECONDS (7 days) for audit / RLHF retention.

    outcome  — "accepted" | "rejected" | "failed"
    metadata — optional extra fields, e.g. {"calendar_link": "...", "jira_key": "..."}
    """
    client = _get_client()
    try:
        raw = client.get(session_id)
        data = json.loads(raw) if raw else {}
    except Exception as exc:
        logger.error("redis_store: record_feedback GET failed for %s: %s", session_id, exc)
        data = {}

    data["feedback"]    = outcome
    data["feedback_at"] = datetime.now(timezone.utc).isoformat()
    if metadata:
        data.update(metadata)

    try:
        client.setex(session_id, FEEDBACK_TTL_SECONDS, json.dumps(data))
        logger.info(
            "redis_store: recorded feedback session=%s outcome=%s metadata_keys=%s",
            session_id,
            outcome,
            list(metadata.keys()) if metadata else [],
        )
    except Exception as exc:
        logger.error("redis_store: record_feedback SET failed for %s: %s", session_id, exc)
