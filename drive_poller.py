"""
drive_poller.py — Google Drive poller for meeting transcript detection
Polls a Drive folder every POLL_INTERVAL seconds, detects new .txt files,
and triggers the Lambda webhook.

FIX: Tracks already-triggered file IDs in a local JSON file so that:
  1. The same file never triggers Lambda twice (even across poller restarts).
  2. Eliminates the duplicate trigger that the Redis lock was silently eating.

Processed file IDs are persisted to PROCESSED_IDS_FILE (default: ~/processed_files.json).
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build as gdrive_build

# ── Config ─────────────────────────────────────────────────────────────────────
POLL_INTERVAL        = int(os.environ.get("POLL_INTERVAL_SECONDS", "30"))
LAMBDA_WEBHOOK_URL   = os.environ.get("LAMBDA_WEBHOOK_URL", "")
WEBHOOK_SECRET       = os.environ.get("WEBHOOK_SECRET", "meeting-summarizer-secret-2026")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")
PROCESSED_IDS_FILE   = Path(os.environ.get("PROCESSED_IDS_FILE", str(Path.home() / "processed_files.json")))
MAX_PROCESSED_IDS    = 500   # cap to avoid unbounded growth; trims oldest when exceeded

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ── Processed-file registry ────────────────────────────────────────────────────

def _load_processed_ids() -> dict:
    """
    Load {file_id: iso_timestamp} from disk.
    Returns empty dict if file doesn't exist or is corrupt.
    """
    if not PROCESSED_IDS_FILE.exists():
        return {}
    try:
        data = json.loads(PROCESSED_IDS_FILE.read_text())
        logger.info("Loaded %d processed file IDs from %s", len(data), PROCESSED_IDS_FILE)
        return data
    except Exception as e:
        logger.warning("Could not load processed IDs file (%s) — starting fresh: %s", PROCESSED_IDS_FILE, e)
        return {}


def _save_processed_ids(processed: dict) -> None:
    """
    Persist processed IDs to disk atomically (write to .tmp then rename).
    Trims to MAX_PROCESSED_IDS oldest entries if the set grows too large.
    """
    # Trim if needed — sort by timestamp, keep newest
    if len(processed) > MAX_PROCESSED_IDS:
        sorted_ids = sorted(processed.items(), key=lambda x: x[1])
        processed = dict(sorted_ids[-MAX_PROCESSED_IDS:])
        logger.info("Trimmed processed IDs to %d entries", len(processed))

    tmp_path = PROCESSED_IDS_FILE.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(processed, indent=2))
        tmp_path.rename(PROCESSED_IDS_FILE)
    except Exception as e:
        logger.error("Could not save processed IDs: %s", e)


def _mark_processed(processed: dict, file_id: str) -> dict:
    """Add file_id with current UTC timestamp, save to disk, return updated dict."""
    processed[file_id] = datetime.now(timezone.utc).isoformat()
    _save_processed_ids(processed)
    return processed


# ── Google Drive ────────────────────────────────────────────────────────────────

def _build_drive_service():
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
    sa_info = json.loads(sa_json)
    creds   = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return gdrive_build("drive", "v3", credentials=creds)


def _list_txt_files(service) -> list[dict]:
    """Return list of .txt files in the watched folder, ordered by creation time desc."""
    query = (
        f"'{GOOGLE_DRIVE_FOLDER_ID}' in parents "
        f"and mimeType='text/plain' "
        f"and trashed=false"
    )
    result = service.files().list(
        q=query,
        orderBy="createdTime desc",
        pageSize=20,
        fields="files(id, name, createdTime)",
    ).execute()
    return result.get("files", [])


# ── Webhook trigger ─────────────────────────────────────────────────────────────

def _trigger_lambda(file_id: str, file_name: str) -> bool:
    """
    POST the new_transcript webhook to Lambda.
    Returns True on success (2xx), False otherwise.
    """
    payload = {
        "type":      "new_transcript",
        "secret":    WEBHOOK_SECRET,
        "file_id":   file_id,
        "file_name": file_name,
    }
    try:
        resp = requests.post(LAMBDA_WEBHOOK_URL, json=payload, timeout=10)
        if resp.ok:
            logger.info("Triggered Lambda for %s (%s) → %s", file_name, file_id, resp.status_code)
            return True
        else:
            logger.error("Lambda webhook returned %s for %s: %s", resp.status_code, file_name, resp.text[:200])
            return False
    except Exception as e:
        logger.error("Lambda webhook failed for %s (%s): %s", file_name, file_id, e)
        return False


# ── Main poll loop ──────────────────────────────────────────────────────────────

def main():
    logger.info(
        "Drive poller starting — folder=%s interval=%ds webhook=%s processed_ids_file=%s",
        GOOGLE_DRIVE_FOLDER_ID, POLL_INTERVAL, LAMBDA_WEBHOOK_URL, PROCESSED_IDS_FILE,
    )

    # Load persisted processed file IDs (survives restarts)
    processed_ids: dict = _load_processed_ids()
    logger.info("Starting with %d already-processed file ID(s)", len(processed_ids))

    service = _build_drive_service()

    while True:
        try:
            files = _list_txt_files(service)
            new_files = [f for f in files if f["id"] not in processed_ids]

            if new_files:
                logger.info("Found %d new file(s), %d already processed", len(new_files), len(processed_ids))
            else:
                logger.debug("No new files (total=%d, all processed)", len(files))

            for f in new_files:
                file_id   = f["id"]
                file_name = f["name"]
                logger.info("New transcript detected: %s (%s)", file_name, file_id)

                # Mark BEFORE triggering — prevents duplicate on next poll cycle
                # even if the webhook call takes a while or Lambda is slow to set its lock
                processed_ids = _mark_processed(processed_ids, file_id)
                logger.info("Marked %s as processed (persisted to disk)", file_id)

                success = _trigger_lambda(file_id, file_name)
                if not success:
                    # Remove from processed so it will be retried on next poll
                    # (only retry if webhook delivery itself failed, not if Lambda rejected it)
                    del processed_ids[file_id]
                    _save_processed_ids(processed_ids)
                    logger.warning("Removed %s from processed IDs — will retry on next poll", file_id)

        except Exception as e:
            logger.error("Poll cycle error: %s", e, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
