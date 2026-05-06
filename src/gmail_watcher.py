"""
gmail_watcher.py
================
Run this ONCE to register Gmail push notifications.
Gmail will automatically push to your Pub/Sub topic whenever
a new email arrives — triggering your Lambda autonomously.

The watch expires after 7 days — re-run weekly or set up a
CloudWatch EventBridge rule to call this on a schedule.

Usage:
    python gmail_watcher.py
"""

import json, os
from dotenv import load_dotenv
load_dotenv()

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

GOOGLE_TOKEN = os.environ.get("GOOGLE_TOKEN_JSON", "")
PUBSUB_TOPIC = os.environ.get("PUBSUB_TOPIC", "")


def _get_google_creds():
    token_data = json.loads(GOOGLE_TOKEN)
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data.get("scopes", []),
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def setup_gmail_watch():
    if not PUBSUB_TOPIC:
        print("❌ PUBSUB_TOPIC not set in .env")
        print("   Add: PUBSUB_TOPIC=projects/YOUR_PROJECT_ID/topics/gmail-push")
        return

    creds   = _get_google_creds()
    service = build("gmail", "v1", credentials=creds)

    try:
        result = service.users().watch(
            userId="me",
            body={
                "labelIds":  ["INBOX"],
                "topicName": PUBSUB_TOPIC,
            }
        ).execute()
        history_id = result.get("historyId")
        print(f"✅ Gmail watch registered")
        print(f"   History ID : {history_id}")
        print(f"   Expires    : {result.get('expiration')} ms epoch (~7 days)")
        print(f"   Topic      : {PUBSUB_TOPIC}")

        # Seed SSM with the watch historyId so email_fetch_and_parse has a
        # valid starting checkpoint on the first Pub/Sub notification
        try:
            from gmail_history import set_last_history_id
            set_last_history_id(str(history_id))
            print(f"✅ Seeded SSM historyId={history_id}")
        except Exception as ssm_err:
            print(f"⚠️  Could not write SSM (check IAM): {ssm_err}")

        print(f"\n⚠️  Re-run this script every 7 days to keep the watch active")
    except Exception as e:
        print(f"❌ Failed: {e}")


def stop_gmail_watch():
    creds   = _get_google_creds()
    service = build("gmail", "v1", credentials=creds)
    try:
        service.users().stop(userId="me").execute()
        print("✅ Gmail watch stopped")
    except Exception as e:
        print(f"❌ Failed to stop: {e}")


if __name__ == "__main__":
    setup_gmail_watch()