from __future__ import print_function
import os
from datetime import datetime, timezone

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Scopes needed by this project:
# - gmail.readonly: fetch/parse incoming emails (history + message get)
# - gmail.send: send the confirmation email
# - calendar.events: create calendar events
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.events",
]

def get_creds():
    creds = None

    # Load saved token if it exists
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If no valid token, get one
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            # Desktop (installed) OAuth clients typically authorize http://localhost
            # and allow an ephemeral port chosen at runtime.
            creds = flow.run_local_server(port=0, open_browser=True)

        # Save token for next run
        with open("token.json", "w") as f:
            f.write(creds.to_json())
        print("✓ token.json saved\n")

    return creds


def test_gmail(creds):
    print("=" * 50)
    print("GMAIL — latest 5 messages")
    print("=" * 50)
    try:
        service = build("gmail", "v1", credentials=creds)
        results = service.users().messages().list(
            userId="me", maxResults=5
        ).execute()
        messages = results.get("messages", [])

        if not messages:
            print("No messages found.")
            return

        for m in messages:
            msg = service.users().messages().get(
                userId="me", id=m["id"], format="metadata"
            ).execute()
            headers = msg["payload"]["headers"]
            subject = next((h["value"] for h in headers if h["name"].lower() == "subject"), "(no subject)")
            sender  = next((h["value"] for h in headers if h["name"].lower() == "from"),    "(no sender)")
            print(f"  From:    {sender[:60]}")
            print(f"  Subject: {subject[:60]}")
            print()

    except HttpError as e:
        print(f"  ERROR: {e.status_code} — {e.reason}")


def test_calendar(creds):
    print("=" * 50)
    print("GOOGLE CALENDAR — next 5 events")
    print("=" * 50)
    try:
        service = build("calendar", "v3", credentials=creds)
        now = datetime.now(timezone.utc).isoformat()

        events = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=5,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        items = events.get("items", [])

        if not items:
            print("  No upcoming events found.")
            return

        for e in items:
            start = e["start"].get("dateTime", e["start"].get("date", "?"))
            title = e.get("summary", "(no title)")
            print(f"  {start}  |  {title}")

    except HttpError as e:
        print(f"  ERROR: {e.status_code} — {e.reason}")


if __name__ == "__main__":
    print("\nGetting credentials...\n")
    creds = get_creds()
    print("✓ Authenticated\n")

    test_gmail(creds)
    print()
    test_calendar(creds)

    print("\n✓ All done — both APIs are working!")
