"""
register_watch.py
Run this weekly to keep Gmail push notifications active.
Gmail watch expires every 7 days.
"""

import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

PUBSUB_TOPIC = "projects/email-cal-assistant/topics/gmail-push"

def register_watch():
    token_data = json.loads(open("token.json").read())
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data["scopes"],
    )
    if creds.expired:
        creds.refresh(Request())
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)

    # Stop existing watch
    try:
        service.users().stop(userId="me").execute()
        print("Stopped existing watch")
    except Exception as e:
        print(f"No existing watch: {e}")

    # Register fresh watch
    result = service.users().watch(
        userId="me",
        body={
            "labelIds": ["INBOX"],
            "topicName": PUBSUB_TOPIC
        }
    ).execute()

    print(f"Watch registered!")
    print(f"historyId:  {result['historyId']}")
    print(f"expiration: {result['expiration']}")
    return result

if __name__ == "__main__":
    register_watch()
