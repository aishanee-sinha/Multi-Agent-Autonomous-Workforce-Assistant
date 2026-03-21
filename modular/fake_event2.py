import json, urllib.parse, time
import os
from dotenv import load_dotenv
load_dotenv()

from orchestrator import handler

print("=" * 50)
print("STEP 1: Sending Slack message...")
print("=" * 50)

step1_event = {
    "headers": {},
    "isBase64Encoded": False,
    "body": json.dumps({
        "event": {
            "type": "message",
            "channel": os.getenv("SLACK_CHANNEL_ID"),        # ← replace with real channel ID
            "ts": "1234567890.123456",
            "text": "Hey can someone assign a task to alice to fix the login bug"
        }
    })
}

result1 = handler(step1_event, None)
print(f"Result: {json.dumps(result1, indent=2)}")
print("✅ Slack preview card posted — check your Slack channel")

# Small delay to mimic user reading the card
print("\nWaiting 3 seconds (simulating user reading the card)...")
time.sleep(3)

print("\n" + "=" * 50)
print("STEP 2: User clicks ✅ Create Ticket...")
print("=" * 50)

step2_event = {
    "headers": {},
    "isBase64Encoded": False,
    "body": "payload=" + urllib.parse.quote_plus(json.dumps({
        "type": "block_actions",
        "actions": [
            {
                "action_id": "create_jira",
                "value": json.dumps({
                    "s": "Fix the login bug",
                    "a": "alice"             # ← must match key in TEAM_MAP_JSON
                })
            }
        ],
        "channel": {"id": os.getenv("SLACK_CHANNEL_ID")},   # ← replace with real channel ID
        "container": {"message_ts": "1234567890.123456"}
    }))
}

result2 = handler(step2_event, None)
print(f"Result: {json.dumps(result2, indent=2)}")
print("✅ Jira ticket created — Slack card updated with ticket link")

