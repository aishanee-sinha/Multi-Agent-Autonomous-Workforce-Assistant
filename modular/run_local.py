import json
from dotenv import load_dotenv
load_dotenv()  # must be before any other import

from orchestrator import handler

# --- Simulate a Slack message event ---
fake_event = {
    "headers": {},
    "isBase64Encoded": False,
    "body": json.dumps({
        "event": {
            "channel": "all-slack-api-test",
            "type": "message",
            "text": "Hey can someone assign a task to alice to fix the login bug"
        }
    })
}

result = handler(fake_event, None)
print(json.dumps(result, indent=2))