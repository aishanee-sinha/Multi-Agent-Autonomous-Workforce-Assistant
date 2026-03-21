import json, urllib.parse
from dotenv import load_dotenv
load_dotenv()

from orchestrator import handler

fake_event = {
    "headers": {},
    "isBase64Encoded": False,
    "body": "payload=" + urllib.parse.quote_plus(json.dumps({
        "type": "block_actions",
        "actions": [
            {
                "action_id": "create_jira",
                "value": json.dumps({
                    "s": "Fix the login bug",
                    "a": "alice"
                })
            }
        ],
        "channel": {"id": "C09CCHY0T8S"},
        "container": {"message_ts": "1234567890.123456"}
    }))
}

result = handler(fake_event, None)
print(json.dumps(result, indent=2))