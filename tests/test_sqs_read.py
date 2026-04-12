"""
test_sqs_read.py
================
Fetches the latest message from SQS, detects its type, and displays it:
  - Slack message event    → prints channel, user, text
  - Slack button click     → prints action, user, value
  - Gmail Pub/Sub          → fetches and prints the actual email from Gmail

Does NOT run the orchestrator or post anything to Slack.

Run:
    ../llm_venv/Scripts/python test_sqs_read.py           # read only
    ../llm_venv/Scripts/python test_sqs_read.py --delete  # read and delete from queue
    ../llm_venv/Scripts/python test_sqs_read.py --dry-run # for pubsub: skip SSM update

Required in .env:
    SQS_QUEUE_URL, GOOGLE_TOKEN_JSON (only needed for Pub/Sub messages)
    SSM_HISTORY_ID_PARAM (optional, default: /agent/gmail_history_id)
"""

import argparse, base64, json, logging, os, sys, urllib.parse
import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,  # suppress boto3 noise; script uses print
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)

parser = argparse.ArgumentParser()
parser.add_argument("--delete",  action="store_true", help="Delete message from queue after reading")
parser.add_argument("--dry-run", action="store_true", help="For Pub/Sub: do not advance SSM checkpoint")
args = parser.parse_args()

SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")
SSM_PARAM     = os.environ.get("SSM_HISTORY_ID_PARAM", "/agent/gmail_history_id")

print("=" * 60)
print(f"SQS_QUEUE_URL : {SQS_QUEUE_URL or '❌ NOT SET'}")
print(f"SSM_PARAM     : {SSM_PARAM}")
print(f"--delete      : {args.delete}  |  --dry-run: {args.dry_run}")
print("=" * 60)

if not SQS_QUEUE_URL:
    print("❌ SQS_QUEUE_URL missing from .env"); raise SystemExit(1)

# ── Fetch from SQS ───────────────────────────────────────────────────────────
print("\n[1] Fetching latest message from SQS...")
_sqs = boto3.client("sqs")
try:
    resp = _sqs.receive_message(
        QueueUrl            = SQS_QUEUE_URL,
        MaxNumberOfMessages = 1,
        WaitTimeSeconds     = 5,
        VisibilityTimeout   = 30,
    )
except Exception as e:
    print(f"❌ Cannot reach SQS: {e}"); raise SystemExit(1)

msgs = resp.get("Messages", [])
if not msgs:
    print("❌ Queue is empty"); raise SystemExit(0)

msg = msgs[0]
print(f"✅ Got SQS message  MessageId={msg['MessageId']}")

# ── Decode HTTP event wrapper ─────────────────────────────────────────────────
try:
    http_event = json.loads(msg["Body"])
    raw_body   = http_event.get("body", "")
    if http_event.get("isBase64Encoded") and raw_body:
        raw_body = base64.b64decode(raw_body).decode("utf-8")
except Exception as e:
    print(f"❌ Could not decode SQS body: {e}")
    print(f"   Raw: {msg['Body'][:500]}"); raise SystemExit(1)

# ── Detect and display ────────────────────────────────────────────────────────
print("\n[2] Message type and contents:")
print("-" * 60)

# ── Slack interactivity (button click) ───────────────────────────────────────
if raw_body.startswith("payload="):
    payload = json.loads(urllib.parse.unquote_plus(raw_body.split("payload=")[1]))
    action  = payload.get("actions", [{}])[0]
    user    = payload.get("user", {})
    print(f"  Type       : Slack button click")
    print(f"  action_id  : {action.get('action_id')}")
    print(f"  user       : {user.get('name')} ({user.get('id')})")
    print(f"  channel    : {payload.get('channel', {}).get('id')}")
    print(f"  message_ts : {payload.get('container', {}).get('message_ts')}")
    print(f"  value      : {action.get('value', '')[:300]}")

else:
    try:
        body = json.loads(raw_body)
    except Exception:
        print(f"  (not JSON):\n  {raw_body[:500]}"); raise SystemExit(0)

    # ── Slack message event ───────────────────────────────────────────────────
    if body.get("type") == "event_callback":
        slack_evt = body.get("event", {})
        print(f"  Type     : Slack message")
        print(f"  event_id : {body.get('event_id')}")
        print(f"  team_id  : {body.get('team_id')}")
        print(f"  channel  : {slack_evt.get('channel')}")
        print(f"  user     : {slack_evt.get('user')}")
        print(f"  is_bot   : {'yes' if slack_evt.get('bot_id') else 'no'}")
        print(f"  ts       : {slack_evt.get('ts')}")
        print(f"  text     :\n    {slack_evt.get('text', '')}")

    # ── Gmail Pub/Sub ─────────────────────────────────────────────────────────
    elif body.get("message", {}).get("data"):
        pubsub_msg        = body["message"]
        decoded           = json.loads(base64.b64decode(pubsub_msg["data"]).decode())
        pubsub_history_id = str(decoded.get("historyId", ""))
        email_address     = decoded.get("emailAddress", "")

        print(f"  Type         : Gmail Pub/Sub")
        print(f"  emailAddress : {email_address}")
        print(f"  historyId    : {pubsub_history_id}")
        print(f"  messageId    : {pubsub_msg.get('messageId')}")

        if not os.environ.get("GOOGLE_TOKEN_JSON"):
            print("\n⚠️  GOOGLE_TOKEN_JSON not set — skipping Gmail fetch")
        else:
            from gmail_history import get_last_history_id, set_last_history_id
            from calendar_agent import _get_google_creds, _parse_gmail_message
            from googleapiclient.discovery import build

            last_history_id  = get_last_history_id()
            query_history_id = last_history_id or str(int(pubsub_history_id) - 10)

            print(f"\n  SSM checkpoint   : {last_history_id or '(not set, using pubsub-10)'}")
            print(f"  Querying since   : {query_history_id}")

            creds   = _get_google_creds()
            service = build("gmail", "v1", credentials=creds)

            profile = service.users().getProfile(userId="me").execute()
            print(f"  Token account    : {profile.get('emailAddress')}")
            if email_address and profile.get("emailAddress") != email_address:
                print(f"  ⚠️  ACCOUNT MISMATCH — Pub/Sub={email_address} Token={profile.get('emailAddress')}")

            history = service.users().history().list(
                userId         = "me",
                startHistoryId = query_history_id,
                historyTypes   = ["messageAdded"],
                labelId        = "INBOX",
            ).execute()

            records = history.get("history", [])
            print(f"  History records  : {len(records)}")

            emails = []
            for record in records:
                for msg_added in record.get("messagesAdded", []):
                    mid    = msg_added["message"]["id"]
                    labels = msg_added["message"].get("labelIds", [])
                    print(f"    → messageId={mid}  labels={labels}")
                    full   = service.users().messages().get(
                        userId="me", id=mid, format="full"
                    ).execute()
                    emails.append(_parse_gmail_message(full))

            if emails:
                print(f"\n  ✅ {len(emails)} email(s) fetched:")
                for i, email in enumerate(emails, 1):
                    print(f"\n  --- Email {i} ---")
                    print(f"  from    : {email.get('from_email')}")
                    print(f"  to      : {email.get('to_emails')}")
                    print(f"  subject : {email.get('subject')}")
                    print(f"  date    : {email.get('date')}")
                    print(f"  body    :\n{email.get('body', '').strip()[:800]}")
            else:
                print("  ❌ No emails found for this historyId window")
                print("     → Gmail history may be too stale, or email skipped INBOX label")

            if not args.dry_run:
                set_last_history_id(pubsub_history_id)
                print(f"\n  SSM checkpoint advanced → {pubsub_history_id}")
            else:
                print(f"\n  Dry run — SSM checkpoint NOT updated (still {last_history_id})")

    else:
        print(f"  Type : unknown")
        print(f"  body : {json.dumps(body, indent=4)[:500]}")

# ── Delete if requested ───────────────────────────────────────────────────────
print("-" * 60)
if args.delete:
    _sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
    print("✅ Message deleted from queue")
else:
    print("Message left in queue (run with --delete to remove it)")
