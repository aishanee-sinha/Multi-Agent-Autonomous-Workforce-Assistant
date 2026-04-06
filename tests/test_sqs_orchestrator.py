"""
test_sqs_orchestrator.py
========================
Fetches the latest message from SQS and runs it through the full
production graph in calendar_cod.py:

    parse_input → router_agent
        ├─ slack_subgraph   (extract → Jira preview → create ticket)
        └─ calendar_subgraph (fetch email → classify → CoD → calendar preview → create event)

Mirrors exactly what sqs_handler does in production.

Run:
    ../llm_venv/Scripts/python test_sqs_orchestrator.py
    ../llm_venv/Scripts/python test_sqs_orchestrator.py --delete   # delete from queue after
    ../llm_venv/Scripts/python test_sqs_orchestrator.py --dry-run  # skip SSM checkpoint update

Required in .env:
    SQS_QUEUE_URL + all orchestrator env vars (EC2_IP, SLACK_BOT_TOKEN, etc.)
"""

import argparse, base64, json, logging, os, sys, urllib.parse
from uuid import uuid4
import boto3
from dotenv import load_dotenv

load_dotenv()

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--delete",  action="store_true", help="Delete message from queue after processing")
parser.add_argument("--dry-run", action="store_true", help="For Pub/Sub: skip SSM checkpoint update")
args = parser.parse_args()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("test_sqs_orchestrator")

SQS_QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "")

print("=" * 60)
print(f"SQS_QUEUE_URL : {SQS_QUEUE_URL or '❌ NOT SET'}")
print(f"--delete      : {args.delete}  |  --dry-run: {args.dry_run}")
print("=" * 60)

if not SQS_QUEUE_URL:
    print("❌ SQS_QUEUE_URL missing from .env"); raise SystemExit(1)

# ── [1] Fetch from SQS ───────────────────────────────────────────────────────
print("\n[1] Fetching latest message from SQS...")
_sqs = boto3.client("sqs")
try:
    resp = _sqs.receive_message(
        QueueUrl            = SQS_QUEUE_URL,
        MaxNumberOfMessages = 1,
        WaitTimeSeconds     = 5,
        VisibilityTimeout   = 60,
    )
except Exception as e:
    print(f"❌ Cannot reach SQS: {e}"); raise SystemExit(1)

msgs = resp.get("Messages", [])
if not msgs:
    print("❌ Queue is empty — nothing to process"); raise SystemExit(0)

msg = msgs[0]
print(f"✅ Got SQS message  MessageId={msg['MessageId']}")

# ── [2] Decode and display the message ───────────────────────────────────────
try:
    http_event = json.loads(msg["Body"])
    raw_body   = http_event.get("body", "")
    if http_event.get("isBase64Encoded") and raw_body:
        raw_body = base64.b64decode(raw_body).decode("utf-8")
except Exception as e:
    print(f"❌ Could not decode SQS body: {e}"); raise SystemExit(1)

print("\n[2] Message type and contents:")
print("-" * 60)

pubsub_history_id = None

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
        print(f"  (not JSON): {raw_body[:300]}"); raise SystemExit(0)

    if body.get("type") == "event_callback":
        slack_evt = body.get("event", {})
        print(f"  Type     : Slack message → will route to slack_subgraph")
        print(f"  event_id : {body.get('event_id')}")
        print(f"  channel  : {slack_evt.get('channel')}")
        print(f"  user     : {slack_evt.get('user')}")
        print(f"  is_bot   : {'yes' if slack_evt.get('bot_id') else 'no'}")
        print(f"  text     :\n    {slack_evt.get('text', '')}")

    elif body.get("message", {}).get("data"):
        pubsub_msg        = body["message"]
        decoded           = json.loads(base64.b64decode(pubsub_msg["data"]).decode())
        pubsub_history_id = str(decoded.get("historyId", ""))
        print(f"  Type         : Gmail Pub/Sub → will route to calendar_subgraph (CoD)")
        print(f"  emailAddress : {decoded.get('emailAddress')}")
        print(f"  historyId    : {pubsub_history_id}")
        from gmail_history import get_last_history_id
        last_hid = get_last_history_id()
        print(f"  SSM checkpoint   : {last_hid or '(not set)'}")
        print(f"  Will query since : {last_hid or str(int(pubsub_history_id) - 10)}")
    else:
        print(f"  Type : unknown — {body.get('type')}")

print("-" * 60)

# ── [3] Load the full graph from calendar_cod ─────────────────────────────────
print("\n[3] Loading calendar_cod graph (parse_input → router → slack/calendar+CoD)...")
try:
    from calendar_cod import handler
    print("✅ Graph loaded")
except Exception as e:
    print(f"❌ Failed to load graph: {e}"); raise SystemExit(1)

# ── [4] Run — identical to what sqs_handler does in production ────────────────
print("\n[4] Running graph (this may take 20-40s for CoD)...")
print("-" * 60)

try:
    result = handler(http_event, None)
except Exception as e:
    print(f"\n❌ handler raised: {e}")
    logger.exception("handler failed")
    raise SystemExit(1)

print("-" * 60)
print(f"\n[5] Handler response: {result}")

# Advance SSM for Pub/Sub
if pubsub_history_id and not args.dry_run:
    from gmail_history import set_last_history_id
    set_last_history_id(pubsub_history_id)
    print(f"    SSM checkpoint advanced → {pubsub_history_id}")

# ── [6] Delete from queue ─────────────────────────────────────────────────────
print()
if args.delete:
    _sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=msg["ReceiptHandle"])
    print("✅ Message deleted from queue")
else:
    print("Message left in queue (run with --delete to remove it)")

print("\n✅ Done")
