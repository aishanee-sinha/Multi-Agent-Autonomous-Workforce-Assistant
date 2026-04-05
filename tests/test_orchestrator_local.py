#!/usr/bin/env python3
"""
Local test harness for orchestrator workflow.
Run this to test the full pipeline without deploying to Lambda.
"""

import json
import logging
import os
import sys
import dotenv
dotenv.load_dotenv()  # Load .env file for local testing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

from orchestrator_cod import handler

# For local testing, ensure OPENAI_API_KEY is set
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  OPENAI_API_KEY not set. Using mock key for testing.")
    print("   Set it via: export OPENAI_API_KEY='your-key'")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Test Events
# ─────────────────────────────────────────────────────────────────────────────

# Test 1: Slack message asking to create a Jira task
slack_jira_event = {
    "headers": {
        "x-slack-request-id": "test-slack-jira-001",
    },
    "body": json.dumps({
        "event": {
            "type": "message",
            "text": "Hey, can you create a Jira ticket to fix the login bug?",
            "channel": "C12345",
            "ts": "1234567890.123456",
            "user": "U98765",
        }
    }),
}

# Test 2: Email meeting scheduling request
email_meeting_event = {
    "headers": {
        "x-slack-request-id": "test-email-meeting-001",
    },
    "body": json.dumps({
        "from_email": "client@example.com",
        "subject": "Schedule Q1 Planning Meeting",
        "body": "Can we schedule a meeting next week to discuss the Q1 roadmap?",
        "snippet": "Can we schedule a meeting next week to discuss the Q1 roadmap?",
    }),
}

# Test 3: Ambiguous Slack message (tests chain-of-debate)
ambiguous_event = {
    "headers": {
        "x-slack-request-id": "test-ambiguous-001",
    },
    "body": json.dumps({
        "event": {
            "type": "message",
            "text": "Let's sync up tomorrow",
            "channel": "C12345",
            "ts": "1234567890.123456",
            "user": "U98765",
        }
    }),
}

# Test 4: Non-actionable message
non_actionable_event = {
    "headers": {
        "x-slack-request-id": "test-non-actionable-001",
    },
    "body": json.dumps({
        "event": {
            "type": "message",
            "text": "Let's grab a cup of coffee sometime!",
            "channel": "C12345",
            "ts": "1234567890.123456",
            "user": "U98765",
        }
    }),
}

# Test 5: URL verification (Slack setup)
slack_url_verification_event = {
    "headers": {
        "x-slack-request-id": "test-url-verify-001",
    },
    "body": json.dumps({
        "type": "url_verification",
        "challenge": "3eZbrw1aBcUqE4GZ50Oy8Q",
        "team_id": "T12345",
        "api_app_id": "A12345",
        "event": {},
        "authed_users": [],
        "event_id": "Ev12345",
        "event_time": 1234567890,
    }),
}

# ─────────────────────────────────────────────────────────────────────────────
# Test Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_test(name, event, description=""):
    """Run a single test and print results."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    if description:
        print(f"DESC: {description}")
    print(f"{'='*80}")
    
    try:
        response = handler(event, context=None)
        print(f"✅ Status Code: {response.get('statusCode')}")
        print(f"📋 Response Body:\n{response.get('body', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# LLM Connection Check
# ─────────────────────────────────────────────────────────────────────────────

def check_openai_connection():
    """Verify OpenAI API key and connectivity."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("\n❌ OPENAI_API_KEY not set!")
        print("   Set it via: export OPENAI_API_KEY='sk-...'")
        return False
    
    try:
        import urllib.request
        payload = json.dumps({
            "model": "gpt-4-turbo",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "test"}],
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "content-type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if "choices" in result:
                print("✅ OpenAI API connection successful!")
                return True
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("\n❌ OpenAI API authentication failed (401)!")
            print("   Check your OPENAI_API_KEY is valid.")
        else:
            print(f"\n❌ OpenAI API error (HTTP {e.code}): {e}")
        return False
    except Exception as e:
        print(f"\n❌ OpenAI connection failed: {type(e).__name__}: {e}")
        return False


def check_router_llm_connection():
    """Verify EC2-hosted router LLM endpoint connectivity."""
    ec2_ip = os.environ.get("EC2_IP")

    if not ec2_ip:
        print("\n❌ EC2_IP not set!")
        print("   Set it via: export EC2_IP='x.x.x.x'")
        return False

    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"http://{ec2_ip}:8000/v1/models",
            headers={"Authorization": "Bearer none"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if "data" in body:
                print("✅ Router LLM (EC2) connection successful!")
                return True

        print("\n❌ Router LLM endpoint responded unexpectedly.")
        return False
    except urllib.error.HTTPError as e:
        print(f"\n❌ Router LLM API error (HTTP {e.code}): {e}")
        return False
    except Exception as e:
        print(f"\n❌ Router LLM connection failed: {type(e).__name__}: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("🚀 ORCHESTRATOR LOCAL TEST SUITE")
    print("="*80)
    print("\nTesting chain-of-debate + routing workflow...\n")

    # Check LLM connection first
    print("Checking OpenAI API connection...")
    if not check_openai_connection():
        print("\n❌ Aborting tests — OpenAI API unavailable.")
        sys.exit(1)

    print("Checking router LLM (EC2) connection...")
    if not check_router_llm_connection():
        print("\n❌ Aborting tests — router LLM unavailable.")
        sys.exit(1)

    tests = [
        ("Slack Jira Task", slack_jira_event, "Clear Slack message asking to create Jira task"),
        ("Email Meeting", email_meeting_event, "Clear email asking to schedule a meeting"),
        ("Ambiguous Message", ambiguous_event, "Ambiguous message that needs debate resolution"),
        ("Non-actionable", non_actionable_event, "General greeting, should route to 'none'"),
        # ("URL Verification", slack_url_verification_event, "Slack event subscription verification"),
    ]
    
    for name, event, desc in tests:
        run_test(name, event, desc)
    
    print(f"\n{'='*80}")
    print("✅ All tests completed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
