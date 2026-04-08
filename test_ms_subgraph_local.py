#!/usr/bin/env python3
"""
test_ms_subgraph_local.py  —  Stream B local test
===================================================
Tests the full meeting_agent.py LangGraph subgraph with all external
calls mocked. No EC2, no AWS, no Slack needed.

Run (from project root with venv active):
    python test_ms_subgraph_local.py

Tests:
  1. New transcript flow  — full pipeline: fetch → preprocess → summarize
                            → triage_cod → artifacts → s3 → slack
  2. Confirm button       — meeting_send_email invoked, SES called
  3. Cancel button        — meeting_post_cancel invoked, Slack updated
  4. Duplicate trigger    — lock held → processing skipped silently
  5. parse_input routing  — new_transcript and button events route correctly
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Set env vars before any imports that read them at module level ────────────
os.environ.setdefault("EC2_IP",                   "mock-ec2")
os.environ.setdefault("SLACK_BOT_TOKEN",           "xoxb-mock")
os.environ.setdefault("SLACK_NOTIFY_CHANNEL",      "C_MOCK_CHANNEL")
os.environ.setdefault("S3_BUCKET",                 "mock-bucket")
os.environ.setdefault("S3_TRANSCRIPT_PREFIX",      "transcript_summarizer")
os.environ.setdefault("VLLM_MODEL_NAME",           "meeting")
os.environ.setdefault("SES_FROM_EMAIL",            "test@example.com")
os.environ.setdefault("PARTICIPANT_EMAILS",        "a@gmail.com,b@gmail.com")
os.environ.setdefault("WEBHOOK_SECRET",            "meeting-summarizer-secret-2026")
os.environ.setdefault("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
os.environ.setdefault("TEAM_MAP_JSON",             "{}")
os.environ.setdefault("GROUP_EMAILS_JSON",         "[]")
os.environ.setdefault("CALENDAR_TOKENS_JSON",      "{}")

from unittest.mock import patch, MagicMock

# ── Python 3.9 compatibility fix ─────────────────────────────────────────────
# calendar_agent.py uses `str | None` union syntax (Python 3.10+).
# Mock it before importing calendar_cod so the import chain doesn't fail.
# Safe — this test only exercises meeting_agent and parse_input routing.
_mock_cal = MagicMock()
sys.modules["calendar_agent"] = _mock_cal
for _name in ["email_fetch_and_parse", "email_classify", "email_post_slack_preview",
              "email_create_calendar", "email_post_cancel",
              "route_email_entry", "route_after_classify"]:
    setattr(_mock_cal, _name, MagicMock())
# ─────────────────────────────────────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver

from state import OrchestratorState
import meeting_agent as _ma
from calendar_cod import parse_input   # merged from orchestrator.py

# ── Fake transcript and model output ─────────────────────────────────────────

FAKE_TRANSCRIPT = """[Alice]: Good morning. Let's start the Q2 planning meeting at 9am PDT.
[Bob]: We should prioritise the mobile redesign. I think we defer desktop to Q3.
[Alice]: Agreed — mobile beta by May 15th. That's the decision.
[Carlos]: I'll handle backend API updates. Done by end of next week.
[Alice]: We have a problem — frontend team is two engineers short.
[Bob]: That's blocking the mobile deadline. Critical risk.
[Alice]: DevOps, patch the auth vulnerability by EOD today — it's blocking production.
[DevOps]: On it. EOD PDT."""

FAKE_SUMMARY = """ABSTRACT:
The team aligned on Q2 priorities: mobile redesign first, desktop deferred to Q3.
Critical auth vulnerability patch required immediately.

DECISIONS:
- Launch mobile beta by May 15th
- Defer desktop feature to Q3

PROBLEMS:
- Frontend team understaffed by 2 engineers — risk to mobile deadline
- Auth vulnerability blocking production deployment

ACTIONS:
[Carlos] - Backend API updates - Due: end of next week
[DevOps] - Patch auth vulnerability - Due: EOD today

ACTIONS_JSON:
[
  {"owner": "Carlos", "task": "Backend API updates", "deadline": "end of next week", "discussed_at_sec": 120.0},
  {"owner": "DevOps", "task": "Patch auth vulnerability", "deadline": "EOD today", "discussed_at_sec": 240.0}
]"""


# ── Mock factories ────────────────────────────────────────────────────────────

def _mock_vllm(content: str):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def _make_s3_no_lock():
    """S3 mock: no existing lock (404 on get_object)."""
    from botocore.exceptions import ClientError
    s3 = MagicMock()
    s3.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
    )
    s3.put_object.return_value = {}
    return s3


def _make_s3_active_lock():
    """S3 mock: active lock (age < TTL)."""
    s3 = MagicMock()
    body = json.dumps({"ts": time.time(), "status": "processing"}).encode()
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: body)}
    return s3


def _make_s3_with_meta(meta: dict):
    """S3 mock: returns meta.json on get_object (for email send)."""
    from botocore.exceptions import ClientError
    s3 = MagicMock()
    meta_body = json.dumps(meta).encode()

    def smart_get(**kwargs):
        key = kwargs.get("Key", "")
        if key.endswith("meta.json"):
            return {"Body": MagicMock(read=lambda: meta_body)}
        raise ClientError({"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject")

    s3.get_object.side_effect = smart_get
    return s3


# Canned CoD structured responses for mocking _llm()
class FakeProposal:
    priority = "Critical"; risk = "Production is blocked."; deadline_ok = True
    argument = "Auth vulnerability blocks all deployments."

class FakeChallenge:
    agrees = True; counter_priority = "Critical"
    argument = "Agree — auth patch is clearly Critical given the context."

class FakeVerdict:
    final_priority = "Critical"; risk_summary = "Production blocked if missed."
    deadline_note  = "On track"; rationale = "Unambiguously blocking production."

class FakeMediumProposal:
    priority = "Medium"; risk = "Backend updates delay mobile launch."; deadline_ok = True
    argument = "Backend APIs are needed but not an immediate blocker today."

class FakeMediumVerdict:
    final_priority = "Medium"; risk_summary = "Delays mobile beta if missed."
    deadline_note  = "On track"; rationale = "Important but not immediately blocking."


# ── Test helpers ──────────────────────────────────────────────────────────────

PASS = 0; FAIL = 0

def check(condition: bool, description: str):
    global PASS, FAIL
    marker = "  PASS" if condition else "  FAIL"
    print(f"{marker}  {description}")
    if condition: PASS += 1
    else:         FAIL += 1


def _build_subgraph():
    return _ma.build_meeting_subgraph()


def _invoke(graph, state_kwargs: dict, thread_id: str = "test") -> OrchestratorState:
    initial = OrchestratorState(raw_event={}, **state_kwargs)
    config  = {"configurable": {"thread_id": thread_id}}
    result  = graph.invoke(initial, config=config)
    return OrchestratorState(**result) if isinstance(result, dict) else result


# ── Test 1: Full new-transcript pipeline ──────────────────────────────────────

def test_full_new_transcript_flow():
    print("\n[Test 1] Full new-transcript pipeline")
    graph = _build_subgraph()

    s3_mock    = _make_s3_no_lock()
    slack_mock = MagicMock()
    slack_mock.chat_postMessage.return_value = {"ts": "111.222"}

    # CoD mock: alternates between proposals/challenges/verdicts
    call_counts = {"proposal": 0, "challenge": 0, "verdict": 0}

    def fake_llm_invoke(messages):
        # Detect which CoD round by message content
        sys_content = next((m.content for m in messages if hasattr(m, 'content') and
                            "Proposer" in m.content and "propose" in m.content.lower()), "")
        if "Proposer" in str(messages[0].content if hasattr(messages[0], 'content') else ""):
            call_counts["proposal"] += 1
            return FakeProposal() if call_counts["proposal"] % 2 == 1 else FakeMediumProposal()
        if "Challenger" in str(messages[0].content if hasattr(messages[0], 'content') else ""):
            return FakeChallenge()
        return FakeVerdict() if call_counts["verdict"] % 2 == 0 else FakeMediumVerdict()

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke = fake_llm_invoke

    # vLLM response counter for smart dispatch
    vllm_calls = [0]

    def fake_vllm_post(url, **kwargs):
        vllm_calls[0] += 1
        return _mock_vllm(FAKE_SUMMARY)

    with patch("meeting_agent.boto3.client", return_value=s3_mock), \
         patch("meeting_agent._download_from_drive", return_value=FAKE_TRANSCRIPT), \
         patch("meeting_agent.requests.post", side_effect=fake_vllm_post), \
         patch("meeting_agent._llm", return_value=mock_llm_instance), \
         patch("meeting_agent.WebClient", return_value=slack_mock):

        final = _invoke(graph, {
            "intent":               "meeting_transcript",
            "transcript_file_id":   "file_test_001",
            "transcript_file_name": "project_meeting.txt",
        }, thread_id="test-new-transcript")

    # State assertions
    check(final.transcript_text is not None,       "transcript_text set after download")
    check(final.transcript_processed is not None,  "transcript_processed set after preprocess")
    check(final.meeting_summary_parsed is not None,"meeting_summary_parsed set after summarize")
    check(final.meeting_triage is not None,        "meeting_triage set after CoD")
    check(len(final.meeting_triage) == 2,          f"2 action items triaged (got {len(final.meeting_triage or [])})")
    check(final.meeting_s3_key is not None,        "meeting_s3_key set after S3 store")
    check(final.meeting_ics_bytes is not None,     "ICS bytes generated")
    check(final.meeting_csv_bytes is not None,     "CSV bytes generated")

    # Summary content
    parsed = final.meeting_summary_parsed
    check(parsed.get("abstract", "") != "",        "Abstract non-empty")
    check(len(parsed.get("actions_json", [])) == 2,"2 action items parsed")
    check("Carlos" in parsed.get("actions", ""),   "Carlos action in output")
    check("DevOps" in parsed.get("actions", ""),   "DevOps action in output")

    # ICS UTC check (FIX-3)
    ics_text = final.meeting_ics_bytes.decode("utf-8")
    dtstart  = [l for l in ics_text.splitlines() if l.startswith("DTSTART:")][0]
    check(dtstart.endswith("Z"), f"DTSTART ends with Z (UTC): {dtstart}")

    # Triage priorities present
    priorities = [t.get("final_priority") for t in (final.meeting_triage or [])]
    check(any(p in ("Critical", "High", "Medium", "Low") for p in priorities),
          f"Triage priorities valid: {priorities}")

    # Slack posted once
    slack_mock.chat_postMessage.assert_called_once()
    call_kwargs = slack_mock.chat_postMessage.call_args[1]
    check(call_kwargs["channel"] == os.environ["SLACK_NOTIFY_CHANNEL"],
          "Slack posted to correct channel")

    # Triage emoji in Slack blocks
    blocks_text = json.dumps(call_kwargs.get("blocks", []), ensure_ascii=False)
    check(any(emoji in blocks_text for emoji in ("🔴","🟠","🟡","🟢")),
          "Priority emoji present in Slack blocks")

    # S3 stored multiple objects
    check(s3_mock.put_object.call_count >= 3,
          f"S3 put_object called ≥3 times (got {s3_mock.put_object.call_count})")


# ── Test 2: Confirm button → email sent ───────────────────────────────────────

def test_confirm_button_sends_email():
    print("\n[Test 2] Confirm button → SES email")
    graph = _build_subgraph()

    meta = {
        "summary_text": "Meeting about Q2 planning.",
        "triage": [{"owner":"DevOps","task":"Patch auth","final_priority":"Critical",
                    "risk_summary":"Blocks prod.","deadline_note":"On track"}],
    }
    s3_mock  = _make_s3_with_meta(meta)
    ses_mock = MagicMock()
    slack_mock = MagicMock()

    with patch("meeting_agent.boto3.client") as mock_boto, \
         patch("meeting_agent.WebClient", return_value=slack_mock):
        mock_boto.side_effect = lambda svc, **kw: ses_mock if svc == "ses" else s3_mock
        final = _invoke(graph, {
            "intent":             "meeting_transcript",
            "slack_action_id":    "confirm_summary",
            "slack_action_value": {"file_name": "test.txt",
                                   "s3_key":    "transcript_summarizer/meetings/ts_test"},
            "channel_id":         "C_MOCK_CHANNEL",
            "preview_ts":         "111.222",
        }, thread_id="test-confirm")

    ses_mock.send_raw_email.assert_called_once()
    check(True, "SES send_raw_email called once")

    slack_mock.chat_update.assert_called_once()
    update_text = slack_mock.chat_update.call_args[1].get("text", "")
    check("confirmed" in update_text.lower() or "sent" in update_text.lower(),
          f"Slack updated with confirmation: '{update_text[:60]}'")


# ── Test 3: Cancel button → Slack dismissed ───────────────────────────────────

def test_cancel_button_dismisses():
    print("\n[Test 3] Cancel button → Slack dismissed, no email")
    graph = _build_subgraph()

    slack_mock = MagicMock()
    ses_mock   = MagicMock()

    with patch("meeting_agent.boto3.client", return_value=ses_mock), \
         patch("meeting_agent.WebClient", return_value=slack_mock):
        _invoke(graph, {
            "intent":          "meeting_transcript",
            "slack_action_id": "cancel_summary",
            "channel_id":      "C_MOCK_CHANNEL",
            "preview_ts":      "333.444",
        }, thread_id="test-cancel")

    slack_mock.chat_update.assert_called_once()
    ses_mock.send_raw_email.assert_not_called()
    update_text = slack_mock.chat_update.call_args[1].get("text", "")
    check("dismissed" in update_text.lower() or "no email" in update_text.lower(),
          f"Slack updated with dismissal: '{update_text[:60]}'")
    check(True, "SES not called on cancel")


# ── Test 4: Duplicate trigger → skipped ───────────────────────────────────────

def test_duplicate_trigger_skipped():
    print("\n[Test 4] Duplicate trigger → S3 lock held → skip")
    graph = _build_subgraph()

    s3_mock    = _make_s3_active_lock()
    drive_mock = MagicMock()
    slack_mock = MagicMock()

    with patch("meeting_agent.boto3.client", return_value=s3_mock), \
         patch("meeting_agent._download_from_drive", side_effect=drive_mock), \
         patch("meeting_agent.WebClient", return_value=slack_mock):
        final = _invoke(graph, {
            "intent":               "meeting_transcript",
            "transcript_file_id":   "file_locked",
            "transcript_file_name": "locked.txt",
        }, thread_id="test-duplicate")

    drive_mock.assert_not_called()
    check(True, "Drive download NOT called when lock is active")
    check(final.error == "duplicate_trigger",
          f"error='duplicate_trigger' in state (got '{final.error}')")
    slack_mock.chat_postMessage.assert_not_called()
    check(True, "Slack NOT posted for duplicate trigger")


# ── Test 5: parse_input routing ───────────────────────────────────────────────

def test_parse_input_routing():
    print("\n[Test 5] parse_input correctly routes meeting events")

    # new_transcript → meeting_transcript
    state = OrchestratorState(raw_event={
        "headers": {},
        "body": json.dumps({
            "type":      "new_transcript",
            "file_id":   "file_abc",
            "file_name": "meeting.txt",
            "secret":    "meeting-summarizer-secret-2026",
        }),
    })
    result = parse_input(state)
    check(result.intent == "meeting_transcript",
          f"new_transcript → intent=meeting_transcript (got '{result.intent}')")
    check(result.transcript_file_id == "file_abc",
          f"transcript_file_id='file_abc' (got '{result.transcript_file_id}')")

    # Wrong secret → intent=none
    state2 = OrchestratorState(raw_event={
        "headers": {},
        "body": json.dumps({
            "type": "new_transcript", "file_id": "x",
            "file_name": "x.txt", "secret": "wrong-secret",
        }),
    })
    result2 = parse_input(state2)
    check(result2.intent == "none",
          f"Wrong secret → intent=none (got '{result2.intent}')")

    # confirm_summary button → meeting_transcript
    import urllib.parse
    payload = {
        "actions": [{"action_id": "confirm_summary",
                     "value": json.dumps({"file_name":"f.txt","s3_key":"s3/key"})}],
        "channel":   {"id": "C123"},
        "container": {"message_ts": "111.222"},
    }
    body_raw = "payload=" + urllib.parse.quote_plus(json.dumps(payload))
    state3   = OrchestratorState(raw_event={"headers": {}, "body": body_raw})
    result3  = parse_input(state3)
    check(result3.intent == "meeting_transcript",
          f"confirm_summary → intent=meeting_transcript (got '{result3.intent}')")
    check(result3.slack_action_id == "confirm_summary",
          f"slack_action_id='confirm_summary' (got '{result3.slack_action_id}')")

    # cancel_summary button → meeting_transcript
    payload2 = {
        "actions": [{"action_id": "cancel_summary", "value": "{}"}],
        "channel": {"id": "C123"}, "container": {"message_ts": "222.333"},
    }
    body_raw2 = "payload=" + urllib.parse.quote_plus(json.dumps(payload2))
    state4    = OrchestratorState(raw_event={"headers": {}, "body": body_raw2})
    result4   = parse_input(state4)
    check(result4.intent == "meeting_transcript",
          f"cancel_summary → intent=meeting_transcript (got '{result4.intent}')")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("test_ms_subgraph_local.py  —  Stream B local test")
    print("=" * 60)

    test_full_new_transcript_flow()
    test_confirm_button_sends_email()
    test_cancel_button_dismisses()
    test_duplicate_trigger_skipped()
    test_parse_input_routing()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed  |  {FAIL} failed")
    if FAIL:
        print("\nFix failing tests before deploying Stream B to Lambda.")
        sys.exit(1)
    else:
        print("All subgraph tests passed — ready for Lambda deployment.")
    print("=" * 60)
