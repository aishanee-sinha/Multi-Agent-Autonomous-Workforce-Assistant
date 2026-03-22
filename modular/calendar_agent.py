"""
calendar_agent.py — Autonomous Email → Slack approval → Google Calendar
========================================================================
Flow:
  1. email_fetch_and_parse   — fetch email from Gmail via Pub/Sub history ID
  2. email_classify          — Qwen LLM checks for meeting intent
  3. email_post_slack_preview — post ✅ Create Meeting / ❌ Cancel card to Slack
                               (pending_meeting embedded in button value)
  4. [user clicks button]    — Slack sends interactivity payload to Lambda
  5. email_create_calendar   — create Google Calendar event + send invites

Routing within subgraph:
  pubsub/direct → email_fetch_and_parse → email_classify
                    → if meeting  → email_post_slack_preview → END
                    → if no meeting → END

  interactivity (create_meeting) → email_create_calendar → END
  interactivity (cancel_meeting) → email_post_cancel     → END
"""

import base64, json, logging, re
from datetime import datetime, timezone, timedelta
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from state import (
    OrchestratorState,
    GROUP_EMAILS, GOOGLE_TOKEN, SLACK_BOT_TOKEN, SLACK_NOTIFY_CHANNEL,
    _llm,
)

logger = logging.getLogger(__name__)
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def _logged_slack(client: WebClient) -> WebClient:
    """Wrap every Slack API method to log the call and args before executing."""
    original_api_call = client.api_call

    @wraps(original_api_call)
    def logged_api_call(api_method, *args, **kwargs):
        # Log the method name and all arguments
        safe_kwargs = {
            k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
            for k, v in kwargs.items()
            if k not in ("token",)  # never log the token
        }
        logger.info(f"SLACK API CALL  : {api_method}")
        logger.info(f"SLACK API ARGS  : {safe_kwargs}")
        result = original_api_call(api_method, *args, **kwargs)
        logger.info(f"SLACK API RESULT: ok={result.get('ok')} error={result.get('error','none')}")
        return result

    client.api_call = logged_api_call
    return client
# ─────────────────────────────────────────────────────────────────────────────
# LLM system prompt
# ─────────────────────────────────────────────────────────────────────────────
EMAIL_SYSTEM_PROMPT = """You are a specialized meeting information extraction system.
TASK: Analyze emails and extract structured meeting details with high precision.
OUTPUT FORMAT (JSON only, no explanations):
{
  "is_meeting": boolean,
  "title": string or null,
  "attendees": array of email addresses,
  "start_time": ISO 8601 string or null,
  "end_time": ISO 8601 string or null,
  "location": string or null,
  "time_confidence": "high" | "medium" | "low" | "none"
}
CRITICAL: ALWAYS return valid JSON. NEVER add explanations."""


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def email_fetch_and_parse(state: OrchestratorState) -> OrchestratorState:
    """
    For Pub/Sub events — fetch emails from Gmail using history ID.
    For direct invocations — email_data is already set, skip fetch.
    """
    if state.email_source != "pubsub" or not state.email_data:
        return state

    try:
        pubsub_message = state.email_data.get("message", {})
        decoded    = json.loads(base64.b64decode(pubsub_message["data"]).decode("utf-8"))
        history_id = str(decoded.get("historyId", ""))
        emails     = _fetch_new_emails(history_id)
        if emails:
            return state.model_copy(update={"email_data": emails[0]})
        return state.model_copy(update={"intent": "none"})
    except Exception as e:
        logger.error(f"Email fetch error: {e}")
        return state.model_copy(update={"error": str(e)})


def email_classify(state: OrchestratorState) -> OrchestratorState:
    """Qwen LLM classifies the email and extracts meeting details."""
    if not state.email_data:
        return state.model_copy(update={"is_meeting": False})

    ed        = state.email_data
    sender    = ed.get("from_email", "").lower()
    group_set = {e.lower() for e in GROUP_EMAILS}
    if GROUP_EMAILS and sender not in group_set:
        logger.info(f"Sender {sender} not in group — skipping")
        return state.model_copy(update={"is_meeting": False})

    email_text = (
        f"Subject: {ed.get('subject', '')}\n"
        f"From: {ed.get('from_email', '')}\n"
        f"To: {', '.join(ed.get('to_emails', []))}\n"
        f"Cc: {', '.join(ed.get('cc_emails', []))}\n\n"
        f"{ed.get('body', '')[:2000]}"
    )

    llm = _llm()
    try:
        resp = llm.invoke([SystemMessage(content=EMAIL_SYSTEM_PROMPT), HumanMessage(content=email_text)])
        raw  = resp.content if hasattr(resp, "content") else str(resp)
        raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        return state.model_copy(update={
            "is_meeting":        data.get("is_meeting", False),
            "meeting_title":     data.get("title"),
            "meeting_start":     data.get("start_time"),
            "meeting_end":       data.get("end_time"),
            "meeting_location":  data.get("location"),
            "meeting_attendees": data.get("attendees") or [],
            "time_confidence":   data.get("time_confidence", "none"),
        })
    except Exception as e:
        logger.error(f"Email classify error: {e}")
        return state.model_copy(update={"error": str(e), "is_meeting": False})


def email_post_slack_preview(state: OrchestratorState) -> OrchestratorState:
    """
    Post a Slack card with ✅ Create Meeting / ❌ Cancel buttons.
    The full pending_meeting dict is embedded in the button value
    so Lambda can reconstruct it on the next invocation — no DynamoDB needed.
    """
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))

    pending = {
        "email_data": state.email_data,
        "model_output": {
            "title":           state.meeting_title,
            "start_time":      state.meeting_start,
            "end_time":        state.meeting_end,
            "location":        state.meeting_location,
            "attendees":       state.meeting_attendees,
            "time_confidence": state.time_confidence,
        }
    }

    title     = state.meeting_title or "Meeting"
    start     = state.meeting_start or "TBD"
    end       = state.meeting_end   or "TBD"
    location  = state.meeting_location or "N/A"
    attendees = ", ".join(state.meeting_attendees[:3]) or "N/A"
    confidence = state.time_confidence or "none"

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"📧 *Meeting detected in incoming email*\n"
                    f"*Title:*      {title}\n"
                    f"*Start:*      {start}\n"
                    f"*End:*        {end}\n"
                    f"*Location:*   {location}\n"
                    f"*Attendees:*  {attendees}\n"
                    f"*Confidence:* {confidence}"
                ),
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Create Meeting"},
                    "style": "primary",
                    "value": json.dumps(pending),
                    "action_id": "create_meeting",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Cancel"},
                    "style": "danger",
                    "value": "{}",
                    "action_id": "cancel_meeting",
                },
            ],
        },
    ]

    try:
        resp = client.chat_postMessage(
            channel=SLACK_NOTIFY_CHANNEL,
            blocks=blocks,
            text=f"Meeting detected: {title}",
        )
        return state.model_copy(update={"preview_ts": resp["ts"]})
    except SlackApiError as e:
        logger.error(f"Slack post error: {e}")
        return state.model_copy(update={"error": str(e)})


def email_create_calendar(state: OrchestratorState) -> OrchestratorState:
    """
    User clicked ✅ Create Meeting in Slack.
    pending_meeting is read from state (passed from button value by parse_input).
    Creates the Google Calendar event and updates the Slack card.
    """
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))
    pending = state.pending_meeting

    if not pending:
        logger.error("No pending_meeting in state")
        return state.model_copy(update={"error": "No pending meeting data found"})

    model_output   = pending.get("model_output", {})
    original_email = pending.get("email_data", {})

    link = _create_calendar_event(model_output, original_email)

    # Update the Slack preview card with result
    title = model_output.get("title") or "Meeting"
    try:
        if link:
            client.chat_update(
                channel=state.channel_id,
                ts=state.preview_ts,
                text=f"✅ Calendar event created: {title}",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"✅ *Calendar event created:* <{link}|{title}>",
                        },
                    }
                ],
            )
        else:
            client.chat_update(
                channel=state.channel_id,
                ts=state.preview_ts,
                text=f"⚠️ Failed to create calendar event",
                blocks=None,
            )
    except SlackApiError as e:
        logger.error(f"Slack update error: {e}")

    return state.model_copy(update={"calendar_link": link})


def email_post_cancel(state: OrchestratorState) -> OrchestratorState:
    """User clicked ❌ Cancel — update the Slack card."""
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))
    try:
        client.chat_update(
            channel=state.channel_id,
            ts=state.preview_ts,
            text="❌ Meeting creation cancelled.",
            blocks=None,
        )
    except SlackApiError as e:
        logger.error(f"Slack cancel error: {e}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────
def route_email_entry(state: OrchestratorState) -> Literal[
    "email_fetch_and_parse", "email_create_calendar", "email_post_cancel", "__end__"
]:
    """
    - New email (pubsub/direct) → fetch and classify
    - Button press create_meeting → create calendar event
    - Button press cancel_meeting → post cancel
    """
    if state.slack_action_id == "create_meeting":
        return "email_create_calendar"
    if state.slack_action_id == "cancel_meeting":
        return "email_post_cancel"
    return "email_fetch_and_parse"


def route_after_classify(state: OrchestratorState) -> Literal[
    "email_post_slack_preview", "__end__"
]:
    if state.is_meeting and not state.error:
        return "email_post_slack_preview"
    return "__end__"


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_calendar_subgraph() -> StateGraph:
    b = StateGraph(OrchestratorState)
    b.add_node("email_fetch_and_parse",   email_fetch_and_parse)
    b.add_node("email_classify",          email_classify)
    b.add_node("email_post_slack_preview",email_post_slack_preview)
    b.add_node("email_create_calendar",   email_create_calendar)
    b.add_node("email_post_cancel",       email_post_cancel)

    b.add_conditional_edges(START, route_email_entry, {
        "email_fetch_and_parse":    "email_fetch_and_parse",
        "email_create_calendar":    "email_create_calendar",
        "email_post_cancel":        "email_post_cancel",
        "__end__":                  END,
    })
    b.add_edge("email_fetch_and_parse", "email_classify")
    b.add_conditional_edges("email_classify", route_after_classify, {
        "email_post_slack_preview": "email_post_slack_preview",
        "__end__":                  END,
    })
    b.add_edge("email_post_slack_preview", END)
    b.add_edge("email_create_calendar",    END)
    b.add_edge("email_post_cancel",        END)
    return b.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — Google
# ─────────────────────────────────────────────────────────────────────────────
def _get_google_creds():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
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


def _fetch_new_emails(history_id: str) -> list[dict]:
    try:
        from googleapiclient.discovery import build
        creds   = _get_google_creds()
        service = build("gmail", "v1", credentials=creds)
        history = service.users().history().list(
            userId="me", startHistoryId=history_id, historyTypes=["messageAdded"]
        ).execute()
        emails = []
        for record in history.get("history", []):
            for msg_added in record.get("messagesAdded", []):
                msg_id = msg_added["message"]["id"]
                msg    = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
                emails.append(_parse_gmail_message(msg))
        return emails
    except Exception as e:
        logger.error(f"Gmail fetch error: {e}")
        return []


def _parse_gmail_message(msg: dict) -> dict:
    headers = {h["name"].lower(): h["value"] for h in msg["payload"]["headers"]}

    def extract_addr(raw):
        if "<" in raw and ">" in raw:
            return raw.split("<")[1].split(">")[0].strip().lower()
        return raw.strip().lower()

    def extract_all(raw):
        return [extract_addr(p) for p in raw.split(",") if p.strip()]

    return {
        "message_id": msg["id"],
        "thread_id":  msg["threadId"],
        "subject":    headers.get("subject", ""),
        "from_email": extract_addr(headers.get("from", "")),
        "to_emails":  extract_all(headers.get("to", "")),
        "cc_emails":  extract_all(headers.get("cc", "")),
        "date":       headers.get("date", ""),
        "body":       _extract_body(msg["payload"])[:3000],
        "snippet":    msg.get("snippet", ""),
    }


def _extract_body(payload: dict) -> str:
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        result = _extract_body(part)
        if result:
            return result
    return ""


def _create_calendar_event(model_output: dict, original_email: dict) -> str:
    try:
        from googleapiclient.discovery import build
        from dateutil.parser import parse as dp
        creds   = _get_google_creds()
        service = build("calendar", "v3", credentials=creds)

        start_iso = model_output.get("start_time")
        end_iso   = model_output.get("end_time")
        if not start_iso:
            start_dt  = datetime.now(timezone.utc) + timedelta(days=1)
            start_iso = start_dt.isoformat()
            end_iso   = (start_dt + timedelta(hours=1)).isoformat()
        elif not end_iso:
            end_iso = (dp(start_iso) + timedelta(hours=1)).isoformat()

        attendees = (
            [{"email": e} for e in GROUP_EMAILS] if GROUP_EMAILS
            else [{"email": a} for a in model_output.get("attendees", [])]
        )
        event = {
            "summary":     model_output.get("title") or "Meeting",
            "location":    model_output.get("location") or "",
            "description": f"Auto-created from email. Subject: {original_email.get('subject', '')}",
            "start":       {"dateTime": start_iso, "timeZone": "UTC"},
            "end":         {"dateTime": end_iso,   "timeZone": "UTC"},
            "attendees":   attendees,
            "sendUpdates": "all",
        }
        created = service.events().insert(
            calendarId="primary", body=event, sendNotifications=True
        ).execute()
        return created.get("htmlLink", "")
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        return ""
