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
from pydantic import BaseModel, EmailStr, EmailStr, Field
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
def _build_email_system_prompt() -> str:
    from datetime import date as _date
    today     = _date.today()
    today_str = today.strftime("%Y-%m-%d (%A)")
    tomorrow  = today + timedelta(days=1)

    # Compute "next <weekday>" for all 7 days dynamically — no hardcoding
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    next_weekdays = {}
    for target_wd, name in enumerate(weekday_names):
        days_ahead = (target_wd - today.weekday()) % 7 or 7
        next_weekdays[name] = today + timedelta(days=days_ahead)

    next_weekday_lines = "\n".join(
        f'     "next {name}" -> "{d.strftime("%Y-%m-%d")}"'
        for name, d in next_weekdays.items()
    )
    next_monday = next_weekdays["Monday"]
    next_friday = next_weekdays["Friday"]
    this_friday = today + timedelta(days=(4 - today.weekday()) % 7)

    # Compute local UTC offset dynamically (e.g. "+05:30", "-07:00")
    _now        = datetime.now().astimezone()
    _tz_name    = _now.strftime("%Z")                          # e.g. "IST", "PDT", "EST"
    _raw_offset = _now.strftime("%z")                          # e.g. "+0530", "-0700"
    _tz_offset  = _raw_offset[:3] + ":" + _raw_offset[3:]     # e.g. "+05:30", "-07:00"

    return f"""You are a meeting intent extraction system.

TASK: Analyze emails and extract meeting intent and a SEARCH WINDOW for scheduling.
Do NOT extract exact times — extract the date range the sender intends to meet within.
A downstream scheduler will find the best available slot inside that window.

OUTPUT FORMAT (JSON only, no explanations):
{{
  "is_meeting": boolean,
  "title": string or null,
  "attendees": array of email addresses,
  "start_window": "YYYY-MM-DDTHH:MM:SS{_tz_offset}" or null,
  "end_window": "YYYY-MM-DDTHH:MM:SS{_tz_offset}" or null,
  "time_confidence": "high" | "medium" | "low" | "none"
}}

EXTRACTION RULES:

1. MEETING DETECTION (is_meeting):
   - TRUE if the email is requesting or proposing to schedule a meeting, call, sync, or appointment
   - FALSE for announcements, reports, updates, or questions with no scheduling intent

2. TITLE EXTRACTION:
   - Short description of the meeting purpose (max 100 characters)
   - Use the email subject if it describes the meeting
   - Set to null for non-meetings

3. ATTENDEES EXTRACTION:
   - All email addresses in To, Cc, or mentioned in the body
   - Lowercase only. Empty array [] for non-meetings.

IMPORTANT: start_window and end_window define a SEARCH RANGE for a scheduler to scan.
They are NOT the meeting start/end times and have NOTHING to do with meeting duration.
end_window is always the LAST MOMENT the scheduler should look — typically 17:00 on the last candidate day.
NEVER compute end_window as start_window + meeting duration.

All datetimes use the local timezone: {_tz_name} (UTC{_tz_offset}).

4. START_WINDOW — earliest datetime the scheduler should begin searching (YYYY-MM-DDTHH:MM:SS{_tz_offset}):
   TODAY is {today_str}. "tomorrow" is {tomorrow.strftime("%Y-%m-%d")}.
   Use the exact dates below for "next <weekday>" references:
{next_weekday_lines}
   - "next week" (no specific day) -> next Monday = "{next_monday.strftime('%Y-%m-%d')}"
   - "this week" (no specific day) -> tomorrow = "{tomorrow.strftime('%Y-%m-%d')}"
   - No date reference at all      -> null
   TIME within the day:
   - If a start time is mentioned (e.g. "after 2pm", "from 10am") -> use that time
   - If only "morning" mentioned -> 09:00:00
   - If only "afternoon" mentioned -> 13:00:00
   - If no time constraint at all -> default to 09:00:00 (start of work day)

5. END_WINDOW — latest datetime the scheduler should stop searching (YYYY-MM-DDTHH:MM:SS{_tz_offset}):
   This marks the END OF THE SEARCH RANGE, not the end of the meeting.
   DATE:
   - Multiple days mentioned (e.g. "Monday or Tuesday") -> use the LAST mentioned day
   - Single specific day mentioned -> same date as start_window
   - "next week" (no specific day) -> next Friday = "{next_friday.strftime('%Y-%m-%d')}"
   - "this week" (no specific day) -> this Friday = "{this_friday.strftime('%Y-%m-%d')}"
   - No date reference at all      -> null
   TIME within the day:
   - If an end/deadline time is mentioned (e.g. "before 2pm", "by noon", "until 3pm") -> use that time
   - If only "morning" mentioned -> 12:00:00
   - If only "afternoon" mentioned -> 17:00:00
   - If no time constraint at all -> default to 17:00:00 (end of work day)

   EXAMPLES (correct search windows):
   - "next Monday" -> start="{next_weekdays['Monday'].strftime('%Y-%m-%d')}T09:00:00{_tz_offset}"  end="{next_weekdays['Monday'].strftime('%Y-%m-%d')}T17:00:00{_tz_offset}"
   - "next Monday or Tuesday" -> start="{next_weekdays['Monday'].strftime('%Y-%m-%d')}T09:00:00{_tz_offset}"  end="{next_weekdays['Tuesday'].strftime('%Y-%m-%d')}T17:00:00{_tz_offset}"
   - "next week" -> start="{next_monday.strftime('%Y-%m-%d')}T09:00:00{_tz_offset}"  end="{next_friday.strftime('%Y-%m-%d')}T17:00:00{_tz_offset}"
   - "Monday at 2pm" -> start="{next_weekdays['Monday'].strftime('%Y-%m-%d')}T14:00:00{_tz_offset}"  end="{next_weekdays['Monday'].strftime('%Y-%m-%d')}T17:00:00{_tz_offset}"

6. TIME CONFIDENCE:
   - "high"   = specific day AND clock time mentioned (e.g. "Monday at 2pm")
   - "medium" = specific day mentioned, no clock time (e.g. "next Monday", "this Friday")
   - "low"    = only week-level reference, no specific day (e.g. "next week", "this week")
   - "none"   = no time or date information, or not a meeting

CRITICAL: ALWAYS return valid JSON. NEVER add explanations."""


EMAIL_SYSTEM_PROMPT = _build_email_system_prompt()

class EmailMeetingDetails(BaseModel):
    is_meeting: bool
    title: str | None = Field(default=None, max_length=100)
    attendees: list[EmailStr] = Field(default_factory=list)
    start_window: str | None = None   # YYYY-MM-DDTHH:MM:SS±HH:MM — earliest datetime to search (local tz)
    end_window:   str | None = None   # YYYY-MM-DDTHH:MM:SS±HH:MM — latest datetime to search (local tz)
    time_confidence: Literal["high", "medium", "low", "none"] = "none"


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

    llm = _llm(structured_output=EmailMeetingDetails, model_name="Qwen/Qwen2.5-14B-Instruct-AWQ")
    try:
        data = llm.invoke([SystemMessage(content=_build_email_system_prompt()), HumanMessage(content=email_text)])
        # raw  = resp.content if hasattr(resp, "content") else str(resp)
        # raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
        # data = EmailMeetingDetails(**json.loads(raw))
        return state.model_copy(update={
            "is_meeting":        data.is_meeting,
            "meeting_title":     data.title,
            "meeting_start":     data.start_window,   # date string YYYY-MM-DD — search window start
            "meeting_end":       data.end_window,     # date string YYYY-MM-DD — search window end
            "meeting_attendees": data.attendees or [],
            "time_confidence":   data.time_confidence,
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
