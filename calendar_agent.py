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
from redis_store import save_session, record_feedback
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

PDT_TZ = timezone(timedelta(hours=-7), name="PDT")
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
    For Pub/Sub events — fetch emails from Gmail using the last persisted
    historyId from SSM (not the one in the notification, which is the
    current state and would return nothing).
    For direct invocations — email_data is already set, skip fetch.
    """
    if state.email_source != "pubsub" or not state.email_data:
        return state

    try:
        from gmail_history import get_last_history_id, set_last_history_id

        pubsub_message   = state.email_data.get("message", {})
        decoded          = json.loads(base64.b64decode(pubsub_message["data"]).decode("utf-8"))
        pubsub_history_id = str(decoded.get("historyId", ""))
        email_address    = decoded.get("emailAddress", "")
        logger.info(f"email_fetch_and_parse: pubsub historyId={pubsub_history_id} emailAddress={email_address}")

        # Use SSM-persisted historyId as the start of the query window
        last_history_id = get_last_history_id()
        if not last_history_id:
            # First run — seed with pubsub historyId minus a small offset so we
            # catch the triggering email
            last_history_id = str(int(pubsub_history_id) - 10)
            logger.warning(f"email_fetch_and_parse: no SSM historyId found, falling back to {last_history_id}")

        logger.info(f"email_fetch_and_parse: querying history since historyId={last_history_id}")
        emails = _fetch_new_emails(last_history_id)
        logger.info(f"email_fetch_and_parse: fetched {len(emails)} email(s)")

        # Always advance the checkpoint to the Pub/Sub historyId
        set_last_history_id(pubsub_history_id)

        if emails:
            e = emails[0]
            logger.info(f"email_fetch_and_parse: from={e.get('from_email')} subject={e.get('subject')!r} body_len={len(e.get('body', ''))}")
            return state.model_copy(update={"email_data": emails[0]})

        logger.warning("email_fetch_and_parse: no emails returned — stopping")
        return state.model_copy(update={"intent": "none", "email_data": None})
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


def _format_slot_label(start_iso: str) -> str:
    """Format a slot ISO datetime into a human-readable Slack button label."""
    try:
        from dateutil.parser import parse as dp
        dt = dp(start_iso)
        return dt.strftime("%a %b %-d, %-I:%M %p")
    except Exception:
        return start_iso


def email_post_slack_preview(state: OrchestratorState) -> OrchestratorState:
    """
    Post a Slack card with up to 3 CoD-ranked slot buttons + ❌ Cancel.

    A single Redis session is saved for the whole meeting, containing all
    proposed slots. Every button (slot and cancel) shares the same session_id.
    parse_input uses the action_id (select_slot_0/1/2) to know which slot
    was chosen and updates the session accordingly.

    Falls back to a no-slots card when CoD produced nothing.
    """
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))

    title     = state.meeting_title or "Meeting"
    location  = state.meeting_location or "N/A"
    attendees = ", ".join(state.meeting_attendees[:3]) or "N/A"
    top_slots = state.cod_top_slots  # list of {start, end, reason}

    # ── Save ONE session for this meeting with all proposed slots ─────────────
    session_payload = {
        "email_data": state.email_data,
        "meeting_title":    title,
        "meeting_location": state.meeting_location,
        "meeting_attendees": state.meeting_attendees,
        "time_confidence":  state.time_confidence,
        "all_proposed_slots": [
            {"start": s["start"], "end": s["end"], "reason": s.get("reason", "")}
            for s in top_slots
        ],
    }
    session_id = save_session(session_payload)
    logger.info(
        "email_post_slack_preview: saved session=%s with %d proposed slot(s)",
        session_id, len(top_slots),
    )

    # ── Build Slack blocks ────────────────────────────────────────────────────
    header_text = (
        f"📧 *Meeting detected in incoming email*\n"
        f"*Title:*     {title}\n"
        f"*Location:*  {location}\n"
        f"*Attendees:* {attendees}\n"
    )
    if top_slots:
        header_text += "\n*CoD proposed the following slots — pick one:*"
    else:
        header_text += "\n⚠️ *No available slots found — please schedule manually.*"

    blocks: list[dict] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": header_text}}
    ]

    slot_action_ids = ["select_slot_0", "select_slot_1", "select_slot_2"]
    for i, (slot, action_id) in enumerate(zip(top_slots, slot_action_ids)):
        label  = _format_slot_label(slot["start"])
        reason = slot.get("reason", "")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*#{i+1}* — {label}  _{reason}_",
            },
            "accessory": {
                "type":      "button",
                "text":      {"type": "plain_text", "text": f"✅ Slot {i+1}"},
                "style":     "primary",
                "value":     session_id,   # same session for all slot buttons
                "action_id": action_id,
            },
        })

    blocks.append({
        "type": "actions",
        "elements": [{
            "type":      "button",
            "text":      {"type": "plain_text", "text": "❌ Cancel"},
            "style":     "danger",
            "value":     session_id,       # same session for cancel too
            "action_id": "cancel_meeting",
        }],
    })

    try:
        resp = client.chat_postMessage(
            channel=SLACK_NOTIFY_CHANNEL,
            blocks=blocks,
            text=f"Meeting detected: {title} — pick a slot",
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

    # Record human feedback — updates the single meeting session in Redis
    if state.session_id:
        if link:
            record_feedback(state.session_id, "selected", {
                "calendar_link":       link,
                "selected_slot_index": pending.get("selected_slot_index"),
                "meeting_start":       model_output.get("start_time"),
                "meeting_end":         model_output.get("end_time"),
            })
        else:
            record_feedback(state.session_id, "failed", {"reason": "calendar_create_error"})

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
    """User clicked ❌ Cancel — mark the meeting session as rejected."""
    if state.session_id:
        record_feedback(state.session_id, "rejected")

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
            userId="me", startHistoryId=history_id, historyTypes=["messageAdded"],
            labelId="INBOX"
        ).execute()
        logger.info(f"_fetch_new_emails: historyId={history_id} history_records={len(history.get('history', []))}")
        emails = []
        for record in history.get("history", []):
            for msg_added in record.get("messagesAdded", []):
                msg_id = msg_added["message"]["id"]
                logger.info(f"_fetch_new_emails: fetching message id={msg_id}")
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
        from googleapiclient.errors import HttpError
        from dateutil.parser import parse as dp
        creds   = _get_google_creds()
        service = build("calendar", "v3", credentials=creds)

        def _to_pdt(raw_value: str | None) -> str | None:
            """
            Parse the model datetime string and stamp it as PDT (UTC-7).
            The model and email always express times in PDT, so any existing
            timezone info is stripped before stamping — never converted.
            Returns a valid RFC3339 string like 2026-04-14T10:00:00-07:00.
            """
            if not raw_value:
                return None
            text = str(raw_value).strip()
            # Strip common malformed separators before the offset
            text = re.sub(r"([0-9])\s([+-]\d{2}:\d{2})$", r"\1\2", text)
            text = re.sub(r"([0-9])\s(\d{2}:\d{2})$",     r"\1+\2", text)
            # Parse then replace (not convert) timezone with PDT
            dt = dp(text).replace(tzinfo=PDT_TZ)
            return dt.isoformat()

        start_iso = _to_pdt(model_output.get("start_time"))
        end_iso   = _to_pdt(model_output.get("end_time"))

        if not start_iso:
            logger.error("_create_calendar_event: start_time is None — aborting")
            return None
        if not end_iso:
            end_iso = (dp(start_iso) + timedelta(hours=1)).isoformat()

        attendees = (
            [{"email": e} for e in GROUP_EMAILS] if GROUP_EMAILS
            else [{"email": a} for a in model_output.get("attendees", [])]
        )
        event = {
            "summary":     model_output.get("title") or "Meeting",
            "location":    model_output.get("location") or "",
            "description": f"Auto-created from email. Subject: {original_email.get('subject', '')}",
            "start":       {"dateTime": start_iso, "timeZone": "America/Los_Angeles"},
            "end":         {"dateTime": end_iso,   "timeZone": "America/Los_Angeles"},
            "attendees":   attendees,
        }

        # Log the exact request payload and query params sent to Calendar API.
        # This is useful for diagnosing 400 Bad Request responses in Lambda logs.
        logger.info(
            "calendar.insert request params: calendarId=%s sendUpdates=%s",
            "primary",
            "all",
        )
        logger.info("calendar.insert request body: %s", json.dumps(event, default=str))

        created = service.events().insert(
            calendarId="primary", body=event, sendUpdates="all"
        ).execute()
        return created.get("htmlLink", "")
    except HttpError as e:
        status = getattr(getattr(e, "resp", None), "status", "unknown")
        body = ""
        try:
            if getattr(e, "content", None):
                body = e.content.decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        logger.error("Calendar HttpError status=%s message=%s", status, str(e))
        logger.error("Calendar HttpError body=%s", body)
        return ""
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        return ""
