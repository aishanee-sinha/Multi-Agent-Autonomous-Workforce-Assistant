"""
calendar_cod.py — Calendar subgraph with Chain-of-Debate slot selection
========================================================================
Drop-in replacement for calendar_agent.py's build_calendar_subgraph().

NEW behaviour in slot_cod:
  1. Least-conflict fallback — if no fully-free overlapping slot exists, finds
     slots where exactly 1 person has a conflict (minimum disruption).
  2. Urgency analysis — for conflicting slots, fetches event titles and marks
     non-urgent events (lunch, weekly 1-1, recurring syncs) as "displaceable",
     preferring those over slots with urgent conflicts.
  3. "Next week" expansion — if no specific day was extracted (time_confidence
     low/none or meeting_start is None), scans all 5 weekdays of next week
     and picks the best slot across the entire week.

What is NOT touched:
  - Jira / Slack ticket creation (slack_agent / slack subgraph)
  - parse_input, router_agent, route_to_agent (routing)
  - Calendar nodes: email_fetch_and_parse, email_classify,
    email_post_slack_preview, email_create_calendar, email_post_cancel

Fallback: any failure (no tokens, API error, CoD error) leaves state unchanged.
"""
from __future__ import annotations
import base64, json, logging, os, re, requests, urllib.parse
from datetime import datetime, timezone, timedelta, date
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv

# Must load env before importing state.py — it reads EC2_IP at module level
load_dotenv()

from redis_store import load_session

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Re-use all existing nodes — no duplication
from calendar_agent import (
    email_fetch_and_parse,
    email_classify,
    email_post_slack_preview,
    email_create_calendar,
    email_post_cancel,
    route_email_entry,
    route_after_classify,
)
from state import OrchestratorState, _llm, CALENDAR_TOKENS

logger = logging.getLogger(__name__)



CLIENT_ID     = os.getenv("GOOGLE_CALENDAR_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET")
MODEL         = "Qwen/Qwen2.5-14B-Instruct-AWQ"
INCLUDE_EMAIL_SENDER = os.getenv("CALENDAR_INCLUDE_SENDER", "true").strip().lower() in {
    "1", "true", "yes", "y", "on"
}

PDT_TZ   = timezone(timedelta(hours=-7), name="PDT")
LOCAL_TZ = PDT_TZ  # all times in this project are PDT


# ─────────────────────────────────────────────────────────────────────────────
# Non-urgent event keyword matching (Feature 2)
# ─────────────────────────────────────────────────────────────────────────────

_NON_URGENT_KEYWORDS = {
    "lunch", "1:1", "1-1", "one-on-one", "one on one",
    "weekly sync", "weekly standup", "weekly stand-up",
    "daily standup", "daily stand-up", "standup", "stand-up",
    "coffee", "coffee chat", "catch up", "catch-up",
    "team social", "happy hour", "offsite", "check-in", "check in",
    "team lunch", "team dinner", "social", "retrospective", "retro",
}

# Recurring meetings with these keywords are also considered displaceable
_RECURRING_DISPLACEABLE = {
    "sync", "meeting", "standup", "stand-up", "1:1", "1-1",
    "weekly", "daily", "monthly", "review", "update",
}


def _is_displaceable(title: str, is_recurring: bool) -> bool:
    """
    Returns True if an event is non-urgent and can be rescheduled to
    accommodate a more important meeting.
    """
    lower = title.lower()
    if any(kw in lower for kw in _NON_URGENT_KEYWORDS):
        return True
    if is_recurring and any(kw in lower for kw in _RECURRING_DISPLACEABLE):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CoD Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class SlotProposal(BaseModel):
    proposed_slot: str = Field(
        description="ISO 8601 datetime string for the start of the proposed slot"
    )
    argument: str = Field(
        description="1-3 sentence argument for why this slot is best"
    )


class SlotChallenge(BaseModel):
    agrees: bool = Field(
        description="True if challenger agrees with the proposal"
    )
    counter_slot: str = Field(
        description="ISO 8601 datetime for alternative slot, or same as proposed if agreeing"
    )
    argument: str = Field(
        description="1-3 sentence challenge or concession"
    )


class RankedSlot(BaseModel):
    start: str = Field(description="ISO 8601 datetime for the slot start")
    end:   str = Field(description="ISO 8601 datetime for the slot end (1 hour after start)")
    reason: str = Field(description="One sentence rationale for this slot's rank")


class SlotVerdict(BaseModel):
    top_slots: list[RankedSlot] = Field(
        description="Exactly 3 best slots ranked from best (#1) to third-best (#3). "
                    "Each must appear in the candidate list."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Historical slot preferences from ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

def _load_slot_preferences() -> str | None:
    """
    Mirrors the hour-counting logic in check_chroma_data.summarize_calendar_feedback().
    Reads calendar_feedback from ChromaDB, counts meeting_start hours of accepted
    meetings, and returns a formatted context string for CoD prompts.
    Returns None if ChromaDB is unreachable or fewer than 3 sessions exist.
    """
    try:
        import chromadb as _chromadb
        from collections import Counter

        host = os.environ.get("CHROMADB_HOST", os.environ.get("EC2_IP", "localhost"))
        port = int(os.environ.get("CHROMADB_PORT", "8001"))
        client = _chromadb.HttpClient(host=host, port=port)

        try:
            col = client.get_collection("calendar_feedback")
        except Exception:
            logger.info("_load_slot_preferences: calendar_feedback collection not found")
            return None

        results = col.get()
        if not results or not results.get("documents"):
            return None

        hour_counter: Counter = Counter()
        total = 0

        for doc in results["documents"]:
            try:
                data = json.loads(doc)
            except (json.JSONDecodeError, TypeError):
                continue

            meeting_start = data.get("meeting_start")
            if not meeting_start:
                continue

            try:
                dt = datetime.fromisoformat(str(meeting_start))
                hour_counter[dt.hour] += 1
                total += 1
            except Exception:
                pass

        if total < 3:
            logger.info(
                "_load_slot_preferences: only %d session(s) — skipping preference injection",
                total,
            )
            return None

        lines = [
            f"=== HISTORICAL SLOT PREFERENCES (from {total} past scheduled meetings) ===",
            "Hour-of-day preference (how often meetings were scheduled at each hour):",
        ]
        for hour in sorted(hour_counter, key=lambda h: -hour_counter[h]):
            count = hour_counter[hour]
            rate  = count / total * 100
            label = f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}"
            lines.append(f"  {label:<8}: {count} meeting(s)  ({rate:.0f}% of all scheduled)")

        lines.append(
            "TIEBREAKER: When candidate slots have equal conflict status, "
            "prefer slots whose start hour matches the most historically preferred hours above."
        )
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("_load_slot_preferences: failed (non-fatal): %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Google Calendar helpers
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_token(refresh_token_str: str) -> str:
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": refresh_token_str,
            "grant_type":    "refresh_token",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _get_events_with_titles(access_token: str, time_min: str, time_max: str) -> list[dict]:
    """
    Fetch calendar events with titles and recurrence info.
    Uses the Events API (not freeBusy) so we get event names for urgency analysis.
    """
    resp = requests.get(
        "https://www.googleapis.com/calendar/v3/calendars/primary/events",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "timeMin":       time_min,
            "timeMax":       time_max,
            "singleEvents":  True,
            "orderBy":       "startTime",
        },
        timeout=10,
    )
    resp.raise_for_status()
    events = []
    for item in resp.json().get("items", []):
        start = item.get("start", {})
        end   = item.get("end", {})
        start_str = start.get("dateTime", start.get("date", ""))
        end_str   = end.get("dateTime",   end.get("date", ""))
        if not start_str or not end_str:
            continue
        events.append({
            "title":        item.get("summary", "(No title)"),
            "start":        start_str,
            "end":          end_str,
            "is_recurring": bool(item.get("recurringEventId")),
        })
    return events


def _load_participant_tokens(state: OrchestratorState) -> dict[str, str]:
    """
    Loads participant tokens from CALENDAR_TOKENS_JSON env var (email → refresh_token),
    but only for participants extracted from email_classify attendees. Optionally
    includes the sender if CALENDAR_INCLUDE_SENDER=true.

    Returns {email: access_token}.
    """
    if not CALENDAR_TOKENS:
        logger.warning("slot_cod: CALENDAR_TOKENS_JSON not set — no calendars to fetch")
        return {}

    attendees_raw = state.meeting_attendees or []
    email_data = state.email_data or {}
    sender_raw = email_data.get("from_email")

    selected_emails: set[str] = {
        str(a).strip().lower()
        for a in attendees_raw
        if isinstance(a, str) and "@" in a
    }

    if INCLUDE_EMAIL_SENDER and isinstance(sender_raw, str) and "@" in sender_raw:
        selected_emails.add(sender_raw.strip().lower())

    matched_emails = [email for email in CALENDAR_TOKENS.keys() if email.lower() in selected_emails]
    logger.info(
        "slot_cod: attendee-driven participant selection attendees=%s include_sender=%s matched=%s",
        sorted(selected_emails), INCLUDE_EMAIL_SENDER, matched_emails,
    )

    if not matched_emails:
        logger.warning("slot_cod: no attendee emails matched CALENDAR_TOKENS_JSON keys")
        return {}

    participants: dict[str, str] = {}
    for email in matched_emails:
        refresh_tok = CALENDAR_TOKENS[email]
        try:
            participants[email] = _refresh_token(refresh_tok)
            logger.info("slot_cod: loaded token for %s", email)
        except Exception as e:
            logger.warning("slot_cod: failed to refresh token for %s: %s", email, e)

    return participants


# ─────────────────────────────────────────────────────────────────────────────
# Slot analysis — free + least-conflict (Features 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

def _find_slots_with_conflicts(
    events_by_participant: dict[str, list[dict]],
    day: date,
    start_hour: int = 9,
    end_hour: int = 17,
    slot_minutes: int = 60,
) -> tuple[list[dict], list[dict]]:
    """
    Walks the work window in slot_minutes increments and classifies each slot.

    Returns (free_slots, partial_slots):
      free_slots    — all participants free
      partial_slots — exactly 1 participant has a conflicting event,
                      with conflict details and displaceability flag
    """
    window_start = datetime(day.year, day.month, day.day, start_hour, 0, tzinfo=LOCAL_TZ)
    window_end   = datetime(day.year, day.month, day.day, end_hour,   0, tzinfo=LOCAL_TZ)
    delta        = timedelta(minutes=slot_minutes)

    # Build per-participant parsed intervals with urgency metadata
    intervals_by_person: dict[str, list[dict]] = {}
    for name, events in events_by_participant.items():
        intervals_by_person[name] = []
        for ev in events:
            try:
                s = datetime.fromisoformat(ev["start"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
                e = datetime.fromisoformat(ev["end"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
                intervals_by_person[name].append({
                    "start":          s,
                    "end":            e,
                    "title":          ev["title"],
                    "is_recurring":   ev["is_recurring"],
                    "is_displaceable": _is_displaceable(ev["title"], ev["is_recurring"]),
                })
            except Exception:
                pass

    free_slots    = []
    partial_slots = []
    cursor        = window_start

    while cursor + delta <= window_end:
        slot_end  = cursor + delta
        conflicts = []

        for name, intervals in intervals_by_person.items():
            for iv in intervals:
                if iv["start"] < slot_end and iv["end"] > cursor:
                    conflicts.append({
                        "participant":    name,
                        "event_title":   iv["title"],
                        "is_recurring":  iv["is_recurring"],
                        "is_displaceable": iv["is_displaceable"],
                    })
                    break  # count max one conflict per person

        slot = {
            "start":          cursor.isoformat(),
            "end":            slot_end.isoformat(),
            "label":          f"{cursor.strftime('%I:%M %p')} - {slot_end.strftime('%I:%M %p')} {cursor.strftime('%Z')}",
            "day_label":      day.strftime("%A, %b %d"),
            "conflict_count": len(conflicts),
            "conflicts":      conflicts,
        }

        if len(conflicts) == 0:
            free_slots.append(slot)
        elif len(conflicts) == 1:
            partial_slots.append(slot)

        cursor += delta

    return free_slots, partial_slots


# ─────────────────────────────────────────────────────────────────────────────
# Search window resolution (Feature 3)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_events_for_day(all_events: list[dict], day: date) -> list[dict]:
    """
    Filter a full-window event list down to events that overlap a single calendar day.
    Used after fetching the entire search window in one API call per participant.
    """
    day_start = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=LOCAL_TZ)
    day_end   = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=LOCAL_TZ)
    result = []
    for ev in all_events:
        try:
            s = datetime.fromisoformat(ev["start"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
            e = datetime.fromisoformat(ev["end"].replace("Z", "+00:00")).astimezone(LOCAL_TZ)
            if s < day_end and e > day_start:
                result.append(ev)
        except Exception:
            pass
    return result


def _next_week_days() -> list[date]:
    """Returns Monday–Friday of next calendar week."""
    today         = date.today()
    days_to_monday = (7 - today.weekday()) % 7
    if days_to_monday == 0:
        days_to_monday = 7
    next_monday = today + timedelta(days=days_to_monday)
    return [next_monday + timedelta(days=i) for i in range(5)]


def _determine_search_days(state: OrchestratorState) -> tuple[list[date], str, int, int]:
    """
    Returns (days_to_search, human_label, start_hour, end_hour).

    meeting_start / meeting_end are YYYY-MM-DDTHH:MM:SS-08:00 datetime strings.
    The date part determines which days to scan.
    The time part determines the hour bounds for slot search within each day.

    - Both set, same date  → single day
    - Both set, range      → all weekdays between start and end (inclusive)
    - Only start set       → single day (start)
    - Neither set          → fall back to all of next week, 9am–5pm
    """
    start_str = state.meeting_start
    end_str   = state.meeting_end

    if start_str:
        try:
            # Parse date portion
            start_date = date.fromisoformat(str(start_str)[:10])
            end_date   = date.fromisoformat(str(end_str)[:10]) if end_str else start_date

            # Parse hour bounds from time portion (default 9–17)
            try:
                start_hour = datetime.fromisoformat(str(start_str)).hour
            except Exception:
                start_hour = 9
            try:
                end_hour = datetime.fromisoformat(str(end_str)).hour if end_str else 17
            except Exception:
                end_hour = 17

            # Collect weekdays within the date range
            days = []
            cursor = start_date
            while cursor <= end_date:
                if cursor.weekday() < 5:   # Mon–Fri only
                    days.append(cursor)
                cursor += timedelta(days=1)

            if not days:
                days = [start_date]

            if len(days) == 1:
                label = days[0].strftime("%A, %B %d %Y")
                logger.info(
                    "slot_cod: single-day window (%s %02d:00-%02d:00, confidence=%s)",
                    label, start_hour, end_hour, state.time_confidence,
                )
            else:
                label = f"{days[0].strftime('%b %d')} - {days[-1].strftime('%b %d')}"
                logger.info(
                    "slot_cod: multi-day window %s %02d:00-%02d:00 (%d days, confidence=%s)",
                    label, start_hour, end_hour, len(days), state.time_confidence,
                )

            return days, label, start_hour, end_hour
        except Exception:
            pass

    # No window extracted — fall back to all of next week, 9am–5pm
    days = _next_week_days()
    label = f"next week ({days[0].strftime('%b %d')} - {days[-1].strftime('%b %d')})"
    logger.info("slot_cod: no window extracted (confidence=%s) — scanning %s 09:00-17:00",
                state.time_confidence, label)
    return days, label, 9, 17


# ─────────────────────────────────────────────────────────────────────────────
# Chain-of-Debate — 3 rounds via Qwen
# ─────────────────────────────────────────────────────────────────────────────

def _build_email_context(state: OrchestratorState) -> dict:
    """Extract email_classify output into a plain dict for CoD prompts."""
    email_data = state.email_data or {}
    body = email_data.get("body", "")
    return {
        "title":           state.meeting_title or "Meeting",
        "attendees":       state.meeting_attendees or [],
        "start_window":    state.meeting_start,
        "end_window":      state.meeting_end,
        "time_confidence": state.time_confidence or "none",
        "body_snippet":    body[:400].replace("\n", " "),
    }


def _build_cod_context(
    enriched_slots: list[dict],
    participants: list[str],
    search_label: str,
    email_context: dict,
    slot_prefs_text: str | None = None,
) -> str:
    """Build the shared context block for all three CoD rounds."""
    # --- Email intent section ---
    attendees_str = ", ".join(email_context["attendees"]) or ", ".join(participants)
    lines = [
        "=== EMAIL INTENT ===",
        f"Meeting title    : {email_context['title']}",
        f"Attendees        : {attendees_str}",
        f"Requested window : {email_context['start_window']} to {email_context['end_window']}",
        f"Time confidence  : {email_context['time_confidence']}",
        f"Email snippet    : {email_context['body_snippet']}",
        "",
        "=== CALENDAR AVAILABILITY (within search window) ===",
        f"Search window : {search_label}",
        f"Participants  : {', '.join(participants)}",
        "Candidate slots (free or single displaceable conflict):",
    ]

    for s in enriched_slots:
        prefix = f"{s['day_label']}  {s['label']}"
        if s["conflict_count"] == 0:
            lines.append(f"  - {prefix}  [ALL FREE]  (start={s['start']})")
        else:
            c = s["conflicts"][0]
            urgency = "NON-URGENT — can be rescheduled" if c["is_displaceable"] else "URGENT — cannot be moved"
            lines.append(
                f"  - {prefix}  "
                f"[1 CONFLICT: {c['participant']} has '{c['event_title']}' — {urgency}]"
                f"  (start={s['start']})"
            )

    if slot_prefs_text:
        lines.append("")
        lines.append(slot_prefs_text)

    return "\n".join(lines)


def _run_cod(
    enriched_slots: list[dict],
    participants: list[str],
    search_label: str,
    email_context: dict,
    slot_prefs_text: str | None = None,
) -> SlotVerdict | None:
    context = _build_cod_context(enriched_slots, participants, search_label, email_context, slot_prefs_text)

    try:
        # ── Round 1: Proposer ────────────────────────────────────────────────
        proposal: SlotProposal = _llm(
            structured_output=SlotProposal, model_name=MODEL
        ).invoke([
            SystemMessage(content=(
                "You are the Proposer in a meeting scheduling debate.\n"
                "You are given the original email intent AND the calendar availability of all attendees.\n\n"
                "Your job: pick the single best 1-hour slot from the candidate list.\n"
                "Rules (in priority order):\n"
                "  1. Respect any time preferences in the email "
                "(e.g. 'morning', 'after 2pm', 'before noon') — these take highest priority\n"
                "  2. Prefer [ALL FREE] slots over conflict slots\n"
                "  3. Among free slots, prefer mid-morning (10–11am) unless email says otherwise\n"
                "  4. Only consider NON-URGENT/displaceable conflicts if no free slot exists\n"
                "  5. Never pick a slot with an URGENT conflict\n"
                "  6. TIEBREAKER — among slots with equal conflict status, prefer the slot "
                "whose start hour has the highest historical selection rate from "
                "HISTORICAL SLOT PREFERENCES (if present in context). Never override rules 1–5 for this.\n"
                "Argue your choice in 1–3 sentences referencing the email context.\n"
                "Return a SlotProposal with proposed_slot as the ISO 8601 start datetime from the list."
            )),
            HumanMessage(content=context),
        ])
        logger.info("slot_cod proposer: %s | %s", proposal.proposed_slot, proposal.argument[:100])

        # ── Round 2: Challenger ──────────────────────────────────────────────
        challenge: SlotChallenge = _llm(
            structured_output=SlotChallenge, model_name=MODEL
        ).invoke([
            SystemMessage(content=(
                "You are the Challenger in a meeting scheduling debate.\n"
                "You are given the original email intent, calendar availability, "
                "and the Proposer's choice.\n\n"
                "Your job: verify the Proposer's choice against the email intent and calendar data.\n"
                "Challenge if any of these are true:\n"
                "  - The Proposer ignored a time preference from the email "
                "(e.g. email says 'morning' but Proposer picked afternoon)\n"
                "  - An [ALL FREE] slot was skipped in favour of a conflict slot\n"
                "  - The Proposer picked an URGENT conflict when a free slot exists\n"
                "  - A clearly better slot exists given the meeting's context and attendees\n"
                "Agree if the Proposer's choice genuinely fits the email intent and calendar data.\n"
                "Never propose a slot with an URGENT conflict.\n"
                "Return a SlotChallenge."
            )),
            HumanMessage(content=(
                f"{context}\n\n"
                f"Proposer chose: {proposal.proposed_slot}\n"
                f"Proposer argument: {proposal.argument}"
            )),
        ])
        logger.info(
            "slot_cod challenger: agrees=%s counter=%s",
            challenge.agrees, challenge.counter_slot,
        )

        # ── Round 3: Judge ───────────────────────────────────────────────────
        verdict: SlotVerdict = _llm(
            structured_output=SlotVerdict, model_name=MODEL
        ).invoke([
            SystemMessage(content=(
                "You are the Judge in a meeting scheduling debate.\n"
                "You are given the original email intent, calendar availability, "
                "and both the Proposer's and Challenger's arguments.\n\n"
                "Your job: pick the TOP 3 best slots from the candidate list, ranked best to third-best.\n"
                "Rules:\n"
                "  1. Every slot MUST appear in the candidate list — never invent a time\n"
                "  2. Every slot MUST honour any time constraints from the email "
                "(e.g. 'morning', 'before 2pm')\n"
                "  3. Strongly prefer [ALL FREE] slots over conflict slots\n"
                "  4. Among conflict slots, only accept NON-URGENT/displaceable ones\n"
                "  5. Prefer earlier days and mid-morning when comparing across the week\n"
                "  6. If fewer than 3 candidate slots exist, return as many as are available\n"
                "  7. TIEBREAKER — when ranking slots with equal conflict status, rank higher "
                "those whose start hour matches historically preferred hours from "
                "HISTORICAL SLOT PREFERENCES (if present in context). Never override rules 1–6 for this.\n"
                "Return a SlotVerdict with top_slots: a list of up to 3 RankedSlot objects.\n"
                "Each RankedSlot has: start (ISO 8601), end (1 hour after start, ISO 8601), "
                "reason (one sentence)."
            )),
            HumanMessage(content=(
                f"{context}\n\n"
                f"Proposer chose: {proposal.proposed_slot}\n"
                f"Proposer argument: {proposal.argument}\n\n"
                f"Challenger {'agreed' if challenge.agrees else 'countered with'}: "
                f"{challenge.counter_slot}\n"
                f"Challenger argument: {challenge.argument}"
            )),
        ])
        for i, s in enumerate(verdict.top_slots):
            logger.info("slot_cod verdict #%d: %s | %s", i + 1, s.start, s.reason)
        return verdict

    except Exception as e:
        logger.error("slot_cod CoD error: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# slot_cod node
# ─────────────────────────────────────────────────────────────────────────────

def slot_cod(state: OrchestratorState) -> OrchestratorState:
    """
    Chain-of-Debate slot selector inserted after email_classify.

    The CoD judge is the SOLE authority on meeting_start / meeting_end.
    The email-extracted time (from email_classify) is used only as a search
    hint to determine which day(s) to scan — it is never passed through as
    the final scheduled time.

    Three-tier candidate selection:
      Tier 1 — fully overlapping free slots on the proposed day/week
      Tier 2 — single-conflict slots where the conflict is non-urgent
                (lunch, recurring 1-1, etc.) and displaceable
      Tier 3 — no viable slots found → meeting_start/end set to None
                (Slack preview will show TBD; human decides)
    """
    if not state.is_meeting:
        return state

    email = state.email_data or {}
    logger.info("=" * 60)
    logger.info("STARTING COD — EMAIL CONTEXT")
    logger.info(f"  subject   : {email.get('subject')}")
    logger.info(f"  from      : {email.get('from_email')}")
    logger.info(f"  is_meeting: {state.is_meeting}")
    logger.info(f"  title     : {state.meeting_title}")
    logger.info(f"  confidence: {state.time_confidence}")
    logger.info(f"  start_win : {state.meeting_start}")
    logger.info(f"  end_win   : {state.meeting_end}")
    logger.info(f"  body      :\n{email.get('body', '(empty)')[:500]}")
    logger.info("=" * 60)

    # Capture the full email_classify output as context for CoD BEFORE any state changes.
    # The search window (meeting_start/end) is used as a hint to pick days to scan;
    # the CoD judge sets the final meeting_start/end — email values do not leak through.
    email_context = _build_email_context(state)

    # 1. Determine which day(s) and hour bounds to scan
    search_days, search_label, start_hour, end_hour = _determine_search_days(state)
    logger.info("slot_cod: search window = %s (%d day(s), %02d:00-%02d:00)",
                search_label, len(search_days), start_hour, end_hour)

    # 2. Load participant tokens
    access_tokens = _load_participant_tokens(state)
    if not access_tokens:
        logger.warning("slot_cod: no tokens loaded — cod_top_slots set to []")
        return state.model_copy(update={"cod_top_slots": []})

    # 3. Fetch events for the FULL search window in one API call per participant.
    #    fetch_min/fetch_max come directly from email_classify's start_window/end_window.
    #    If the window is missing (fallback), derive bounds from the search days.
    if email_context["start_window"] and email_context["end_window"]:
        # Strip any existing tz info and stamp PDT — email_classify output is always PDT
        from dateutil.parser import parse as _dp
        fetch_min = _dp(email_context["start_window"]).replace(tzinfo=None).replace(tzinfo=PDT_TZ).isoformat()
        fetch_max = _dp(email_context["end_window"]).replace(tzinfo=None).replace(tzinfo=PDT_TZ).isoformat()
    else:
        fetch_min = datetime(search_days[0].year,  search_days[0].month,  search_days[0].day,
                             start_hour, 0, tzinfo=PDT_TZ).isoformat()
        fetch_max = datetime(search_days[-1].year, search_days[-1].month, search_days[-1].day,
                             end_hour, 0, tzinfo=PDT_TZ).isoformat()

    logger.info("slot_cod: fetching calendars for search window  %s  ->  %s", fetch_min, fetch_max)

    all_window_events: dict[str, list[dict]] = {}
    for name, token in access_tokens.items():
        try:
            all_window_events[name] = _get_events_with_titles(token, fetch_min, fetch_max)
            logger.info("slot_cod: %s — %d event(s) in window", name, len(all_window_events[name]))
            for ev in all_window_events[name]:
                disp = _is_displaceable(ev["title"], ev["is_recurring"])
                logger.info("  -> '%s'  recurring=%s  displaceable=%s",
                            ev["title"], ev["is_recurring"], disp)
        except Exception as e:
            logger.warning("slot_cod: events fetch failed for %s: %s", name, e)

    if not all_window_events:
        logger.warning("slot_cod: no events fetched — cod_top_slots set to []")
        return state.model_copy(update={"cod_top_slots": []})

    # 4. Analyse slots day by day — filter the pre-fetched window events to each day
    all_free_slots:    list[dict] = []
    all_partial_slots: list[dict] = []

    for day in search_days:
        events_by_participant = {
            name: _filter_events_for_day(events, day)
            for name, events in all_window_events.items()
        }

        free, partial = _find_slots_with_conflicts(
            events_by_participant, day, start_hour=start_hour, end_hour=end_hour
        )
        all_free_slots.extend(free)
        all_partial_slots.extend(partial)
        logger.info("slot_cod: %s — %d free slot(s), %d single-conflict slot(s)",
                    day.strftime("%a %b %d"), len(free), len(partial))

    # 4. Select candidate slots for CoD
    if all_free_slots:
        candidate_slots = all_free_slots
        logger.info("slot_cod: using %d fully-free slot(s)", len(candidate_slots))

    elif all_partial_slots:
        # Prefer displaceable conflicts over non-displaceable
        displaceable = [s for s in all_partial_slots if s["conflicts"][0]["is_displaceable"]]
        candidate_slots = displaceable if displaceable else all_partial_slots
        if displaceable:
            logger.info(
                "slot_cod: no fully-free slots — using %d displaceable-conflict slot(s)",
                len(candidate_slots),
            )
        else:
            logger.info(
                "slot_cod: no displaceable slots — using %d least-conflict slot(s) as last resort",
                len(candidate_slots),
            )

    else:
        logger.info("slot_cod: no viable slots found — cod_top_slots set to []")
        return state.model_copy(update={"cod_top_slots": []})

    # 5. Load historical hour preferences from ChromaDB and run Chain-of-Debate
    slot_prefs_text = _load_slot_preferences()
    if slot_prefs_text:
        logger.info("slot_cod: injecting historical slot preferences into CoD context")
    verdict = _run_cod(candidate_slots, list(access_tokens.keys()), search_label, email_context, slot_prefs_text)
    if not verdict or not verdict.top_slots:
        logger.warning("slot_cod: CoD returned no verdict — cod_top_slots set to []")
        return state.model_copy(update={"cod_top_slots": []})

    top_slots = [
        {"start": s.start, "end": s.end, "reason": s.reason}
        for s in verdict.top_slots[:3]
    ]
    logger.info("slot_cod: CoD judge returned %d ranked slot(s)", len(top_slots))
    return state.model_copy(update={"cod_top_slots": top_slots})


# ─────────────────────────────────────────────────────────────────────────────
# Calendar subgraph with CoD injected
# ─────────────────────────────────────────────────────────────────────────────

def build_calendar_subgraph_cod() -> StateGraph:
    """
    Flow:
      email_fetch_and_parse
        -> email_classify
          -> slot_cod           (CoD picks best free/least-conflict slot)
            -> email_post_slack_preview
          -> END                (if not meeting / error)
      email_create_calendar     (button: create_meeting)
      email_post_cancel         (button: cancel_meeting)
    """
    b = StateGraph(OrchestratorState)
    b.add_node("email_fetch_and_parse",    email_fetch_and_parse)
    b.add_node("email_classify",           email_classify)
    b.add_node("slot_cod",                 slot_cod)
    b.add_node("email_post_slack_preview", email_post_slack_preview)
    b.add_node("email_create_calendar",    email_create_calendar)
    b.add_node("email_post_cancel",        email_post_cancel)

    b.add_conditional_edges(START, route_email_entry, {
        "email_fetch_and_parse": "email_fetch_and_parse",
        "email_create_calendar": "email_create_calendar",
        "email_post_cancel":     "email_post_cancel",
        "__end__":               END,
    })
    b.add_edge("email_fetch_and_parse", "email_classify")
    b.add_conditional_edges("email_classify", route_after_classify, {
        "email_post_slack_preview": "slot_cod",  # route through CoD first
        "__end__":                  END,
    })
    b.add_edge("slot_cod",                 "email_post_slack_preview")
    b.add_edge("email_post_slack_preview", END)
    b.add_edge("email_create_calendar",    END)
    b.add_edge("email_post_cancel",        END)
    return b.compile()