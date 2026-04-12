"""
test_cod_email_flow.py
======================
End-to-end test of the CoD calendar flow.

What this test does:
  1. Runs email_classify  → logs LLM output
  2. Runs slot_cod        → logs calendar fetch, slot analysis, all 3 CoD rounds,
                            and the top-3 ranked slots from the Judge
  3. Skips Slack / Redis  → no posts, no Redis writes
  4. Simulates user picking Slot 1, Slot 2, Slot 3, and Cancel
     → logs the calendar event that WOULD be created for each

Run:
    cd d:\\agent-project
    ..\\llm_venv\\Scripts\\python tests\\test_cod_email_flow.py
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
for noisy in ("httpx", "httpcore", "openai", "urllib3", "requests", "slack_sdk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("test_cod_email_flow")

PDT_TZ = timezone(timedelta(hours=-7), name="PDT")


def _fmt_pdt(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        dt = dt.replace(tzinfo=None).replace(tzinfo=PDT_TZ)
        return dt.strftime("%a %b %d  %I:%M %p PDT")
    except Exception:
        return iso


def _divider(title="", width=70, char="─"):
    if title:
        pad = (width - len(title) - 2) // 2
        logger.info("%s %s %s", char * pad, title, char * (width - pad - len(title) - 2))
    else:
        logger.info(char * width)


# ─────────────────────────────────────────────────────────────────────────────
# Patch Redis — no-ops (no reads or writes)
# ─────────────────────────────────────────────────────────────────────────────
import redis_store as _rs

def _noop_save_session(data):
    logger.info("[REDIS SKIP] save_session — would store: %s", list(data.keys()))
    return "test-session-id-001"

def _noop_load_session(session_id):
    logger.info("[REDIS SKIP] load_session(%s)", session_id)
    return {}

def _noop_record_feedback(session_id, outcome, metadata=None):
    logger.info("[REDIS SKIP] record_feedback(%s, outcome=%s, metadata=%s)",
                session_id, outcome, list(metadata.keys()) if metadata else None)

_rs.save_session     = _noop_save_session
_rs.load_session     = _noop_load_session
_rs.record_feedback  = _noop_record_feedback

# Also patch the imported references in calendar_agent
import calendar_agent as _cal_mod
_cal_mod.save_session    = _noop_save_session
_cal_mod.record_feedback = _noop_record_feedback

# ─────────────────────────────────────────────────────────────────────────────
# Patch Slack — no posts, just log
# ─────────────────────────────────────────────────────────────────────────────
_original_email_post_slack_preview = _cal_mod.email_post_slack_preview
_original_email_create_calendar    = _cal_mod.email_create_calendar
_original_email_post_cancel        = _cal_mod.email_post_cancel

def _patched_email_post_slack_preview(state):
    _divider("SLACK PREVIEW (skipped — test mode)")
    top_slots = state.cod_top_slots
    if top_slots:
        logger.info("CoD proposed %d slot(s):", len(top_slots))
        for i, s in enumerate(top_slots):
            logger.info("  Slot %d: %s  (%s)", i + 1, s["start"], _fmt_pdt(s["start"]))
            logger.info("          %s", s.get("reason", ""))
    else:
        logger.info("No slots proposed — TBD card would be shown")
    logger.info("[SLACK SKIP] Would post preview card to Slack")
    return state.model_copy(update={"preview_ts": "test-preview-ts"})

def _patched_email_post_cancel(state):
    _divider("CANCEL (skipped — test mode)")
    logger.info("[SLACK SKIP] Would update Slack card to cancelled")
    logger.info("[REDIS SKIP] Would record feedback=rejected for session %s", state.session_id)
    return state

_cal_mod.email_post_slack_preview = _patched_email_post_slack_preview
_cal_mod.email_post_cancel        = _patched_email_post_cancel

# email_create_calendar: keep real logic but skip actual Google API call
_original_create_calendar_event = _cal_mod._create_calendar_event

def _patched_create_calendar_event(model_output, original_email):
    _divider("CALENDAR EVENT (dry run — Google API skipped)")
    logger.info("  Title     : %s", model_output.get("title"))
    logger.info("  Start     : %s  (%s)",
                model_output.get("start_time"), _fmt_pdt(model_output.get("start_time") or ""))
    logger.info("  End       : %s  (%s)",
                model_output.get("end_time"), _fmt_pdt(model_output.get("end_time") or ""))
    logger.info("  Location  : %s", model_output.get("location"))
    logger.info("  Attendees : %s", model_output.get("attendees"))
    logger.info("[GOOGLE SKIP] Would call calendar.events().insert()")
    return "https://calendar.google.com/event?eid=test-dry-run"

_cal_mod._create_calendar_event = _patched_create_calendar_event


# ─────────────────────────────────────────────────────────────────────────────
# Patch email_classify — log LLM result
# ─────────────────────────────────────────────────────────────────────────────
_original_email_classify = _cal_mod.email_classify

def _patched_email_classify(state):
    _divider("EMAIL CLASSIFY — SYSTEM PROMPT")
    sys_prompt = _cal_mod._build_email_system_prompt()
    logger.info("\n%s", sys_prompt)
    _divider()
    result = _original_email_classify(state)
    _divider("EMAIL CLASSIFY — LLM OUTPUT")
    logger.info("  is_meeting      : %s", result.is_meeting)
    logger.info("  title           : %s", result.meeting_title)
    logger.info("  time_confidence : %s", result.time_confidence)
    logger.info("  start_window    : %s", result.meeting_start)
    logger.info("  end_window      : %s", result.meeting_end)
    logger.info("  attendees       : %s", result.meeting_attendees)
    _divider()
    return result

_cal_mod.email_classify = _patched_email_classify


# ─────────────────────────────────────────────────────────────────────────────
# Patch slot_cod internals — log calendar fetch and CoD rounds
# ─────────────────────────────────────────────────────────────────────────────
import calendar_cod as _cod_mod

_original_load_tokens           = _cod_mod._load_participant_tokens
_original_get_events            = _cod_mod._get_events_with_titles
_original_filter_events_for_day = _cod_mod._filter_events_for_day
_original_find_slots            = _cod_mod._find_slots_with_conflicts
_original_determine_search_days = _cod_mod._determine_search_days
_original_run_cod               = _cod_mod._run_cod
_original_slot_cod              = _cod_mod.slot_cod

_current_participant_name = ["?"]


def _patched_get_events(access_token, time_min, time_max):
    events = _original_get_events(access_token, time_min, time_max)
    name   = _current_participant_name[0]
    _divider(f"CALENDAR FETCH — {name}")
    logger.info("  Window : %s  ->  %s", _fmt_pdt(time_min), _fmt_pdt(time_max))
    logger.info("  %d event(s) returned:", len(events))
    for i, ev in enumerate(events, 1):
        disp = _cod_mod._is_displaceable(ev["title"], ev["is_recurring"])
        flag = "NON-URGENT (displaceable)" if disp else "URGENT"
        logger.info("  #%d  '%s'%s", i, ev["title"], " [recurring]" if ev["is_recurring"] else "")
        logger.info("      %s -> %s  [%s]", _fmt_pdt(ev["start"]), _fmt_pdt(ev["end"]), flag)
    if not events:
        logger.info("  (calendar fully free in this window)")
    _divider()
    return events


def _patched_determine_search_days(state):
    days, label, start_hour, end_hour = _original_determine_search_days(state)
    _divider("SEARCH WINDOW")
    logger.info("  time_confidence : %s", state.time_confidence)
    logger.info("  label           : %s", label)
    logger.info("  hours           : %02d:00 – %02d:00 PDT", start_hour, end_hour)
    logger.info("  days            : %s", [d.strftime("%A %b %d") for d in days])
    _divider()
    return days, label, start_hour, end_hour


def _patched_find_slots(events_by_participant, day, **kwargs):
    free, partial = _original_find_slots(events_by_participant, day, **kwargs)
    logger.info("  SLOT ANALYSIS — %s", day.strftime("%A %b %d"))
    logger.info("    Free slots     : %d", len(free))
    for s in free:
        logger.info("      [FREE]     %s", s["label"])
    logger.info("    Conflict slots : %d", len(partial))
    for s in partial:
        c = s["conflicts"][0]
        disp = "NON-URGENT" if c["is_displaceable"] else "URGENT"
        logger.info("      [CONFLICT] %s  -> %s has '%s' [%s]",
                    s["label"], c["participant"], c["event_title"], disp)
    return free, partial


def _patched_run_cod(enriched_slots, participants, search_label, email_context):
    _divider("CHAIN OF DEBATE")
    logger.info("  Title       : %s", email_context.get("title"))
    logger.info("  Window      : %s -> %s",
                email_context.get("start_window"), email_context.get("end_window"))
    logger.info("  Participants: %s", ", ".join(participants))
    logger.info("  Candidates  : %d slot(s)", len(enriched_slots))

    # Intercept _llm to log each round
    round_counter = [0]
    round_names   = ["Proposer", "Challenger", "Judge"]
    _real_llm     = _cod_mod._llm

    def _logging_llm(structured_output=None, model_name=None):
        idx  = round_counter[0]
        name = round_names[idx] if idx < len(round_names) else f"Round {idx+1}"
        round_counter[0] += 1
        inner = _real_llm(structured_output=structured_output, model_name=model_name)

        class _LoggingWrapper:
            def invoke(self, messages):
                _divider(f"CoD Round {idx+1}: {name}", char="·")
                result = inner.invoke(messages)
                if name == "Proposer":
                    logger.info("  Proposed : %s  (%s)",
                                result.proposed_slot, _fmt_pdt(result.proposed_slot))
                    logger.info("  Argument : %s", result.argument)
                elif name == "Challenger":
                    logger.info("  Status   : %s", "AGREES" if result.agrees else "DISAGREES")
                    logger.info("  Counter  : %s  (%s)",
                                result.counter_slot, _fmt_pdt(result.counter_slot))
                    logger.info("  Argument : %s", result.argument)
                elif name == "Judge":
                    logger.info("  Top %d slots:", len(result.top_slots))
                    for i, s in enumerate(result.top_slots):
                        logger.info("    #%d  %s  (%s)", i + 1, s.start, _fmt_pdt(s.start))
                        logger.info("        %s", s.reason)
                return result
        return _LoggingWrapper()

    _cod_mod._llm = _logging_llm
    try:
        verdict = _original_run_cod(enriched_slots, participants, search_label, email_context)
    finally:
        _cod_mod._llm = _real_llm

    if verdict:
        _divider("COD VERDICT — TOP 3 SLOTS")
        for i, s in enumerate(verdict.top_slots):
            logger.info("  #%d  %s  (%s)", i + 1, s.start, _fmt_pdt(s.start))
            logger.info("      %s", s.reason)
        _divider()
    return verdict


def _patched_slot_cod(state):
    _real_load = _cod_mod._load_participant_tokens

    def _name_tracking_load(state=state):
        tokens = _real_load(state)

        def _named_get(access_token, time_min, time_max):
            for n, t in tokens.items():
                if t == access_token:
                    _current_participant_name[0] = n
                    break
            return _patched_get_events(access_token, time_min, time_max)

        _cod_mod._get_events_with_titles = _named_get
        _divider("TOKEN LOADING")
        logger.info("  Participants: %s", list(tokens.keys()) or "(none)")
        _divider()
        return tokens

    _cod_mod._load_participant_tokens   = _name_tracking_load
    _cod_mod._filter_events_for_day     = _original_filter_events_for_day
    _cod_mod._find_slots_with_conflicts = _patched_find_slots
    _cod_mod._determine_search_days     = _patched_determine_search_days
    _cod_mod._run_cod                   = _patched_run_cod

    try:
        result = _original_slot_cod(state)
    finally:
        _cod_mod._load_participant_tokens   = _original_load_tokens
        _cod_mod._get_events_with_titles    = _original_get_events
        _cod_mod._filter_events_for_day     = _original_filter_events_for_day
        _cod_mod._find_slots_with_conflicts = _original_find_slots
        _cod_mod._determine_search_days     = _original_determine_search_days
        _cod_mod._run_cod                   = _original_run_cod

    return result

_cod_mod.slot_cod = _patched_slot_cod


# ─────────────────────────────────────────────────────────────────────────────
# Simulate slot selection / cancel
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_button_click(cod_state, slot_index: int | None, email_data: dict):
    """
    slot_index: 0, 1, 2 → select that slot
                None     → cancel
    """
    from state import OrchestratorState

    top_slots = cod_state.cod_top_slots

    if slot_index is None:
        _divider(f"SIMULATE — ❌ Cancel")
        logger.info("  [REDIS SKIP] Would record feedback=rejected")
        logger.info("  [SLACK SKIP] Would update card to cancelled")
        return

    if slot_index >= len(top_slots):
        logger.info("  ⚠️  Slot %d not available (only %d slot(s) proposed)",
                    slot_index + 1, len(top_slots))
        return

    chosen = top_slots[slot_index]
    _divider(f"SIMULATE — ✅ Slot {slot_index + 1} selected")
    logger.info("  Start : %s  (%s)", chosen["start"], _fmt_pdt(chosen["start"]))
    logger.info("  End   : %s  (%s)", chosen["end"],   _fmt_pdt(chosen["end"]))
    logger.info("  Reason: %s", chosen.get("reason", ""))

    pending_meeting = {
        "email_data": email_data,
        "model_output": {
            "title":           cod_state.meeting_title,
            "start_time":      chosen["start"],
            "end_time":        chosen["end"],
            "location":        cod_state.meeting_location,
            "attendees":       cod_state.meeting_attendees,
            "time_confidence": cod_state.time_confidence,
        },
        "all_proposed_slots": [
            {"start": s["start"], "end": s["end"]} for s in top_slots
        ],
        "selected_slot_index": slot_index,
    }

    click_state = cod_state.model_copy(update={
        "intent":           "email",
        "slack_event_type": "interactivity",
        "slack_action_id":  "create_meeting",
        "pending_meeting":  pending_meeting,
        "selected_slot":    pending_meeting["model_output"],
        "session_id":       "test-session-id-001",
        "channel_id":       "test-channel",
        "preview_ts":       "test-preview-ts",
    })

    # Invoke email_create_calendar directly (Slack/Google already patched to dry-run)
    from calendar_agent import email_create_calendar
    result = email_create_calendar(click_state)
    logger.info("  calendar_link : %s", result.calendar_link)
    logger.info("  [REDIS SKIP] Would record feedback=selected, slot_index=%d", slot_index)
    _divider()


# ─────────────────────────────────────────────────────────────────────────────
# Fake email
# ─────────────────────────────────────────────────────────────────────────────
PARTICIPANTS = {
    "Agent":    "msadi.finalproject@gmail.com",
    "Aishanee": "aishanee.sinha@sjsu.edu",
    "Soham":    "sohan.juetce@gmail.com",
}

FAKE_EMAIL = {
    "subject":    "Lets meet",
    "from_email": PARTICIPANTS["Aishanee"],
    "to_emails":  [PARTICIPANTS["Agent"]],
    "cc_emails":  [],
    "body": (
        "Hi,\n"
        "Lets meet on next Tuesday to discuss update on workforce assistant project.\n"
        "Regards,\n"
        "Aishanee Sinha"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    from calendar_cod import build_orchestrator
    from state import OrchestratorState

    _divider("END-TO-END TEST: CoD Meeting Scheduler", char="═")
    logger.info("  Subject : %s", FAKE_EMAIL["subject"])
    logger.info("  From    : %s", FAKE_EMAIL["from_email"])
    logger.info("  To      : %s", FAKE_EMAIL["to_emails"])
    logger.info("  Body    :\n%s", FAKE_EMAIL["body"])
    _divider(char="═")

    graph  = build_orchestrator()
    event  = {"headers": {}, "isBase64Encoded": False, "body": json.dumps(FAKE_EMAIL)}
    config = {"configurable": {"thread_id": "test-cod-001"}}

    try:
        result = graph.invoke(OrchestratorState(raw_event=event), config=config)
        final  = OrchestratorState(**result) if isinstance(result, dict) else result
    except Exception:
        logger.exception("Graph raised an exception")
        sys.exit(1)

    _divider("FINAL STATE", char="═")
    logger.info("  intent          : %s", final.intent)
    logger.info("  is_meeting      : %s", final.is_meeting)
    logger.info("  meeting_title   : %s", final.meeting_title)
    logger.info("  time_confidence : %s", final.time_confidence)
    logger.info("  error           : %s", final.error)
    logger.info("  cod_top_slots   : %d slot(s)", len(final.cod_top_slots))
    for i, s in enumerate(final.cod_top_slots):
        logger.info("    #%d  %s  (%s)", i + 1, s["start"], _fmt_pdt(s["start"]))
    _divider(char="═")

    if not final.cod_top_slots:
        logger.info("No slots to simulate — exiting")
        return

    # ── Simulate all 4 outcomes ───────────────────────────────────────────────
    _divider("SIMULATING USER CHOICES", char="═")

    _simulate_button_click(final, 0, FAKE_EMAIL)   # Slot 1
    _simulate_button_click(final, 1, FAKE_EMAIL)   # Slot 2
    _simulate_button_click(final, 2, FAKE_EMAIL)   # Slot 3
    _simulate_button_click(final, None, FAKE_EMAIL) # Cancel

    _divider("TEST COMPLETE", char="═")


if __name__ == "__main__":
    main()
