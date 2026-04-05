"""
test_cod_email_flow.py
======================
End-to-end test of the CoD-enhanced calendar flow:

  fake email (meeting intent)
    -> parse_input                (detects direct email, intent=email)
    -> router_agent               (skipped, intent already set)
    -> calendar_subgraph
        -> email_fetch_and_parse  (skips: direct email, data already set)
        -> email_classify         (Qwen extracts meeting details)
        -> slot_cod
            [Feature 1] free slots, or least-conflict slots if none free
            [Feature 2] urgency analysis — displaceable events prioritised
            [Feature 3] week-wide scan if no specific day extracted
            [Proposer -> Challenger -> Judge via Qwen]
        -> email_post_slack_preview  (posts Slack card)

Run:
    ../llm_venv/Scripts/python test_cod_email_flow.py
"""

import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Must load before importing state.py / calendar_cod.py
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging — verbose, timestamped
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  |  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
for noisy in ("httpx", "httpcore", "openai", "urllib3", "requests", "slack_sdk"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("test_cod_email_flow")

LOCAL_TZ = datetime.now().astimezone().tzinfo


def _fmt_pst(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(LOCAL_TZ)
        return dt.strftime("%a %b %d  %I:%M %p %Z")
    except Exception:
        return iso


# ─────────────────────────────────────────────────────────────────────────────
# Load GROUP_EMAILS from .env
# ─────────────────────────────────────────────────────────────────────────────
GROUP_EMAILS = json.loads(os.getenv("GROUP_EMAILS_JSON", "[]"))
if not GROUP_EMAILS:
    logger.warning("GROUP_EMAILS_JSON not set in .env — sender filter will be skipped")

# ─────────────────────────────────────────────────────────────────────────────
# Import cod module first so we can patch its internals before graph build
# ─────────────────────────────────────────────────────────────────────────────
import calendar_cod as _cod_mod

_original_load_tokens           = _cod_mod._load_participant_tokens
_original_get_events            = _cod_mod._get_events_with_titles
_original_filter_events_for_day = _cod_mod._filter_events_for_day
_original_find_slots            = _cod_mod._find_slots_with_conflicts
_original_determine_search_days = _cod_mod._determine_search_days
_original_run_cod               = _cod_mod._run_cod

# ─────────────────────────────────────────────────────────────────────────────
# Patch: email_classify — print is_meeting result immediately after LLM call
# ─────────────────────────────────────────────────────────────────────────────
import calendar_agent as _cal_mod

_original_email_classify = _cal_mod.email_classify

def _patched_email_classify(state):
    result = _original_email_classify(state)
    logger.info("")
    logger.info("=" * 65)
    logger.info("EMAIL CLASSIFY RESULT")
    logger.info("  is_meeting      : %s", result.is_meeting)
    logger.info("  title           : %s", result.meeting_title)
    logger.info("  time_confidence : %s", result.time_confidence)
    logger.info("  start_window    : %s", result.meeting_start)
    logger.info("  end_window      : %s", result.meeting_end)
    logger.info("  attendees       : %s", result.meeting_attendees)
    logger.info("  error           : %s", result.error)
    logger.info("=" * 65)
    return result

_cal_mod.email_classify = _patched_email_classify


# ─────────────────────────────────────────────────────────────────────────────
# Patch: _load_participant_tokens — log which participants loaded
# ─────────────────────────────────────────────────────────────────────────────
def _patched_load_tokens():
    tokens = _original_load_tokens()
    logger.info("")
    logger.info("-" * 65)
    logger.info("STEP 1 — TOKEN LOADING")
    logger.info("  Participants loaded: %s", list(tokens.keys()) or "(none)")
    logger.info("-" * 65)
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Patch: _determine_search_days — log feature 3 decision
# ─────────────────────────────────────────────────────────────────────────────
def _patched_determine_search_days(state):
    days, label, start_hour, end_hour = _original_determine_search_days(state)
    logger.info("")
    logger.info("-" * 65)
    logger.info("STEP 2 — SEARCH WINDOW (Feature 3)")
    logger.info("  time_confidence : %s", state.time_confidence)
    logger.info("  meeting_start   : %s", state.meeting_start)
    logger.info("  Time bounds     : %02d:00 - %02d:00 PST", start_hour, end_hour)
    if len(days) == 1:
        logger.info("  Mode            : Single day — %s", label)
    else:
        logger.info("  Mode            : Full week scan — %s", label)
        for d in days:
            logger.info("    -> %s", d.strftime("%A, %B %d %Y"))
    logger.info("-" * 65)
    return days, label, start_hour, end_hour


# ─────────────────────────────────────────────────────────────────────────────
# Patch: _get_events_with_titles — log each participant's events w/ urgency
# ─────────────────────────────────────────────────────────────────────────────
_current_participant_name = ["?"]  # mutable cell so inner patch can write it


def _patched_get_events(access_token, time_min, time_max):
    events = _original_get_events(access_token, time_min, time_max)
    name   = _current_participant_name[0]
    logger.info("")
    logger.info("  [%s] calendar fetch  %s  ->  %s", name, _fmt_pst(time_min), _fmt_pst(time_max))
    logger.info("  [%s] %d event(s) returned:", name, len(events))
    if events:
        for i, ev in enumerate(events, 1):
            disp      = _cod_mod._is_displaceable(ev["title"], ev["is_recurring"])
            flag      = "NON-URGENT (displaceable)" if disp else "URGENT"
            recur_tag = " [recurring]" if ev["is_recurring"] else ""
            logger.info("    #%d  title     : %s%s", i, ev["title"], recur_tag)
            logger.info("        start     : %s  (%s)", ev["start"], _fmt_pst(ev["start"]))
            logger.info("        end       : %s  (%s)", ev["end"],   _fmt_pst(ev["end"]))
            logger.info("        urgency   : %s", flag)
    else:
        logger.info("    (no events in window — fully free)")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Patch: _find_slots_with_conflicts — log free vs conflict slots (Features 1+2)
# ─────────────────────────────────────────────────────────────────────────────
def _patched_find_slots(events_by_participant, day, **kwargs):
    free, partial = _original_find_slots(events_by_participant, day, **kwargs)

    logger.info("")
    logger.info("  SLOT ANALYSIS for %s:", day.strftime("%A, %B %d %Y"))

    if free:
        logger.info("  Fully-free slots (%d):", len(free))
        for s in free:
            logger.info("    [FREE]    %s", s["label"])
    else:
        logger.info("  Fully-free slots : (none)")

    if partial:
        logger.info("  Single-conflict slots (%d):", len(partial))
        for s in partial:
            c    = s["conflicts"][0]
            disp = "NON-URGENT (displaceable)" if c["is_displaceable"] else "URGENT"
            logger.info(
                "    [CONFLICT] %s  -> %s has '%s' [%s]",
                s["label"], c["participant"], c["event_title"], disp,
            )
    else:
        logger.info("  Single-conflict slots : (none)")

    return free, partial


# ─────────────────────────────────────────────────────────────────────────────
# Patch: _run_cod — intercept each LLM round for detailed output
# ─────────────────────────────────────────────────────────────────────────────
def _patched_run_cod(enriched_slots, participants, search_label, email_context):
    logger.info("")
    logger.info("=" * 65)
    logger.info("CHAIN OF DEBATE - SLOT SELECTION")
    logger.info("=" * 65)
    logger.info("  Meeting title   : %s", email_context.get("title"))
    logger.info("  Attendees       : %s", ", ".join(email_context.get("attendees", [])))
    logger.info("  Search window   : %s  ->  %s",
                email_context.get("start_window"), email_context.get("end_window"))
    logger.info("  Time confidence : %s", email_context.get("time_confidence"))
    logger.info("  CoD label       : %s", search_label)
    logger.info("  Participants    : %s", ", ".join(participants))
    logger.info("  Candidate slots passed to CoD (%d):", len(enriched_slots))
    for s in enriched_slots:
        if s["conflict_count"] == 0:
            logger.info("    [FREE]     %s  %s  (start=%s)",
                        s.get("day_label", ""), s["label"], s["start"])
        else:
            c = s["conflicts"][0]
            disp = "NON-URGENT" if c["is_displaceable"] else "URGENT"
            logger.info("    [CONFLICT] %s  %s  -> %s has '%s' [%s]  (start=%s)",
                        s.get("day_label", ""), s["label"],
                        c["participant"], c["event_title"], disp, s["start"])

    # Intercept _llm calls to log each round
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
                logger.info("")
                logger.info("-- Round %d: %s --", idx + 1, name)
                for m in messages:
                    role = type(m).__name__.replace("Message", "")
                    logger.debug("  [%s prompt] %s", role, str(m.content)[:400])
                result = inner.invoke(messages)
                if name == "Proposer":
                    logger.info("  Proposed slot : %s  (%s)",
                                result.proposed_slot, _fmt_pst(result.proposed_slot))
                    logger.info("  Argument      : %s", result.argument)
                elif name == "Challenger":
                    status = "AGREES" if result.agrees else "DISAGREES"
                    logger.info("  Status        : %s", status)
                    logger.info("  Counter slot  : %s  (%s)",
                                result.counter_slot, _fmt_pst(result.counter_slot))
                    logger.info("  Argument      : %s", result.argument)
                elif name == "Judge":
                    logger.info("  Final start   : %s  (%s)",
                                result.final_slot_start, _fmt_pst(result.final_slot_start))
                    logger.info("  Final end     : %s  (%s)",
                                result.final_slot_end, _fmt_pst(result.final_slot_end))
                    logger.info("  Reason        : %s", result.reason)
                return result

        return _LoggingWrapper()

    _cod_mod._llm = _logging_llm
    try:
        verdict = _original_run_cod(enriched_slots, participants, search_label, email_context)
    finally:
        _cod_mod._llm = _real_llm

    if verdict:
        logger.info("")
        logger.info("=" * 65)
        logger.info("COD VERDICT")
        logger.info("  Final start : %s  (%s)",
                    verdict.final_slot_start, _fmt_pst(verdict.final_slot_start))
        logger.info("  Final end   : %s  (%s)",
                    verdict.final_slot_end, _fmt_pst(verdict.final_slot_end))
        logger.info("  Reason      : %s", verdict.reason)
        logger.info("=" * 65)
    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# Patch: slot_cod node — wire participant name into _get_events_with_titles
# ─────────────────────────────────────────────────────────────────────────────
_original_slot_cod = _cod_mod.slot_cod


def _patched_slot_cod(state):
    # Wrap _load_participant_tokens so we can track the name when fetching events
    _real_load = _cod_mod._load_participant_tokens

    def _name_tracking_load():
        tokens = _real_load()
        # Wrap _get_events_with_titles to inject current name before each call
        def _named_get(access_token, time_min, time_max):
            # Identify name by matching token
            for n, t in tokens.items():
                if t == access_token:
                    _current_participant_name[0] = n
                    break
            return _patched_get_events(access_token, time_min, time_max)

        _cod_mod._get_events_with_titles = _named_get
        return tokens

    # Patch _filter_events_for_day to log per-day event filtering
    def _patched_filter_events_for_day(all_events, day):
        filtered = _original_filter_events_for_day(all_events, day)
        logger.info("")
        logger.info("  [%s] %d event(s) after day-filter:",
                    day.strftime("%A %b %d"), len(filtered))
        for ev in filtered:
            disp = _cod_mod._is_displaceable(ev["title"], ev["is_recurring"])
            flag = "NON-URGENT" if disp else "URGENT"
            recur_tag = " [recurring]" if ev["is_recurring"] else ""
            logger.info("    '%s'%s  %s -> %s  [%s]",
                        ev["title"], recur_tag,
                        _fmt_pst(ev["start"]), _fmt_pst(ev["end"]), flag)
        return filtered

    logger.info("")
    logger.info("-" * 65)
    logger.info("STEP 3 — FULL WINDOW CALENDAR FETCH (from email_classify search window)")
    logger.info("-" * 65)

    _cod_mod._load_participant_tokens    = _name_tracking_load
    _cod_mod._filter_events_for_day      = _patched_filter_events_for_day
    _cod_mod._find_slots_with_conflicts  = _patched_find_slots
    _cod_mod._determine_search_days      = _patched_determine_search_days
    _cod_mod._run_cod                    = _patched_run_cod

    try:
        result = _original_slot_cod(state)
    finally:
        _cod_mod._load_participant_tokens    = _original_load_tokens
        _cod_mod._get_events_with_titles     = _original_get_events
        _cod_mod._filter_events_for_day      = _original_filter_events_for_day
        _cod_mod._find_slots_with_conflicts  = _original_find_slots
        _cod_mod._determine_search_days      = _original_determine_search_days
        _cod_mod._run_cod                    = _original_run_cod

    return result


_cod_mod.slot_cod = _patched_slot_cod


# ─────────────────────────────────────────────────────────────────────────────
# Re-build graph AFTER all patches are in place
# ─────────────────────────────────────────────────────────────────────────────
from calendar_cod import build_orchestrator
from state import OrchestratorState

# ─────────────────────────────────────────────────────────────────────────────
# Fake meeting email — two variants to exercise Features 1-3
# ─────────────────────────────────────────────────────────────────────────────
PARTICIPANTS = {
    "Agent":    "msadi.finalproject@gmail.com",
    "Aishanee": "aishanee.sinha@sjsu.edu",
    "Soham":    "sohan.juetce@gmail.com",
}

# Variant A: specific day mentioned → single-day scan (Feature 1+2)
FAKE_EMAIL_SPECIFIC_DAY = {
    "subject":    "Follow-up: Project updates",
    "from_email": PARTICIPANTS["Agent"],
    "to_emails":  [PARTICIPANTS["Aishanee"], PARTICIPANTS["Soham"]],
    "cc_emails":  [],
    "body": (
        "Hi Aishanee, Soham,\n\n"
        "I'd like to schedule a Project Kickoff meeting next Monday to align on "
        "goals, deliverables, and timelines.\n\n"
        "Could we find a 1-hour slot that works for everyone?\n\n"
        "Best,\nAgent"
    ),
}

# Variant B: vague timing → full-week scan (Feature 3)
FAKE_EMAIL_NEXT_WEEK = {
    "subject":    "Sync up sometime next week?",
    "from_email": PARTICIPANTS["Agent"],
    "to_emails":  [PARTICIPANTS["Aishanee"], PARTICIPANTS["Soham"]],
    "cc_emails":  [],
    "body": (
        "Hi Aishanee, Soham,\n\n"
        "Can we sync up next week to go over the project roadmap? "
        "No specific day in mind — whatever works best for the team.\n\n"
        "Best,\nAgent"
    ),
}

# Real email from Aishanee — reproduces the production trigger
FAKE_EMAIL_REAL = {
    "subject":    "Lets meet",
    "from_email": PARTICIPANTS["Aishanee"],
    "to_emails":  [PARTICIPANTS["Agent"], PARTICIPANTS["Soham"]],
    "cc_emails":  [],
    "body": (
        "Hi,\n"
        "Lets meet on Wednesday to discuss update on workforce assistant project.\n"
        "Regards,\n"
        "Aishanee Sinha"
    ),
}

# Active variant — switch to FAKE_EMAIL_NEXT_WEEK to test Feature 3
FAKE_EMAIL = FAKE_EMAIL_REAL


def _make_event(email: dict) -> dict:
    return {
        "headers":         {},
        "isBase64Encoded": False,
        "body":            json.dumps(email),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("")
    logger.info("=" * 65)
    logger.info("END-TO-END TEST: CoD Meeting Scheduler + Slack Preview")
    logger.info("=" * 65)
    logger.info("Fake email:")
    logger.info("  Subject : %s", FAKE_EMAIL["subject"])
    logger.info("  From    : %s  (%s)", FAKE_EMAIL["from_email"],
                next(k for k, v in PARTICIPANTS.items() if v == FAKE_EMAIL["from_email"]))
    for addr in FAKE_EMAIL["to_emails"]:
        name = next((k for k, v in PARTICIPANTS.items() if v == addr), addr)
        logger.info("  To      : %s  (%s)", addr, name)
    logger.info("  Body    :\n%s", FAKE_EMAIL["body"])

    graph  = build_orchestrator()
    event  = _make_event(FAKE_EMAIL)
    config = {"configurable": {"thread_id": "test-cod-email-flow-001"}}

    logger.info("")
    logger.info("Running graph...")

    try:
        result = graph.invoke(OrchestratorState(raw_event=event), config=config)
        final  = OrchestratorState(**result) if isinstance(result, dict) else result
    except Exception as e:
        logger.exception("Graph raised an exception: %s", e)
        sys.exit(1)

    logger.info("")
    logger.info("=" * 65)
    logger.info("FINAL STATE")
    logger.info("=" * 65)
    logger.info("  intent          : %s", final.intent)
    logger.info("  is_meeting      : %s", final.is_meeting)
    logger.info("  meeting_title   : %s", final.meeting_title)
    logger.info("  time_confidence : %s", final.time_confidence)
    logger.info("  meeting_start   : %s  (%s)",
                final.meeting_start,
                _fmt_pst(str(final.meeting_start)) if final.meeting_start else "-")
    logger.info("  meeting_end     : %s  (%s)",
                final.meeting_end,
                _fmt_pst(str(final.meeting_end)) if final.meeting_end else "-")
    logger.info("  meeting_location: %s", final.meeting_location)
    logger.info("  attendees       : %s", final.meeting_attendees)
    logger.info("  preview_ts      : %s", final.preview_ts)
    logger.info("  error           : %s", final.error)

    logger.info("")
    if final.preview_ts:
        channel = os.getenv("SLACK_NOTIFY_CHANNEL", "(unknown)")
        logger.info("Slack preview posted -> channel=%s  ts=%s", channel, final.preview_ts)
    elif final.error:
        logger.info("Flow ended with error: %s", final.error)
    else:
        logger.info("No Slack preview posted (is_meeting=%s)", final.is_meeting)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
