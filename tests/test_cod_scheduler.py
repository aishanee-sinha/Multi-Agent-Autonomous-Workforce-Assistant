"""
test_cod_scheduler.py — Chain of Debate meeting scheduler
==========================================================
Finds free slots next Monday 9am–5pm for Agent, Aishanee, and Sohan,
then runs a 3-round Chain of Debate (Proposer → Challenger → Judge)
using the Qwen model via _llm() from state.py to pick the best slot.
"""

import json
import os
import requests
from datetime import datetime, timezone, timedelta, date

from dotenv import load_dotenv

# Must load env BEFORE importing state.py — it reads EC2_IP at module level
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from state import _llm

CLIENT_ID     = os.getenv("GOOGLE_CALENDAR_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CALENDAR_CLIENT_SECRET")
MODEL         = "Qwen/Qwen2.5-14B-Instruct-AWQ"

PARTICIPANTS = ["Agent", "Aishanee", "Soham"]

# Calendars are in PST/PDT. March 30 is after DST spring-forward (2nd Sunday of
# March), so the offset is PDT = UTC-7.
PDT = timezone(timedelta(hours=-7))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def next_monday() -> date:
    today = date.today()
    days_ahead = (7 - today.weekday()) % 7  # Monday = 0
    if days_ahead == 0:
        days_ahead = 7
    return today + timedelta(days=days_ahead)


def refresh_token(refresh_token_str: str) -> str:
    resp = requests.post("https://oauth2.googleapis.com/token", data={
        "client_id":     CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token_str,
        "grant_type":    "refresh_token",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_busy_slots(access_token: str, time_min: str, time_max: str) -> list[dict]:
    """Returns list of {start, end} busy blocks for the primary calendar."""
    resp = requests.post(
        "https://www.googleapis.com/calendar/v3/freeBusy",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "timeMin": time_min,
            "timeMax": time_max,
            "items":   [{"id": "primary"}],
        },
    )
    resp.raise_for_status()
    return resp.json().get("calendars", {}).get("primary", {}).get("busy", [])


def compute_free_slots(
    busy_all: dict[str, list[dict]],
    day: date,
    start_hour: int = 9,
    end_hour: int = 17,
    slot_minutes: int = 60,
) -> list[str]:
    """
    Returns a list of human-readable free slot strings in PDT
    where ALL participants are free for at least slot_minutes.
    """
    window_start = datetime(day.year, day.month, day.day, start_hour, 0, tzinfo=PDT)
    window_end   = datetime(day.year, day.month, day.day, end_hour,   0, tzinfo=PDT)

    # Collect all busy intervals across everyone (normalize to PDT for comparison)
    busy_intervals: list[tuple[datetime, datetime]] = []
    for blocks in busy_all.values():
        for block in blocks:
            s = datetime.fromisoformat(block["start"].replace("Z", "+00:00")).astimezone(PDT)
            e = datetime.fromisoformat(block["end"].replace("Z", "+00:00")).astimezone(PDT)
            busy_intervals.append((s, e))

    # Walk in slot_minutes increments
    free_slots = []
    cursor = window_start
    delta  = timedelta(minutes=slot_minutes)
    while cursor + delta <= window_end:
        slot_end = cursor + delta
        conflict = any(s < slot_end and e > cursor for s, e in busy_intervals)
        if not conflict:
            free_slots.append(
                f"{cursor.strftime('%I:%M %p')} - {slot_end.strftime('%I:%M %p')} PDT"
            )
        cursor += delta

    return free_slots


# ─────────────────────────────────────────────────────────────────────────────
# CoD structured output schemas
# ─────────────────────────────────────────────────────────────────────────────

class SlotProposal(BaseModel):
    proposed_slot: str = Field(description="The time slot being proposed, e.g. '10:00 AM – 11:00 AM UTC'")
    argument:      str = Field(description="1–3 sentence argument for why this slot is best")


class SlotChallenge(BaseModel):
    agrees:          bool  = Field(description="True if challenger agrees with the proposal")
    counter_slot:    str   = Field(description="Alternative slot if disagreeing, otherwise same as proposed")
    argument:        str   = Field(description="1–3 sentence challenge or concession")


class SlotVerdict(BaseModel):
    final_slot:  str = Field(description="The final agreed meeting slot")
    reason:      str = Field(description="One sentence rationale")


# ─────────────────────────────────────────────────────────────────────────────
# Chain of Debate
# ─────────────────────────────────────────────────────────────────────────────

def run_chain_of_debate(
    free_slots: list[str],
    participants: list[str],
    meeting_date: date,
) -> SlotVerdict:
    slots_text      = "\n".join(f"  - {s}" for s in free_slots)
    participants_str = ", ".join(participants)
    context = (
        f"Meeting date: {meeting_date.strftime('%A, %B %d %Y')} (next Monday)\n"
        f"Participants: {participants_str}\n"
        f"Available free slots (all participants free):\n{slots_text}"
    )

    print("\n" + "="*60)
    print("CHAIN OF DEBATE - MEETING SCHEDULER")
    print("="*60)
    print(f"\nContext:\n{context}\n")

    # -- Round 1: Proposer ---------------------------------------------------
    print("-- Round 1: Proposer --")
    proposer_sys = (
        "You are the Proposer in a meeting scheduling debate.\n"
        "Given a list of free slots where all participants are available, "
        "pick the single best slot and argue for it in 1–3 sentences.\n"
        "Prefer mid-morning slots (10–11am) as they tend to be most productive.\n"
        "Return a SlotProposal JSON."
    )
    proposer_llm  = _llm(structured_output=SlotProposal, model_name=MODEL)
    proposal: SlotProposal = proposer_llm.invoke([
        SystemMessage(content=proposer_sys),
        HumanMessage(content=context),
    ])
    print(f"  Proposed slot : {proposal.proposed_slot}")
    print(f"  Argument      : {proposal.argument}\n")

    # -- Round 2: Challenger -------------------------------------------------
    print("-- Round 2: Challenger --")
    challenger_sys = (
        "You are the Challenger in a meeting scheduling debate.\n"
        "Review the Proposer's slot choice. If you agree, say so and keep the slot. "
        "If you think a different slot is better (e.g. after lunch for async teams, "
        "or earlier to leave afternoon free), propose an alternative and argue for it.\n"
        "Return a SlotChallenge JSON."
    )
    challenger_user = (
        f"{context}\n\n"
        f"Proposer chose: {proposal.proposed_slot}\n"
        f"Proposer argument: {proposal.argument}"
    )
    challenger_llm     = _llm(structured_output=SlotChallenge, model_name=MODEL)
    challenge: SlotChallenge = challenger_llm.invoke([
        SystemMessage(content=challenger_sys),
        HumanMessage(content=challenger_user),
    ])
    status = "AGREES" if challenge.agrees else "DISAGREES"
    print(f"  Status        : {status}")
    print(f"  Counter slot  : {challenge.counter_slot}")
    print(f"  Argument      : {challenge.argument}\n")

    # -- Round 3: Judge ------------------------------------------------------
    print("-- Round 3: Judge --")
    judge_sys = (
        "You are the Judge in a meeting scheduling debate.\n"
        "Read both the Proposer's and Challenger's arguments and pick the final slot. "
        "The slot MUST come from the free slots list — do not invent a new time.\n"
        "Return a SlotVerdict JSON with final_slot and a one-sentence reason."
    )
    judge_user = (
        f"{context}\n\n"
        f"Proposer chose    : {proposal.proposed_slot}\n"
        f"Proposer argument : {proposal.argument}\n\n"
        f"Challenger {'agreed' if challenge.agrees else 'countered with'}: {challenge.counter_slot}\n"
        f"Challenger argument: {challenge.argument}"
    )
    judge_llm       = _llm(structured_output=SlotVerdict, model_name=MODEL)
    verdict: SlotVerdict = judge_llm.invoke([
        SystemMessage(content=judge_sys),
        HumanMessage(content=judge_user),
    ])
    print(f"  Final slot    : {verdict.final_slot}")
    print(f"  Reason        : {verdict.reason}\n")

    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    monday   = next_monday()
    time_min = datetime(monday.year, monday.month, monday.day, 9,  0, tzinfo=PDT).isoformat()
    time_max = datetime(monday.year, monday.month, monday.day, 17, 0, tzinfo=PDT).isoformat()

    print(f"Checking calendars for {monday.strftime('%A, %B %d %Y')} (9am-5pm PDT)...")

    busy_all: dict[str, list[dict]] = {}

    for name in PARTICIPANTS:
        token_file = f"tokens_{name}.json"
        if not os.path.exists(token_file):
            print(f"  [{name}] token file not found — skipping")
            continue
        with open(token_file) as f:
            tokens = json.load(f)
        try:
            access_token = refresh_token(tokens["refresh_token"])
            busy         = get_busy_slots(access_token, time_min, time_max)
            busy_all[name] = busy
            print(f"  [{name}] {len(busy)} busy block(s)")
        except Exception as e:
            print(f"  [{name}] error fetching calendar: {e}")

    if not busy_all:
        print("No calendars loaded — aborting.")
        return

    free_slots = compute_free_slots(busy_all, monday)
    if not free_slots:
        print("\nNo overlapping free slots found next Monday 9am-5pm PDT.")
        return

    print(f"\nOverlapping free slots ({len(free_slots)}):")
    for s in free_slots:
        print(f"  {s}")

    verdict = run_chain_of_debate(free_slots, list(busy_all.keys()), monday)

    print("="*60)
    print(f"FINAL DECISION: {verdict.final_slot}")
    print(f"REASON        : {verdict.reason}")
    print("="*60)


if __name__ == "__main__":
    main()
