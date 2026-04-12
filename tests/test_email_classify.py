"""
test_email_classify.py
======================
Tests the email_classify LLM call in isolation — shows the raw structured
output from Qwen before any CoD slot selection happens.

Run:
    ../llm_venv/Scripts/python test_email_classify.py
"""

import json
import logging
import sys
from datetime import date, timedelta

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from calendar_agent import _build_email_system_prompt, EmailMeetingDetails
from state import _llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  |  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("test_email_classify")

MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"

PARTICIPANTS = {
    "Agent":    "msadi.finalproject@gmail.com",
    "Aishanee": "aishanee.sinha@sjsu.edu",
    "Soham":    "sohan.juetce@gmail.com",
}

# ─────────────────────────────────────────────────────────────────────────────
# Test emails
# ─────────────────────────────────────────────────────────────────────────────

def _next_tuesday() -> str:
    today = date.today()
    days_ahead = (1 - today.weekday()) % 7 or 7  # Tuesday = 1
    return (today + timedelta(days=days_ahead)).strftime("%A, %B %d")

TEST_EMAILS = [
    {
        "name": "specific_day_no_time",
        "desc": "Day mentioned (next Tuesday), no clock time",
        "email": {
            "subject":    "Follow-up: Project updates",
            "from_email": PARTICIPANTS["Agent"],
            "to_emails":  [PARTICIPANTS["Aishanee"], PARTICIPANTS["Soham"]],
            "cc_emails":  [],
            "body": (
                f"Hi Aishanee, Soham,\n\n"
                f"I'd like to schedule a Project Kickoff meeting next Monday or Tuesday to align on "
                f"goals, deliverables, and timelines.\n\n"
                f"Could we find a 1-hour slot that works for everyone?\n\n"
                f"Best,\nAgent"
            ),
        },
    },
    {
        "name": "exact_day_and_time",
        "desc": "Exact day and clock time provided",
        "email": {
            "subject":    "Team Sync — Monday 2pm",
            "from_email": PARTICIPANTS["Agent"],
            "to_emails":  [PARTICIPANTS["Aishanee"]],
            "cc_emails":  [],
            "body": (
                "Hi Aishanee,\n\n"
                "Can we sync up this coming Monday at 2pm PST for about an hour?\n\n"
                "Best,\nAgent"
            ),
        },
    },
    {
        "name": "vague_next_week",
        "desc": "No specific day — just 'next week'",
        "email": {
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
        },
    },
    {
        "name": "not_a_meeting",
        "desc": "Not a meeting — weekly report",
        "email": {
            "subject":    "Weekly KPI Report",
            "from_email": PARTICIPANTS["Agent"],
            "to_emails":  [PARTICIPANTS["Aishanee"]],
            "cc_emails":  [],
            "body": "Please find the weekly KPI report attached. No action needed.",
        },
    },
    {
        "name": "next_to_next_week",
        "desc": "Two weeks out — 'week after next'",
        "email": {
            "subject":    "Design Review — week after next",
            "from_email": PARTICIPANTS["Agent"],
            "to_emails":  [PARTICIPANTS["Aishanee"], PARTICIPANTS["Soham"]],
            "cc_emails":  [],
            "body": (
                "Hi Aishanee, Soham,\n\n"
                "I'd like to set up a Design Review session the week after next. "
                "No specific day preference — just somewhere in that week.\n\n"
                "Best,\nAgent"
            ),
        },
    },
    {
        "name": "next_month",
        "desc": "Vague — 'sometime next month'",
        "email": {
            "subject":    "Quarterly Planning — next month",
            "from_email": PARTICIPANTS["Agent"],
            "to_emails":  [PARTICIPANTS["Aishanee"], PARTICIPANTS["Soham"]],
            "cc_emails":  [],
            "body": (
                "Hi Aishanee, Soham,\n\n"
                "Can we schedule a Quarterly Planning session sometime next month? "
                "Flexible on the exact date.\n\n"
                "Best,\nAgent"
            ),
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def classify_email(email: dict) -> EmailMeetingDetails:
    email_text = (
        f"Subject: {email.get('subject', '')}\n"
        f"From: {email.get('from_email', '')}\n"
        f"To: {', '.join(email.get('to_emails', []))}\n"
        f"Cc: {', '.join(email.get('cc_emails', []))}\n\n"
        f"{email.get('body', '')[:2000]}"
    )
    llm = _llm(structured_output=EmailMeetingDetails, model_name=MODEL)
    return llm.invoke([
        SystemMessage(content=_build_email_system_prompt()),
        HumanMessage(content=email_text),
    ])


def main():
    prompt = _build_email_system_prompt()
    logger.info("")
    logger.info("=" * 65)
    logger.info("EMAIL CLASSIFY — RAW LLM OUTPUT TEST")
    logger.info("=" * 65)
    logger.info("System prompt (first 300 chars):")
    logger.info("  %s...", prompt[:300].replace("\n", " "))
    logger.info("")

    for case in TEST_EMAILS:
        logger.info("-" * 65)
        logger.info("CASE : %s", case["name"])
        logger.info("DESC : %s", case["desc"])
        email = case["email"]
        logger.info("EMAIL:")
        logger.info("  Subject : %s", email["subject"])
        logger.info("  From    : %s", email["from_email"])
        logger.info("  To      : %s", ", ".join(email["to_emails"]))
        logger.info("  Body    : %s", email["body"][:120].replace("\n", " "))

        try:
            result = classify_email(email)
            logger.info("OUTPUT:")
            logger.info("  is_meeting      : %s", result.is_meeting)
            logger.info("  title           : %s", result.title)
            logger.info("  start_window    : %s", result.start_window)
            logger.info("  end_window      : %s", result.end_window)
            logger.info("  time_confidence : %s", result.time_confidence)
            logger.info("  attendees       : %s", [str(a) for a in result.attendees])
        except Exception as e:
            logger.error("FAILED: %s", e)

        logger.info("")

    logger.info("=" * 65)
    logger.info("Done.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
