#!/usr/bin/env python3
"""
Local smoke test for Slack preview posting.

This script calls slack_post_preview directly with a synthetic state so you can
verify Slack token/channel permissions without running the full orchestrator.
"""

import argparse
import os

import dotenv

dotenv.load_dotenv()

from state import OrchestratorState, SLACK_BOT_TOKEN, SLACK_NOTIFY_CHANNEL
from slack_agent import slack_post_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Slack preview posting only")
    parser.add_argument(
        "--channel",
        default=SLACK_NOTIFY_CHANNEL or os.environ.get("SLACK_TEST_CHANNEL"),
        help="Slack channel ID (defaults to SLACK_NOTIFY_CHANNEL or SLACK_TEST_CHANNEL)",
    )
    parser.add_argument(
        "--summary",
        default="Local test: Slack preview only",
        help="Ticket summary text to post",
    )
    parser.add_argument(
        "--assignee",
        default="U0ALL7MLBFC",
        help="Assignee text shown in preview card",
    )
    parser.add_argument(
        "--thread-ts",
        default=None,
        help="Optional thread timestamp to post inside a thread",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("SLACK_BOT_TOKEN set:", bool(SLACK_BOT_TOKEN))
    print("Channel:", args.channel)

    if not SLACK_BOT_TOKEN:
        print("ERROR: SLACK_BOT_TOKEN is not set")
        return 1

    if not args.channel:
        print("ERROR: Channel is not set")
        print("Set SLACK_NOTIFY_CHANNEL or pass --channel CXXXXXXX")
        return 1

    state = OrchestratorState(
        channel_id=args.channel,
        message_ts=args.thread_ts,
        slack_ticket_summary=args.summary,
        slack_ticket_assignee=args.assignee,
    )

    out = slack_post_preview(state)

    print("preview_ts:", out.preview_ts)
    print("error:", out.error)

    if out.error:
        return 2
    if not out.preview_ts:
        print("ERROR: No preview_ts returned")
        return 3

    print("OK: Slack preview message posted successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
