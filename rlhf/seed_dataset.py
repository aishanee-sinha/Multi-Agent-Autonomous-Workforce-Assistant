"""
seed_dataset.py — Generate synthetic DPO preference pairs for demo
====================================================================
Creates preference pairs without requiring weeks of production feedback.

Strategy:
  1. Define a set of sample inputs (Slack messages, emails, transcript snippets)
  2. For each input, create a "good" (chosen) and "bad" (rejected) response
  3. Good responses: well-structured, complete, correct assignments
  4. Bad responses: missing fields, wrong assignees, poor formatting
  5. Output: JSONL file with {prompt, chosen, rejected} records

Usage:
  python rlhf/seed_dataset.py --flow slack --output datasets/slack_seed.jsonl
  python rlhf/seed_dataset.py --flow email --output datasets/email_seed.jsonl
  python rlhf/seed_dataset.py --flow meeting --output datasets/meeting_seed.jsonl
"""

import argparse
import json
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Seed data: Slack → Jira
# ─────────────────────────────────────────────────────────────────────────────
SLACK_SEED_PAIRS = [
    {
        "prompt": "Set up CI/CD pipeline for the new repo, assign to Soham",
        "chosen": json.dumps({"task_summary": "Set up CI/CD pipeline for the new repository", "assignee": "Soham", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Set up pipeline", "assignee": "Unassigned", "no_action": False}),
    },
    {
        "prompt": "Fix the login bug on the dashboard, assign to Aishanee",
        "chosen": json.dumps({"task_summary": "Fix login bug on the dashboard page", "assignee": "Aishanee", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Fix bug", "assignee": "Unassigned", "no_action": False}),
    },
    {
        "prompt": "hey, what's for lunch today?",
        "chosen": json.dumps({"task_summary": "", "assignee": "", "no_action": True}),
        "rejected": json.dumps({"task_summary": "Get lunch", "assignee": "Unassigned", "no_action": False}),
    },
    {
        "prompt": "Can someone review PR #42 for the auth service? @ketki should take a look",
        "chosen": json.dumps({"task_summary": "Review PR #42 for the auth service", "assignee": "ketki", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Review PR", "assignee": "someone", "no_action": False}),
    },
    {
        "prompt": "Deploy v2.1 to staging environment before Friday, assigning this to Soham",
        "chosen": json.dumps({"task_summary": "Deploy v2.1 to staging environment before Friday", "assignee": "Soham", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Deploy to staging", "assignee": "Unassigned", "no_action": False}),
    },
    {
        "prompt": "Write unit tests for the new payment module — Aishanee can you handle this?",
        "chosen": json.dumps({"task_summary": "Write unit tests for the new payment module", "assignee": "Aishanee", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Write tests", "assignee": "Unassigned", "no_action": False}),
    },
    {
        "prompt": "The server crashed again last night. We need to investigate memory leaks in the API gateway. Ketki, please look into this ASAP",
        "chosen": json.dumps({"task_summary": "Investigate memory leaks in the API gateway causing server crashes", "assignee": "Ketki", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Server crashed", "assignee": "Unassigned", "no_action": False}),
    },
    {
        "prompt": "Good morning team! Hope everyone had a great weekend 😊",
        "chosen": json.dumps({"task_summary": "", "assignee": "", "no_action": True}),
        "rejected": json.dumps({"task_summary": "Good morning", "assignee": "team", "no_action": False}),
    },
    {
        "prompt": "Update the README with installation instructions for the new Docker setup. Soham this is yours",
        "chosen": json.dumps({"task_summary": "Update README with installation instructions for new Docker setup", "assignee": "Soham", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Update README", "assignee": "Unassigned", "no_action": False}),
    },
    {
        "prompt": "We need to migrate the database from PostgreSQL 14 to 16 by end of sprint. Aishanee please coordinate",
        "chosen": json.dumps({"task_summary": "Migrate database from PostgreSQL 14 to 16 by end of sprint", "assignee": "Aishanee", "no_action": False}),
        "rejected": json.dumps({"task_summary": "Database migration", "assignee": "Unassigned", "no_action": False}),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Seed data: Email → Calendar
# ─────────────────────────────────────────────────────────────────────────────
EMAIL_SEED_PAIRS = [
    {
        "prompt": "Subject: Team sync next Monday\nFrom: soham@example.com\nTo: team@example.com\n\nHi team, can we schedule a sync for next Monday morning? Need to discuss the Q2 roadmap.",
        "chosen": json.dumps({"is_meeting": True, "title": "Team Sync - Q2 Roadmap Discussion", "attendees": ["soham@example.com", "team@example.com"], "start_window": "2026-04-13T09:00:00-07:00", "end_window": "2026-04-13T12:00:00-07:00", "time_confidence": "medium"}),
        "rejected": json.dumps({"is_meeting": False, "title": None, "attendees": [], "start_window": None, "end_window": None, "time_confidence": "none"}),
    },
    {
        "prompt": "Subject: Q3 Budget Report\nFrom: finance@example.com\nTo: team@example.com\n\nPlease find attached the Q3 budget report for review. Let me know if you have questions.",
        "chosen": json.dumps({"is_meeting": False, "title": None, "attendees": [], "start_window": None, "end_window": None, "time_confidence": "none"}),
        "rejected": json.dumps({"is_meeting": True, "title": "Q3 Budget Review Meeting", "attendees": ["finance@example.com"], "start_window": "2026-04-14T09:00:00-07:00", "end_window": "2026-04-14T17:00:00-07:00", "time_confidence": "low"}),
    },
    {
        "prompt": "Subject: Design review Thursday afternoon\nFrom: aishanee@example.com\nTo: soham@example.com, ketki@example.com\n\nCan we meet Thursday afternoon around 2pm to review the new UI designs?",
        "chosen": json.dumps({"is_meeting": True, "title": "UI Design Review", "attendees": ["aishanee@example.com", "soham@example.com", "ketki@example.com"], "start_window": "2026-04-16T14:00:00-07:00", "end_window": "2026-04-16T17:00:00-07:00", "time_confidence": "high"}),
        "rejected": json.dumps({"is_meeting": True, "title": "Meeting", "attendees": ["aishanee@example.com"], "start_window": "2026-04-16T09:00:00-07:00", "end_window": "2026-04-16T17:00:00-07:00", "time_confidence": "low"}),
    },
    {
        "prompt": "Subject: Catch up next week\nFrom: ketki@example.com\nTo: team@example.com\n\nLet's find some time next week to catch up on project progress. Any day works for me.",
        "chosen": json.dumps({"is_meeting": True, "title": "Project Progress Catch-up", "attendees": ["ketki@example.com", "team@example.com"], "start_window": "2026-04-13T09:00:00-07:00", "end_window": "2026-04-17T17:00:00-07:00", "time_confidence": "low"}),
        "rejected": json.dumps({"is_meeting": True, "title": "Catch up", "attendees": ["ketki@example.com"], "start_window": "2026-04-13T09:00:00-07:00", "end_window": "2026-04-13T17:00:00-07:00", "time_confidence": "high"}),
    },
    {
        "prompt": "Subject: Server maintenance notification\nFrom: ops@example.com\nTo: all@example.com\n\nScheduled maintenance window: Saturday 2am-6am. No action required.",
        "chosen": json.dumps({"is_meeting": False, "title": None, "attendees": [], "start_window": None, "end_window": None, "time_confidence": "none"}),
        "rejected": json.dumps({"is_meeting": True, "title": "Server Maintenance", "attendees": ["ops@example.com"], "start_window": "2026-04-18T02:00:00-07:00", "end_window": "2026-04-18T06:00:00-07:00", "time_confidence": "high"}),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Seed data: Meeting Summarizer
# ─────────────────────────────────────────────────────────────────────────────
MEETING_SEED_PAIRS = [
    {
        "prompt": json.dumps({"transcript_preview": "Soham: We need to deploy the hotfix by tonight. The production database is corrupted.\nAishanee: I can handle the rollback but we need Ketki to approve.\nKetki: Approved. Let's do it by 8pm."}),
        "chosen": json.dumps({"abstract": "Emergency meeting to address production database corruption. Team agreed on immediate hotfix deployment with rollback plan.", "n_actions": 2, "decisions": "Deploy hotfix by 8pm tonight. Aishanee handles rollback, Ketki approved."}),
        "rejected": json.dumps({"abstract": "Team discussed database.", "n_actions": 0, "decisions": "None identified."}),
    },
    {
        "prompt": json.dumps({"transcript_preview": "Ketki: Let's review the sprint backlog. We have 12 stories left.\nSoham: I think we should prioritize the auth module.\nAishanee: Agreed. I'll update the Jira board.\nKetki: Also, Soham can you update the wiki page when you get a chance?"}),
        "chosen": json.dumps({"abstract": "Sprint backlog review with 12 remaining stories. Team prioritized auth module work. Action items assigned for Jira board update and wiki documentation.", "n_actions": 2, "decisions": "Prioritize auth module. Aishanee updates Jira board. Soham updates wiki."}),
        "rejected": json.dumps({"abstract": "Sprint review meeting.", "n_actions": 1, "decisions": "Update Jira."}),
    },
    {
        "prompt": json.dumps({"transcript_preview": "Aishanee: The API response time has degraded by 40% since last release.\nSoham: I noticed that too. The new caching layer might be misconfigured.\nKetki: Let's roll back the caching changes and investigate.\nSoham: I'll create a performance benchmark suite to catch this earlier."}),
        "chosen": json.dumps({"abstract": "Performance review meeting addressing 40% API response time degradation since last release. Root cause identified as potentially misconfigured caching layer. Team decided to roll back caching changes.", "n_actions": 2, "decisions": "Roll back caching layer changes. Soham creates performance benchmark suite for early detection."}),
        "rejected": json.dumps({"abstract": "API performance discussed.", "n_actions": 0, "decisions": "None."}),
    },
]

SEED_DATA = {
    "slack": SLACK_SEED_PAIRS,
    "email": EMAIL_SEED_PAIRS,
    "meeting": MEETING_SEED_PAIRS,
}


def _wrap_as_dpo_format(flow: str, pairs: list[dict]) -> list[dict]:
    """Wrap raw pairs into DPO-compatible format with chat template."""
    dpo_records = []
    for pair in pairs:
        sys_msg = {
            "slack": "You are a Jira assistant. Extract task details from the user's Slack message.",
            "email": "You are a meeting intent extraction system. Analyze emails and extract meeting intent.",
            "meeting": "You are a meeting summarizer. Produce a structured summary from the transcript.",
        }[flow]

        dpo_records.append({
            "prompt": json.dumps([
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": pair["prompt"]},
            ]),
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })
    return dpo_records


def generate_seed_dataset(flow: str, output_path: str) -> int:
    """Generate seed dataset for a given flow."""
    pairs = SEED_DATA.get(flow)
    if not pairs:
        print(f"Unknown flow: {flow}. Choose from: {list(SEED_DATA.keys())}")
        return 0

    dpo_records = _wrap_as_dpo_format(flow, pairs)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        for record in dpo_records:
            f.write(json.dumps(record) + "\n")

    print(f"Generated {len(dpo_records)} preference pairs for '{flow}' → {output_path}")
    return len(dpo_records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic DPO seed datasets")
    parser.add_argument("--flow", choices=["slack", "email", "meeting", "all"], default="all")
    parser.add_argument("--output-dir", default="rlhf/datasets")
    args = parser.parse_args()

    flows = list(SEED_DATA.keys()) if args.flow == "all" else [args.flow]
    total = 0
    for flow in flows:
        output_path = os.path.join(args.output_dir, f"{flow}_seed.jsonl")
        total += generate_seed_dataset(flow, output_path)
    print(f"\nTotal: {total} preference pairs generated across {len(flows)} flow(s)")


if __name__ == "__main__":
    main()
