# Real Email End-to-End Test Report
**Date:** 2026-04-09
**Model:** Qwen/Qwen2.5-14B-Instruct-AWQ (base, no LoRA)
**System timezone:** PDT (-07:00)
**Lambda/server endpoint:** Local invocation (no Lambda/ngrok — emails fetched via Gmail API, fed directly to handler)

## Infrastructure Notes

| Check | Status |
|---|---|
| Gmail Pub/Sub firing? | N/A — emails fetched directly via Gmail API (no AWS credentials for SSM) |
| historyId advancing in SSM? | N/A — no AWS credentials available locally |
| Google Calendar API reachable? | YES — calendar fetch returned 0 events (clean calendars) |
| Slack API reachable? | NO — `SLACK_BOT_TOKEN` is empty in .env → `not_authed` error |
| vLLM EC2 responding? | YES — `Qwen/Qwen2.5-14B-Instruct-AWQ` + 3 LoRA adapters loaded |
| Any SSL / network errors? | Slack SSL cert verification failed on first attempt; Slack `not_authed` on subsequent runs |

**Workaround applied:** Since AWS SSM is unavailable and `SLACK_BOT_TOKEN` is not set, emails were fetched directly from Gmail API using `GOOGLE_TOKEN_JSON`, parsed via `_parse_gmail_message()`, and fed through `handler()` as direct email events. The full email → classify → CoD pipeline ran successfully; only the final Slack posting step failed (non-timezone-related).

---

## Email 1 — "Quick sync next Monday at 2pm?"

### Raw Email Data (from Gmail API)
```json
{
  "message_id": "19d73812b2f0ef57",
  "thread_id": "19d73812b2f0ef57",
  "subject": "Quick sync next Monday at 2pm?",
  "from_email": "leelaprasad.dammalapati@sjsu.edu",
  "to_emails": ["msadi.finalproject@gmail.com"],
  "cc_emails": [],
  "date": "Thu, 9 Apr 2026 11:28:45 -0700",
  "body": "Hi,\r\n Can we meet next Monday at 2pm to go over the project updates? Should only\r\ntake about an hour. Thanks, Leela\r\n"
}
```

### Log Values
| Field | Value |
|---|---|
| Email received by parse? | YES |
| `is_meeting` | True |
| `time_confidence` | high |
| `start_window` from classify | 2026-04-13T14:00:00-07:00 |
| `end_window` from classify | 2026-04-13T17:00:00-07:00 |
| Days scanned | Mon Apr 13 (1 day) |
| Calendar fetch worked? | YES — 0 events for both participants |
| Free slots found | 3 fully-free slots (14:00, 15:00, 16:00) |
| CoD proposer | 2026-04-13T14:00:00-07:00 — "The email specifies a high preference for the meeting to occur on Monday at 2pm" |
| CoD challenger | agrees=True |
| CoD verdict start | 2026-04-13T14:00:00-07:00 |
| CoD verdict end | 2026-04-13T15:00:00-07:00 |
| CoD verdict reason | "The slot at 2:00 PM on April 13th is marked as [ALL FREE], aligning perfectly with the email intent and attendees' availability." |
| Slack card posted? | NO — `not_authed` (empty SLACK_BOT_TOKEN) |
| Any errors | Slack API: `{'ok': False, 'error': 'not_authed'}` |

### Slack Card (would have posted)
| Field | Value | Correct? |
|---|---|---|
| Title shown | Quick sync to go over the project updates | YES |
| Start time shown | 2026-04-13T14:00:00-07:00 | YES |
| End time shown | 2026-04-13T15:00:00-07:00 | YES |
| Day correct? | Monday Apr 13 | YES |
| Time correct? | 2:00 PM | YES |
| Timezone shown | -07:00 (PDT) | YES |
| Attendees | leelaprasad.dammalapati@sjsu.edu, msadi.finalproject@gmail.com | YES |
| Confidence | high | YES |

**Notes:** Pipeline worked perfectly end-to-end with a real email. The base model correctly identified "next Monday at 2pm", produced the right timezone offset (-07:00), set appropriate start/end windows, and CoD chose the exact requested slot. All fields match expectations. Only failure was Slack posting due to missing bot token.

---

## Email 2 — "Sync next week?"

### Raw Email Data (from Gmail API)
```json
{
  "message_id": "19d7382063701d3f",
  "thread_id": "19d7382063701d3f",
  "subject": "Sync next week?",
  "from_email": "leelaprasad.dammalapati@sjsu.edu",
  "to_emails": ["msadi.finalproject@gmail.com"],
  "cc_emails": [],
  "date": "Thu, 9 Apr 2026 11:29:42 -0700",
  "body": "Hi ,\r\ncan we sync up sometime next week?\r\n\r\nthanks ,\r\nleela\r\n"
}
```

### Log Values
| Field | Value |
|---|---|
| Email received by parse? | YES |
| `is_meeting` | True |
| `time_confidence` | low |
| `start_window` from classify | 2026-04-13T09:00:00-07:00 |
| `end_window` from classify | **2026-04-10T17:00:00-07:00** |
| Days scanned | **Mon Apr 13 only (1 day)** |
| Calendar fetch worked? | YES — 0 events for both participants |
| Free slots found | 8 fully-free slots (09:00–16:00) |
| CoD proposer | 2026-04-13T10:00:00-07:00 — "Given the email context does not specify a preferred time, and all slots are free, I propose the mid[-morning slot]" |
| CoD challenger | agrees=True |
| CoD verdict start | 2026-04-13T10:00:00-07:00 |
| CoD verdict end | 2026-04-13T11:00:00-07:00 |
| CoD verdict reason | "The slot at 10:00 AM is chosen because it is an available time that aligns with the email intent without specifying a preferred time." |
| Slack card posted? | NO — `not_authed` |
| Any errors | Slack API: `{'ok': False, 'error': 'not_authed'}` + **end_window < start_window (see below)** |

### Slack Card (would have posted)
| Field | Value | Correct? |
|---|---|---|
| Title shown | Sync up | YES |
| Start time shown | 2026-04-13T10:00:00-07:00 | Partial — Monday is right, but only 1 day was searched |
| End time shown | 2026-04-13T11:00:00-07:00 | Partial |
| Day correct? | Monday Apr 13 only | **NO — "next week" should span Mon–Fri (Apr 13–17)** |
| Time correct? | 10:00 AM (within 9–5 range) | YES (for searched day) |
| Timezone shown | -07:00 (PDT) | YES |
| Attendees | msadi.finalproject@gmail.com, leelaprasad.dammalapati@sjsu.edu | YES |
| Confidence | low | YES |

**Notes:**
- **BASE MODEL BUG (same as synthetic test):** `end_window` = `2026-04-10T17:00:00-07:00` (this Friday, April 10) is BEFORE `start_window` = `2026-04-13T09:00:00-07:00` (next Monday). The model returned an inverted date range for "next week".
- **CODE BUG (confirmed):** `_determine_search_days` doesn't validate `end_date >= start_date`. When `end_date (Apr 10) < start_date (Apr 13)`, the loop `while cursor <= end_date` finds no weekdays, falls back to `start_date` only. Result: searched only Monday instead of Mon–Fri.
- **Calendar fetch window is also inverted:** `fetching calendars for search window 2026-04-13T09:00:00-07:00 -> 2026-04-10T17:00:00-07:00`. The Google Calendar API appears to return 0 events for both — unclear if it handles inverted ranges gracefully or just returns nothing.
- The CoD selected a reasonable slot (Mon 10am) for the single day it searched, but missed Tue–Fri entirely.
- This confirms the same bug found in the synthetic test (TIMEZONE_TEST_REPORT.md, Bug #3 and #5).

---

## Email 3 — "Let's connect"

**NOT TESTED** — This email was not found in the Gmail inbox. It may not have been sent yet. The previous synthetic test (TIMEZONE_TEST_REPORT.md, Section 5) covered this scenario using a direct handler invocation and found:
- Base model assigned `medium` confidence with a specific date (tomorrow) for an email with no date reference
- The next-week fallback never triggered

---

## Issues Found

| # | Email | Step where it failed | What happened | Expected |
|---|---|---|---|---|
| 1 | Email 2 | `email_classify` | LLM returned `end_window=2026-04-10` (this Fri) instead of `2026-04-17` (next Fri) | `end_window` should be >= `start_window`, spanning the full next week |
| 2 | Email 2 | `_determine_search_days` | Inverted date range (end < start) caused silent fallback to single-day search | Code should validate `end_date >= start_date` and swap/extend if inverted |
| 3 | Email 2 | Calendar fetch | Calendar API queried with inverted window (`start > end`); returned 0 events | May have silently returned empty set; should validate before calling API |
| 4 | Both | `email_post_slack_preview` | Slack API returned `not_authed` | `SLACK_BOT_TOKEN` is empty in .env — needs to be configured |
| 5 | N/A | Infrastructure | No AWS credentials — SSM history tracking unavailable | Needed for production Pub/Sub flow; local testing workaround used |
| 6 | Email 3 | N/A | Email not found in inbox — not sent | User should send third test email to complete the test |

---

## Pub/Sub + Infrastructure Notes

| Check | Status |
|---|---|
| Gmail Pub/Sub firing? | N/A — bypassed (no AWS SSM for history checkpoint) |
| historyId advancing in SSM? | N/A — no AWS credentials |
| Google Calendar API reachable? | YES |
| Slack API reachable? | YES (network), but NOT authenticated (empty token) |
| vLLM EC2 responding? | YES |
| Any SSL / network errors? | SSL_CERT_FILE workaround needed for Python 3.13 on macOS |

---

## Comparison: Real Emails vs. Synthetic Tests

| Test | Synthetic (TIMEZONE_TEST_REPORT) | Real Email (this report) | Match? |
|---|---|---|---|
| Email 1 — confidence | high | high | YES |
| Email 1 — start_window | 2026-04-13T14:00:00-07:00 | 2026-04-13T14:00:00-07:00 | YES |
| Email 1 — end_window | 2026-04-13T17:00:00-07:00 | 2026-04-13T17:00:00-07:00 | YES |
| Email 1 — CoD verdict | 2026-04-13T14:00:00-07:00 | 2026-04-13T14:00:00-07:00 | YES |
| Email 2 — confidence | low | low | YES |
| Email 2 — start_window | 2026-04-13T09:00:00-07:00 | 2026-04-13T09:00:00-07:00 | YES |
| Email 2 — end_window (BUG) | 2026-04-10T17:00:00-07:00 | 2026-04-10T17:00:00-07:00 | YES — same bug reproduced |
| Email 2 — days searched | 1 (Mon only) | 1 (Mon only) | YES — same degradation |
| Email 2 — CoD verdict | 2026-04-13T09:00:00-07:00 | 2026-04-13T10:00:00-07:00 | Different slot, same day |

**Conclusion:** Real emails reproduce the same behavior as synthetic tests. The `end_window` inversion bug in Email 2 is deterministic — the base model consistently returns `2026-04-10` (this Friday) instead of `2026-04-17` (next Friday) for "next week" emails. This confirms it is a base model classification issue, not a one-off hallucination.
