# CLAUDE.md — Agent Project (Modular)

## Project Overview

Multi-agent LangGraph orchestrator deployed as an AWS Lambda function. Receives events from
Slack (messages + button clicks) and Gmail (Pub/Sub push) and routes them to one of two
human-in-the-loop flows:

- **Slack flow** — extracts a task from a Slack message, proposes a Jira ticket, creates it on approval
- **Email flow** — detects meeting intent in an email, finds the best free slot via Chain-of-Debate, proposes a Google Calendar event, creates it on approval

Both flows post a Slack preview card with Approve/Cancel buttons before taking any action.

---

## Architecture

```
Lambda event (Slack msg / Gmail Pub/Sub / Slack button)
  └─ parse_input          detects event type, sets intent
  └─ router_agent         Qwen LLM classifies intent if still unknown
  └─ [conditional edge]
       ├─ slack_subgraph   extract → preview → [approve] → create Jira ticket
       └─ calendar_subgraph  fetch email → classify → CoD slot → preview → [approve] → create Calendar event
```

### Key Files

| File | Role |
|---|---|
| `state.py` | `OrchestratorState` (shared Pydantic model), `_llm()` factory, all env var constants |
| `orchestrator.py` | Lambda handler, `parse_input`, `router_agent`, full graph assembly |
| `slack_agent.py` | Slack → Jira subgraph |
| `calendar_agent.py` | Email → Calendar subgraph nodes + `email_classify` prompt |
| `calendar_cod.py` | Drop-in calendar subgraph with Chain-of-Debate slot selection (active) |

---

## LLM Setup

- **Inference:** Self-hosted vLLM on EC2, exposed at `http://$EC2_IP:8000/v1`
- **Base model:** `Qwen/Qwen2.5-14B-Instruct-AWQ`
- **LoRA adapters:** `slack` (Jira extraction), `email` (meeting classification)
- **Client:** `ChatOpenAI` from `langchain-openai`, pointed at the EC2 endpoint
- `_llm()` in `state.py` is the single factory — import `state.py` **after** `load_dotenv()`, because `EC2_IP` is read at module level

---

## Calendar Flow (CoD)

The active calendar subgraph is in `calendar_cod.py`, not `calendar_agent.py`.

### email_classify output

`email_classify` (Qwen LLM) returns `EmailMeetingDetails`:
- `start_window` / `end_window` — ISO 8601 datetimes in **local timezone** defining the search range
- These are a **search hint only** — they tell CoD which days/hours to scan, not the final meeting time

### slot_cod node

Inserted between `email_classify` and `email_post_slack_preview`:

1. Reads `start_window`/`end_window` from state as the calendar fetch bounds
2. Fetches events for the **full window** in one API call per participant (`tokens_*.json`)
3. Filters events per day, finds free and single-conflict slots
4. Runs 3-round Chain-of-Debate (Proposer → Challenger → Judge) using email context + calendar data
5. CoD judge is the **sole authority** on `meeting_start`/`meeting_end` — email values never leak through

### CoD candidate tiers

1. Fully-free slots (all participants free)
2. Single-conflict slots where the conflict is a non-urgent/displaceable event (lunch, 1:1, weekly standup, etc.)
3. No viable slots → sets `meeting_start`/`meeting_end` to `None` (Slack preview shows TBD)

---

## Timezone Handling

- All times use the **local system timezone**, derived at runtime via `datetime.now().astimezone()`
- `LOCAL_TZ` in `calendar_cod.py` — used for all datetime construction and event time conversion
- `_tz_offset` and `_tz_name` in `_build_email_system_prompt()` — injected into the LLM prompt so the model outputs datetimes with the correct local UTC offset
- Slot labels use `cursor.strftime("%Z")` so the timezone abbreviation is dynamic

---

## Participant Token Files

`calendar_cod.py` looks for `tokens_*.json` files in the **current working directory**:

```
tokens_Agent.json
tokens_Aishanee.json
tokens_Soham.json
```

Each file must contain a `refresh_token` field. Tokens are refreshed at runtime using
`GOOGLE_CALENDAR_CLIENT_ID` and `GOOGLE_CALENDAR_CLIENT_SECRET`.

---

## Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `EC2_IP` | `state.py` | vLLM endpoint IP |
| `SLACK_BOT_TOKEN` | `state.py` | Slack Web API auth |
| `SLACK_NOTIFY_CHANNEL` | `state.py` | Channel to post previews |
| `GOOGLE_TOKEN_JSON` | `calendar_agent.py` | Gmail + Calendar OAuth for the agent account |
| `GOOGLE_CALENDAR_CLIENT_ID` | `calendar_cod.py` | OAuth client for refreshing participant tokens |
| `GOOGLE_CALENDAR_CLIENT_SECRET` | `calendar_cod.py` | OAuth client secret |
| `JIRA_BASE_URL` | `state.py` | Jira instance URL |
| `JIRA_EMAIL` | `state.py` | Jira auth email |
| `JIRA_API_TOKEN` | `state.py` | Jira API token |
| `JIRA_PROJECT_KEY` | `state.py` | Default Jira project |
| `JIRA_ISSUE_TYPE` | `state.py` | Default issue type |
| `TEAM_MAP_JSON` | `state.py` | JSON mapping Slack user IDs → Jira account IDs |
| `GROUP_EMAILS_JSON` | `state.py` | Email whitelist for meeting detection |

For local runs, these are loaded from a `.env` file via `python-dotenv`.

---

## Running Tests

Use the `llm_venv` virtualenv (not the system Python):

```bash
# Activate venv (Windows)
../llm_venv/Scripts/activate

# Test email_classify in isolation — shows raw LLM output for 6 email cases
python test_email_classify.py

# End-to-end CoD flow — fake email → calendar fetch → CoD → Slack preview
python test_cod_email_flow.py

# Standalone CoD scheduler — fetches real calendars, runs debate
python test_cod_scheduler.py

# Local orchestrator test — simulates a Slack message event
python run_local.py
```

Switch the active email variant in `test_cod_email_flow.py`:
```python
FAKE_EMAIL = FAKE_EMAIL_SPECIFIC_DAY   # single-day scan
FAKE_EMAIL = FAKE_EMAIL_NEXT_WEEK      # full-week scan
```

---

## Deployment

```bash
# Build and push Docker image to ECR, then update Lambda
bash deploy.sh
```

Lambda configuration:
- Runtime: Docker (Python 3.12)
- Timeout: 60 seconds
- Handler: `orchestrator.handler` (or `calendar_cod.handler` for CoD)
- Trigger: Lambda Function URL (no auth)

The same URL is used for:
- Slack Event Subscriptions
- Slack Interactivity & Shortcuts
- Gmail Pub/Sub push subscription

---

## Stateless Design

No database. All button state is embedded in the Slack block's button `value` field as JSON:

```json
{
  "email_data": { ... },
  "model_output": { "title": "...", "start_time": "...", ... }
}
```

On button click, `parse_input` deserializes this JSON back into state. No DynamoDB or Redis needed.

---

## Retry Guard

Slack retries failed requests with the `x-slack-retry-num` header. `parse_input` detects this
and sets `intent=none` immediately, preventing duplicate Jira tickets or calendar events.

---

## OrchestratorState Fields (key ones)

| Field | Type | Purpose |
|---|---|---|
| `intent` | `str` | `slack / email / none / unknown` |
| `email_data` | `dict` | Raw parsed email (subject, from, to, body) |
| `is_meeting` | `bool` | Whether email_classify detected a meeting |
| `meeting_title` | `str` | Meeting title from email_classify |
| `meeting_start` | `str` | ISO datetime — search window start (email) or final time (after CoD) |
| `meeting_end` | `str` | ISO datetime — search window end (email) or final time (after CoD) |
| `time_confidence` | `str` | `high / medium / low / none` |
| `meeting_attendees` | `list[str]` | Email addresses from email_classify |
| `preview_ts` | `str` | Slack message timestamp of the preview card |
| `pending_meeting` | `dict` | Reconstructed from button value on approval click |
| `jira_key` | `str` | Created Jira issue key |
| `calendar_link` | `str` | Google Calendar event HTML link |
| `error` | `str` | Last error message |
