# CLAUDE.md — Agent Project (Modular)

## Project Overview

Multi-agent LangGraph orchestrator deployed as a pair of AWS Lambda functions. Receives events from
Slack (messages + button clicks) and Gmail (Pub/Sub push) and routes them to one of two
human-in-the-loop flows:

- **Slack flow** — extracts a task from a Slack message, proposes a Jira ticket, creates it on approval
- **Email flow** — detects meeting intent in an email, finds the best free slot via Chain-of-Debate, proposes a Google Calendar event, creates it on approval

Both flows post a Slack preview card with Approve/Cancel buttons before taking any action.

---

## Architecture

### Two-Lambda Pattern

```
Lambda Function URL
  └─ lambda_ingest.handler
       ├─ Slack URL verification  → 200 + challenge (synchronous)
       ├─ Slack retry             → 200 dropped (never enqueued)
       └─ Everything else         → SQS FIFO enqueue → 200
                                          │
                                   SQS trigger
                                          │
                              calendar_cod.sqs_handler
                                          │
                              calendar_cod.handler (per record)
                                          │
                              parse_input          detects event type
                              router_agent         Qwen LLM classifies unknown intent
                              [conditional edge]
                                   ├─ slack_subgraph    extract → preview → [approve] → create Jira
                                   └─ calendar_subgraph fetch email → classify → CoD slot → preview → [approve] → create Calendar event
```

### Key Files

| File | Role |
|---|---|
| `state.py` | `OrchestratorState` (shared Pydantic model), `_llm()` factory, all env var constants |
| `lambda_ingest.py` | Ingest Lambda — URL verify, retry drop, SQS enqueue |
| `calendar_cod.py` | Worker Lambda — `parse_input`, `router_agent`, full graph assembly, CoD slot selection |
| `slack_agent.py` | Slack → Jira subgraph nodes |
| `calendar_agent.py` | Email → Calendar subgraph nodes + `email_classify` prompt (imported by `calendar_cod.py`) |
| `gmail_history.py` | SSM-backed Gmail historyId persistence |
| `gmail_watcher.py` | One-time script to register Gmail Pub/Sub watch (re-run every 7 days) |

> **Note:** `orchestrator.py` has been removed. All routing logic (`parse_input`, `router_agent`,
> `route_to_agent`, full graph assembly) now lives in `calendar_cod.py`.

> **Note:** `lambda_function.py` and `lambda_function_*.py` are legacy monolithic Lambdas kept for
> reference only. The active entry points are `lambda_ingest.py` and `calendar_cod.py`.

---

## LLM Setup

- **Inference:** Self-hosted vLLM on EC2, exposed at `http://$EC2_IP:8000/v1`
- **Base model:** `Qwen/Qwen2.5-14B-Instruct-AWQ`
- **LoRA adapters:** `slack` (Jira extraction), `email` (meeting classification)
- **Client:** `ChatOpenAI` from `langchain-openai`, pointed at the EC2 endpoint
- `_llm()` in `state.py` is the single factory — import `state.py` **after** `load_dotenv()`, because `EC2_IP` is read at module level

---

## Calendar Flow (CoD)

The active calendar subgraph is built by `build_calendar_subgraph_cod()` in `calendar_cod.py`.
`calendar_agent.py` provides the individual nodes (`email_fetch_and_parse`, `email_classify`, etc.)
which are imported and assembled there.

### email_classify output

`email_classify` (Qwen LLM) returns `EmailMeetingDetails`:
- `start_window` / `end_window` — ISO 8601 datetimes in **local timezone** defining the search range
- These are a **search hint only** — they tell CoD which days/hours to scan, not the final meeting time

### slot_cod node

Inserted between `email_classify` and `email_post_slack_preview`:

1. Reads `start_window`/`end_window` from state as the calendar fetch bounds
2. Loads participant access tokens from `CALENDAR_TOKENS_JSON` env var via `_load_participant_tokens()`
3. Fetches events for the **full window** in one API call per participant
4. Filters events per day, finds free and single-conflict slots
5. Runs 3-round Chain-of-Debate (Proposer → Challenger → Judge) using email context + calendar data
6. CoD judge is the **sole authority** on `meeting_start`/`meeting_end` — email values never leak through

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

Participant tokens are stored in the `CALENDAR_TOKENS_JSON` environment variable as a JSON object
mapping email address → refresh token string:

```json
{
  "msadi.finalproject@gmail.com": "1//0abc...",
  "aishanee.sinha@sjsu.edu":      "1//0def...",
  "sohan.juetce@gmail.com":       "1//0ghi..."
}
```

`_load_participant_tokens()` in `calendar_cod.py` reads `CALENDAR_TOKENS` from `state.py`, filters
to attendees present in the email, and refreshes each token at runtime using
`GOOGLE_CALENDAR_CLIENT_ID` and `GOOGLE_CALENDAR_CLIENT_SECRET`.

> **Old behaviour (removed):** Previously, tokens were read from `tokens_*.json` files in the CWD.
> Those files are no longer used.

---

## Gmail History Tracking

`gmail_history.py` stores the last processed Gmail `historyId` in AWS SSM Parameter Store
(`/agent/gmail_history_id` by default). This prevents re-processing emails on every Pub/Sub push.

`gmail_watcher.py` is a one-time setup script that registers the Gmail → Pub/Sub watch and seeds
the initial `historyId` into SSM. **Re-run every 7 days** (Gmail watches expire after 7 days).

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
| `CALENDAR_TOKENS_JSON` | `state.py` | JSON dict of participant email → refresh_token |
| `SQS_QUEUE_URL` | `lambda_ingest.py` | SQS FIFO queue URL for enqueuing inbound events |
| `SSM_HISTORY_ID_PARAM` | `gmail_history.py` | SSM parameter name for Gmail historyId (default: `/agent/gmail_history_id`) |
| `PUBSUB_TOPIC` | `gmail_watcher.py` | Gmail Pub/Sub topic name for watch registration |
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

# Local orchestrator test — simulates a Slack message event
python run_local.py

# Local Pub/Sub test — simulates a Gmail Pub/Sub push event
python runlocal_pubsub.py

# Test ingest Lambda with a Pub/Sub payload
python test_ingest_pubsub.py

# Test ingest Lambda with a Slack payload
python test_ingest_slack.py
```

---

## Deployment

There are **two** Lambdas to deploy:

```bash
# Build and push Docker image to ECR, then update both Lambdas
bash deploy.sh
```

| Lambda | Handler | Trigger |
|---|---|---|
| Ingest | `lambda_ingest.handler` | Lambda Function URL (no auth) |
| Worker | `calendar_cod.sqs_handler` | SQS FIFO queue |

Both Lambdas share the same Docker image (Python 3.12). Timeout: 60 seconds each.

The Lambda Function URL (ingest only) is used for:
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

Slack retries failed requests with the `x-slack-retry-num` header. Retries are dropped at **two levels**:

1. **Ingest Lambda** (`lambda_ingest.py`) — drops retries before enqueuing to SQS (never reaches the worker)
2. **Worker** (`calendar_cod.py` → `parse_input`) — secondary check sets `intent=none` if a retry somehow arrives

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
