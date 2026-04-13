# CLAUDE.md — Agent Project (Modular)

## Project Overview

Multi-agent LangGraph orchestrator deployed as a pair of AWS Lambda functions. Receives events from
Slack (messages + button clicks) and Gmail (Pub/Sub push) and routes them to one of **three**
human-in-the-loop flows:

- **Slack flow** — extracts a task from a Slack message, proposes a Jira ticket, creates it on approval
- **Email flow** — detects meeting intent in an email, finds the best free slot via Chain-of-Debate, proposes a Google Calendar event, creates it on approval
- **Meeting flow** — downloads a meeting transcript from Google Drive, summarises it + triages action items via CoD, stores artifacts in S3, sends email via SES on approval

All flows post a Slack preview card with Approve/Cancel buttons before taking any action.

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
                                   ├─ calendar_subgraph fetch email → classify → CoD slot → preview → [approve] → create Calendar event
                                   └─ meeting_subgraph  fetch transcript → preprocess → summarise → CoD triage → artifacts → S3 → preview → [approve] → SES email
```

### Key Files

| File | Role |
|---|---|
| `state.py` | `OrchestratorState` (shared Pydantic model), `_llm()` factory, all env var constants |
| `lambda_ingest.py` | Ingest Lambda — URL verify, retry drop, SQS enqueue |
| `calendar_cod.py` | Worker Lambda — `parse_input`, `router_agent`, full graph assembly, CoD slot selection |
| `slack_agent.py` | Slack → Jira subgraph nodes |
| `calendar_agent.py` | Email → Calendar subgraph nodes + `email_classify` prompt (imported by `calendar_cod.py`) |
| `meeting_agent.py` | Meeting transcript → summarise → CoD triage → S3 → SES subgraph |
| `redis_store.py` | Redis session storage — replaces full JSON in button values with a UUID session_id |
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
- **LoRA adapters:** `slack` (Jira extraction), `email` (meeting classification), `meeting` (transcript summarisation + triage)
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

## Meeting Flow (Transcript → Summary → SES)

`meeting_agent.py` is the third subgraph. Pipeline on new transcript trigger:

1. **`meeting_fetch_transcript`** — Download from Google Drive; S3 idempotency lock prevents reprocessing
2. **`meeting_preprocess`** — Clean/normalise transcript text
3. **`meeting_summarize`** — Chunked vLLM inference (`CHUNK_SIZE=10000, CHUNK_OVERLAP=1000`) + merge-pass abstract
4. **`meeting_triage_cod`** — CoD classifies each action item: Priority (Critical/High/Medium/Low) + risk + deadline assessment
5. **`meeting_generate_artifacts`** — ICS calendar invite (UTC-corrected) + CSV of action items
6. **`meeting_store_s3`** — All artifacts uploaded to S3 (`S3_BUCKET` / `S3_PREFIX`), sets `meeting_s3_key`
7. **`meeting_post_slack`** — Block Kit card with colour-coded triage (🔴🟠🟡🟢) + Confirm/Cancel buttons

On approval:
- **`meeting_send_email`** — Fetches artifacts from S3, sends via AWS SES
- **`meeting_post_cancel`** — Updates Slack card to dismissed

### meeting CoD schemas

Same 3-role Proposer → Challenger → Judge pattern as `slot_cod`:
- `ActionPriorityProposal` → `PriorityChallenge` → `TriageVerdict`
- Each action item is debated independently; Slack card shows the triage results before human approves

---

## Redis Session Store

`redis_store.py` replaces embedding full JSON payloads in Slack button values (2000-char limit).

| Function | Description |
|---|---|
| `save_session(data)` | Stores `data` as JSON in Redis under a UUID, returns the `session_id` |
| `load_session(session_id)` | Fetches and deserialises; returns `None` if expired |
| `record_feedback(session_id, outcome, metadata)` | Merges human decision (`accepted`/`rejected`/`failed`) into session; extends TTL to 7 days for RLHF retention |

- Session TTL: `SESSION_TTL_SECONDS` (default 3600 s / 1 hour)
- Feedback TTL: 7 days (`FEEDBACK_TTL_SECONDS`)
- Connection: `REDIS_URL` env var (default `redis://localhost:6379`), lazy singleton `_get_client()`

**Button value pattern with Redis:**
```
# Posting preview:
session_id = save_session(pending_dict)
button["value"] = session_id          # only UUID in button, not full JSON

# On button click:
data = load_session(session_id)       # retrieve full payload from Redis
record_feedback(session_id, "accepted", {"calendar_link": "..."})
```

---

## Timezone Handling

- All times use the **local system timezone**, derived at runtime via `datetime.now().astimezone()`
- `LOCAL_TZ` in `calendar_cod.py` — used for all datetime construction and event time conversion
- `_tz_offset` and `_tz_name` in `_build_email_system_prompt()` — injected into the LLM prompt so the model outputs datetimes with the correct local UTC offset
- Slot labels use `cursor.strftime("%Z")` so the timezone abbreviation is dynamic
- `meeting_agent.py` has a `TZ_OFFSETS` dict for ICS UTC correction (handles PDT, EST, IST, etc.)

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
| `REDIS_URL` | `state.py` | Redis connection URL (default: `redis://localhost:6379`) |
| `SESSION_TTL_SECONDS` | `state.py` | Redis session TTL in seconds (default: `3600`) |
| `S3_BUCKET` | `state.py` | S3 bucket for transcript artifacts (default: `qwen-lora-weights`) |
| `S3_TRANSCRIPT_PREFIX` | `state.py` | S3 key prefix for transcript artifacts (default: `transcript_summarizer`) |
| `VLLM_MODEL_NAME` | `state.py` | vLLM LoRA adapter name for meeting summarisation (default: `meeting`) |
| `SES_FROM_EMAIL` | `state.py` | SES sender address for meeting summary emails |
| `PARTICIPANT_EMAILS` | `state.py` | Comma-separated list of default meeting participant emails |
| `CALENDAR_INCLUDE_SENDER` | `calendar_cod.py` | Include email sender in calendar fetch (default: `true`) |

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

Test files in `tests/`:

| Test file | What it covers |
|---|---|
| `test_calendar_agent_local.py` | Calendar agent nodes end-to-end |
| `test_cod_email_flow.py` | Full CoD email → calendar flow |
| `test_cod_scheduler.py` | `slot_cod` slot selection logic |
| `test_email_classify.py` | `email_classify` LLM node |
| `test_orchestrator_local.py` | Full orchestrator graph |
| `test_pubsub.py` | Gmail Pub/Sub ingest path |
| `test_redis.py` | `redis_store` save/load/feedback |
| `test_slack_post_preview.py` | Slack preview card posting |
| `test_sqs_read.py` | SQS record unwrapping |

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

## Session / State Design

Button state is stored in **Redis** via `redis_store.py`. Only a UUID `session_id` is placed in
the Slack button `value` field. On button click, `load_session(session_id)` retrieves the full
payload. After the action completes, `record_feedback()` appends the human decision and extends
the TTL to 7 days for RLHF retention.

> **Previous behaviour (removed):** Full JSON payload was embedded directly in the button value
> field, which hit Slack's 2000-character limit for long emails.

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
