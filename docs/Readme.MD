# Workplace Automation — LangGraph Agent

A serverless, event-driven automation system that runs on **AWS Lambda** and uses a self-hosted **Qwen 2.5 14B** model (via vLLM on EC2) to power two workflows:

1. **Slack → Jira** — listens to Slack messages, extracts task details with an LLM, proposes a Jira ticket for human approval, then creates it.
2. **Email → Google Calendar** — listens to incoming Gmail messages via Pub/Sub, classifies them for meeting intent, proposes a calendar event for human approval, then creates it.

Both flows are human-in-the-loop: no ticket or calendar event is created without a button click in Slack.

---

## Table of contents

- [Architecture](#architecture)
- [File structure](#file-structure)
- [Configuration](#configuration)
- [EC2 setup](#ec2-setup)
- [vLLM inference engine](#vllm-inference-engine)
- [Deployment](#deployment)
- [Slack and API Gateway integration](#slack-and-api-gateway-integration)
- [Verification and health checks](#verification-and-health-checks)
- [Gmail Pub/Sub setup](#gmail-pubsub-setup)
- [Event routing](#event-routing)
- [Stateless button callbacks](#stateless-button-callbacks)
- [Slack retry handling](#slack-retry-handling)
- [Local testing](#local-testing)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)

---

## Architecture

```
Lambda handler
    │
    ├── parse_input        — detects event type (Slack message / interactivity /
    │                        Gmail Pub/Sub / direct test payload)
    │
    └── router_agent       — Qwen LLM classifies intent: "slack" | "email" | "none"
             │
             ├── [intent=slack]   → Slack subgraph  (Jira flow)
             └── [intent=email]   → Calendar subgraph  (email → GCal flow)
```

The graph is built with [LangGraph](https://github.com/langchain-ai/langgraph) and compiled with a `MemorySaver` checkpointer. All state is carried through a single `OrchestratorState` Pydantic model — no database, no shared mutable state.

### Slack subgraph — Jira flow

```
slack_extract_ticket   →   slack_post_preview
                                   │
                        ┌──────────┴──────────┐
                     create                cancel
                        │                    │
              slack_resolve_assignee    slack_post_cancel
                        │
              slack_create_jira
                        │
              slack_post_result
```

| Node | What it does |
|---|---|
| `slack_extract_ticket` | Qwen LLM extracts `task_summary` and `assignee` from the Slack message text |
| `slack_post_preview` | Posts a proposed ticket card to Slack with ✅ Create / ❌ Cancel buttons |
| `slack_resolve_assignee` | Maps the raw assignee string to a Jira account ID via `TEAM_MAP` |
| `slack_create_jira` | Calls the Jira REST API (`POST /rest/api/3/issue`) |
| `slack_post_result` | Updates the Slack card with the created ticket link or error |
| `slack_post_cancel` | Updates the Slack card with a cancellation notice |

### Calendar subgraph — email flow

```
email_fetch_and_parse  →  email_classify
                                │
                     ┌──────────┴──────────┐
                  meeting             no meeting
                     │
          email_post_slack_preview
                     │  (second Lambda invocation — button click)
                     │
            ┌────────┴────────┐
         create            cancel
            │                 │
  email_create_calendar   email_post_cancel
```

| Node | What it does |
|---|---|
| `email_fetch_and_parse` | For Pub/Sub events: fetches the email from Gmail using the history ID. For direct invocations: uses `email_data` already in state |
| `email_classify` | Qwen LLM returns structured JSON: `is_meeting`, `title`, `start_time`, `end_time`, `location`, `attendees`, `time_confidence` |
| `email_post_slack_preview` | Posts a meeting proposal card to Slack; the full meeting dict is embedded in the button value (no DynamoDB needed) |
| `email_create_calendar` | Creates the Google Calendar event and sends invites via the Calendar API; updates the Slack card with the event link |
| `email_post_cancel` | Updates the Slack card with a cancellation notice |

---

## File structure

```
.
├── orchestrator.py     # Top-level graph, parse_input, router_agent, Lambda handler
├── slack_agent.py      # Slack + Jira subgraph
├── calendar_agent.py   # Email + Google Calendar subgraph
└── state.py            # OrchestratorState, config, shared LLM factory
```

---

## Configuration

All configuration is injected via environment variables. No config files are committed.

| Variable | Required | Description |
|---|---|---|
| `EC2_IP` | Yes | Public IP of the EC2 instance running vLLM (port 8000) |
| `SLACK_BOT_TOKEN` | Yes | Slack bot OAuth token (`xoxb-...`) |
| `SLACK_NOTIFY_CHANNEL` | Yes | Channel ID where meeting previews are posted |
| `JIRA_BASE_URL` | Yes | Your Jira instance base URL, e.g. `https://yourorg.atlassian.net` |
| `JIRA_EMAIL` | Yes | Email address used for Jira API auth |
| `JIRA_API_TOKEN` | Yes | Jira API token |
| `JIRA_PROJECT_KEY` | No | Jira project key, default `KAN` |
| `JIRA_ISSUE_TYPE` | No | Issue type, default `Task` |
| `GOOGLE_TOKEN_JSON` | Yes | Serialised Google OAuth token JSON (`token`, `refresh_token`, `client_id`, `client_secret`, `token_uri`, `scopes`) |
| `GROUP_EMAILS_JSON` | No | JSON array of allowed sender emails. If set, only emails from these addresses trigger the calendar flow |
| `TEAM_MAP_JSON` | No | JSON object mapping Slack display names / user IDs → Jira account IDs, e.g. `{"SLACK_ID": "JIRA_ACCOUNT_ID"}` |

Set all of the above in the **Lambda Console → Configuration → Environment variables** before first deployment.

---

## EC2 setup

### 1. Connect to the instance

```bash
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
```

### 2. Prepare the environment

```bash
# Activate the virtual environment where vLLM is installed
source llm_env/bin/activate

# Navigate to the adapter directory
cd ~/adapters
```

### 3. Check GPU memory status

Run this before starting the model to confirm VRAM is free:

```bash
nvidia-smi
```

### 4. Clear zombie GPU processes

If the model fails to start with a `ValueError` about free memory, a previous crashed session has left ghost processes occupying VRAM. Force-clear the hardware:

```bash
sudo fuser -v /dev/nvidia* -k
```

Re-run `nvidia-smi` afterwards to confirm memory is released before proceeding.

---

## vLLM inference engine

This command starts the inference brain. It loads the base Qwen model and attaches the three custom LoRA adapters. Keep this terminal open — closing it stops the model.

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --quantization awq \
    --dtype auto \
    --enable-lora \
    --max-loras 3 \
    --max-lora-rank 32 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 4096 \
    --lora-modules \
        slack=~/adapters/slack2jira \
        email=~/adapters/email-calender \
        meeting=~/adapters/transcript_summarizer \
    --host 0.0.0.0 \
    --port 8000
```

The `--lora-modules` names (`slack`, `email`, `meeting`) must match the `model=` values used in `state.py`'s `_llm()` factory. The `slack` adapter handles both the router and the Jira flow; `email` handles the calendar classifier.

The LLM factory in `state.py`:

```python
def _llm(structured_output=None) -> ChatOpenAI:
    base = ChatOpenAI(
        model="slack",
        openai_api_base=f"http://{EC2_IP}:8000/v1",
        openai_api_key="none",
        timeout=45,
        max_retries=0,
    )
    return base.with_structured_output(structured_output) if structured_output else base
```

Structured outputs (Pydantic models passed as `structured_output`) are used for `RouterDecision`, `JiraTicket`, and the email classification JSON.

---

## Deployment

The Lambda is packaged as a Docker container and deployed via Amazon ECR. Heavy dependencies (`slack_sdk`, `langchain`, `langgraph`, Google API clients) are bundled in the image.

### 1. Build the image

```bash
docker build -t slack-jira-agent .
```

### 2. Authenticate Docker to ECR

```bash
aws ecr get-login-password --region us-east-1 | docker login \
    --username AWS \
    --password-stdin [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com
```

### 3. Tag and push

```bash
docker tag slack-jira-agent:latest \
    [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/slack-jira-repo:latest

docker push \
    [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/slack-jira-repo:latest
```

### 4. Update the Lambda function

In the AWS Lambda Console go to **Image → Deploy New Image** and select the URI you just pushed.

Alternatively via CLI:

```bash
aws lambda update-function-code \
    --function-name slack-jira-agent \
    --image-uri [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/slack-jira-repo:latest
```

> Set the Lambda **timeout to 60 seconds** — the default 3-second timeout is too short for LLM calls over the network.

---

## Slack and API Gateway integration

The Lambda needs a public URL so Slack can reach it.

### Lambda Function URL (recommended for testing)

In the Lambda Console go to **Configuration → Function URL** and enable it with auth type **NONE**.

### Slack app wiring

1. **Event Subscriptions** — paste the Lambda Function URL into the *Request URL* field in your Slack App settings. Slack will send a URL verification challenge; `parse_input` handles this automatically and returns the challenge value.
2. **Interactivity & Shortcuts** — paste the same URL into the *Interactivity* section. This is required so the ✅ Create Ticket and ✅ Create Meeting buttons post back to Lambda.

Both fields accept the same URL.

---

## Verification and health checks

### Test the vLLM API from inside EC2

Open a second terminal on the EC2 instance:

```bash
curl http://localhost:8000/v1/models
```

### Test the vLLM API from your laptop

```bash
curl http://[YOUR-EC2-IP]:8000/v1/models
```

If this times out, check that **port 8000 is open** in the EC2 Security Group (inbound TCP rule for your IP or the Lambda's NAT gateway IP).

### Check Lambda logs

If Slack is not responding, open **AWS CloudWatch → Log groups → /aws/lambda/slack-jira-agent** and inspect the latest log stream.

---

## Gmail Pub/Sub setup

To trigger the calendar agent automatically on new emails:

1. Create a Pub/Sub topic in Google Cloud.
2. Grant the Gmail service account (`gmail-api-push@system.gserviceaccount.com`) the **Pub/Sub Publisher** role on the topic.
3. Call `gmail.users().watch()` with your topic name and a label filter (e.g. `INBOX`).
4. Configure the Pub/Sub push subscription to POST to your Lambda Function URL.

The Lambda URL must be publicly accessible. Pub/Sub pushes a JSON body containing `message.data` (base64-encoded), which `parse_input` decodes to extract the Gmail history ID used to fetch the triggering email.

---

## Event routing

`parse_input` is the single entry point for all event types. It sets `intent` directly for unambiguous events, and leaves it as `"unknown"` for Slack message events that the LLM router needs to classify.

| Incoming event | Detected by | `intent` set to |
|---|---|---|
| Slack retry (`x-slack-retry-num > 0`) | Header check | `none` |
| Slack URL verification | `body.type == "url_verification"` | `slack` |
| Slack button press (`payload=...`) | URL-encoded body prefix | `slack` or `email` depending on `action_id` |
| Gmail Pub/Sub push | `body.message.data` with `emailAddress` field | `email` |
| Slack message event | `body.event` present | `unknown` → routed by LLM |
| Direct test payload | `from_email` or `subject` in body | `email` |

Button `action_id` values and their routing:

| `action_id` | Routed to |
|---|---|
| `create_jira` | `slack_resolve_assignee` |
| `cancel_jira` | `slack_post_cancel` |
| `create_meeting` | `email_create_calendar` |
| `cancel_meeting` | `email_post_cancel` |

---

## Stateless button callbacks

Meeting and ticket data is serialised into Slack button `value` fields as JSON — no external state store is required between the preview post and the button click. On the second Lambda invocation, `parse_input` deserialises the button value and reconstructs the `pending_meeting` or ticket dict directly in `OrchestratorState`.

---

## Slack retry handling

Slack retries requests that do not receive a `200` response within 3 seconds, adding an `x-slack-retry-num` header on subsequent attempts. `parse_input` detects this header and immediately returns `intent = "none"`, preventing duplicate ticket or event creation.

---

## Local testing

The `handler` function accepts a plain dict, so it can be invoked directly without a real Lambda runtime.

**Test the Slack → Jira flow:**

```python
from orchestrator import handler

result = handler({
    "headers": {},
    "body": '{"event": {"type": "message", "text": "Can someone create a task for Alice to update the onboarding docs?", "channel": "C123456", "ts": "1234567890.000100"}}'
}, None)
```

**Test the email → calendar flow:**

```python
result = handler({
    "headers": {},
    "body": '{"from_email": "alice@example.com", "subject": "Team sync Thursday 3pm", "body": "Hi, let\'s meet Thursday at 3pm for an hour to discuss Q3 plans."}'
}, None)
```

**Simulate a button press:**

```python
import json, urllib.parse

payload = {
    "actions": [{"action_id": "create_jira", "value": json.dumps({"s": "Update onboarding docs", "a": "alice"})}],
    "channel": {"id": "C123456"},
    "container": {"message_ts": "1234567890.000200"}
}
result = handler({
    "headers": {},
    "body": "payload=" + urllib.parse.quote_plus(json.dumps(payload))
}, None)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| vLLM fails to start — `ValueError: not enough free memory` | Ghost process holding VRAM from a crashed session | Run `sudo fuser -v /dev/nvidia* -k`, confirm with `nvidia-smi`, then restart |
| Lambda returns 404 | Wrong API path | Ensure the endpoint is `/v1/chat/completions`, not `/v1/completions` |
| Lambda times out | Default 3s timeout too short for LLM round-trip | Increase Lambda timeout to **60 seconds** in Configuration |
| Lambda cannot reach EC2 | Port 8000 blocked | Add an inbound TCP rule for port 8000 in the EC2 Security Group |
| `curl http://[EC2-IP]:8000` times out from laptop | Security Group blocks external access | Open port 8000 to your IP (or Lambda NAT IP) in EC2 inbound rules |
| Slack sends duplicate events | Slack retry being processed as a new event | Confirm the `x-slack-retry-num` header check is active in `parse_input` |
| Bot creates ticket or event twice | Race in `MemorySaver` or missing retry guard | Check CloudWatch for a second invocation without the retry header |

---

## Dependencies

```
langchain-openai
langchain-core
langgraph
pydantic
slack-sdk
requests
google-auth
google-auth-httplib2
google-api-python-client
python-dateutil
```