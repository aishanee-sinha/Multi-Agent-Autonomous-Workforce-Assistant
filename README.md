# Multi-Agent Autonomous Workforce Assistant

A multi-agent workplace automation system with two production workflows and a full R\&D pipeline:

- **Slack → Jira (human-in-the-loop)**: propose a Jira ticket from a Slack message, create it only after approval.
- **Email → Google Calendar (human-in-the-loop)**: detect meeting intent from new emails, propose a calendar event in Slack, create it only after approval.
- **Model training notebooks**: data labeling, preprocessing, fine-tuning (LoRA), and evaluation for meeting extraction, meeting summarization, and Slack/Jira extraction.

## What’s in this repo

This repository is split into:

1. **AWSOrchestration** — the event-driven orchestration layer designed to run on **AWS Lambda**, backed by a self-hosted **vLLM OpenAI-compatible endpoint** (typically on GPU EC2).
2. **DataProcessingModelTraining** — Jupyter notebooks for labeling + training + evaluation of task-specific models/adapters.

## Architecture (production)

At a high level, the Lambda handler parses the incoming event (Slack event, Slack button click, Gmail Pub/Sub push, or a direct test payload), routes intent, then runs the relevant subgraph.

```
Lambda handler
  ├─ parse_input   → detects event type (Slack / Gmail PubSub / interactivity)
  └─ router_agent  → LLM routes: "slack" | "email" | "none"
         ├─ Slack subgraph    → Jira ticket proposal + approval + creation
         └─ Calendar subgraph → meeting extraction + approval + calendar creation
```

The orchestration graph is implemented with **LangGraph** and a single `OrchestratorState` model (no database required for button callbacks — the pending payload is embedded in Slack button values).

## Repository structure

```text
.
├── AWSOrchestration/
│   ├── orchestrator.py          # Lambda handler + routing + graph wiring
│   ├── state.py                 # Shared config + OrchestratorState + LLM factory
│   ├── slack_agent.py           # Slack/Jira subgraph
│   ├── calendar_agent.py        # Email/Calendar subgraph
│   ├── gmail_watcher.py         # Helper: register Gmail → Pub/Sub watch
│   ├── get_google_token.py      # Helper: generate OAuth token JSON
│   ├── Dockerfile               # Lambda container build
│   ├── requirements.txt         # Lambda container Python deps
│   ├── Readme.MD                # Deep-dive architecture + deployment
│   ├── Runbook.md               # Operational runbook (EC2 + Docker + Lambda)
│   └── Meeting_Summarizer/
│       ├── ms_agent_call.py
│       ├── google_apps_script.js
│       └── README.md
└── DataProcessingModelTraining/
     ├── EmailCalendar/
     │   ├── DataLabelling_Email.ipynb
     │   ├── DataPreprocessing_Email.ipynb
     │   ├── Finetuning_Email.ipynb
     │   └── Evaluation_Email.ipynb
     ├── MeetingSummarizer/
     │   ├── Phase4_Meeting_Summarizer_14B.ipynb
     │   ├── Phase4_Evaluation_A100_Colab.ipynb
     │   └── README.md
     └── Slack_Jira/
          ├── DataLabeling.ipynb
          ├── DataPreprocessing.ipynb
          ├── SlackJiraModelFT&Eval.ipynb
          ├── Slack_jira_GEval.ipynb
          └── readme.md
```

## Getting started

### Option A — Orchestration (AWS Lambda + EC2 vLLM)

This path is for running the production system.

1. **Start the vLLM server** (typically on a GPU EC2 instance) with your base model + LoRA adapters.
    - The canonical command and adapter naming conventions are documented in `AWSOrchestration/Readme.MD` and `AWSOrchestration/Runbook.md`.

2. **Build and deploy the Lambda container image**:
    ```bash
    cd AWSOrchestration
    docker build -t slack-jira-agent .
    ```
    Then push to ECR and update the Lambda image (see `AWSOrchestration/Readme.MD`).

3. **Configure Slack**:
    - Set the Lambda function URL / API Gateway endpoint as Slack’s Request URL.
    - Enable **Event Subscriptions** and **Interactivity**.

4. **Configure Gmail → Pub/Sub → Lambda**:
    - Create a Pub/Sub topic, connect it to Gmail watch, and route pushes to your Lambda.
    - You can register the Gmail watch with:
      ```bash
      cd AWSOrchestration
      python gmail_watcher.py
      ```
      (requires `GOOGLE_TOKEN_JSON` and `PUBSUB_TOPIC` env vars).

### Option B — Notebooks (data labeling → fine-tuning → evaluation)

This path is for reproducing training/evaluation runs (commonly in Google Colab).

**Email → Calendar meeting extraction pipeline** (recommended order):
1. `DataProcessingModelTraining/EmailCalendar/DataLabelling_Email.ipynb`
2. `DataProcessingModelTraining/EmailCalendar/DataPreprocessing_Email.ipynb`
3. `DataProcessingModelTraining/EmailCalendar/Finetuning_Email.ipynb`
4. `DataProcessingModelTraining/EmailCalendar/Evaluation_Email.ipynb`

Other tracks:
- Meeting summarization: `DataProcessingModelTraining/MeetingSummarizer/README.md`
- Slack → Jira extraction: `DataProcessingModelTraining/Slack_Jira/readme.md`

## Configuration

### Orchestration environment variables (Lambda)

These are read in `AWSOrchestration/state.py` and used by `orchestrator.py`, `slack_agent.py`, and `calendar_agent.py`.

| Variable | Required | Purpose |
|---|---:|---|
| `EC2_IP` | Yes | Public IP/host of the vLLM server (expects OpenAI-compatible `/v1` endpoints) |
| `SLACK_BOT_TOKEN` | Yes | Slack bot token (`xoxb-...`) |
| `SLACK_NOTIFY_CHANNEL` | Yes | Channel ID where meeting proposals are posted |
| `JIRA_BASE_URL` | Yes | Jira base URL, e.g. `https://yourorg.atlassian.net` |
| `JIRA_EMAIL` | Yes | Jira account email used for API auth |
| `JIRA_API_TOKEN` | Yes | Jira API token |
| `JIRA_PROJECT_KEY` | No | Defaults to `KAN` |
| `JIRA_ISSUE_TYPE` | No | Defaults to `Task` |
| `GOOGLE_TOKEN_JSON` | Yes | Serialized Google OAuth token JSON (used for Gmail fetch + Calendar) |
| `GROUP_EMAILS_JSON` | No | JSON array allowlist of sender emails for calendar flow |
| `TEAM_MAP_JSON` | No | JSON map: Slack name/id → Jira account id |

### Helper-script environment variables (local / ops)

| Variable | Required | Used by |
|---|---:|---|
| `PUBSUB_TOPIC` | Yes (for watch) | `AWSOrchestration/gmail_watcher.py` |

### Notebook secrets (training)

The notebooks generally assume execution in Colab and may use:

- `GEMINI_API_KEY` (for automated labeling)
- `HF_TOKEN` (for pulling gated models/checkpoints)

## Evaluation notes

Evaluation for meeting extraction includes:

- **JSON validity** (parseable structured output)
- **Meeting detection** confusion matrix (precision/recall/F1)
- **Field-level extraction** accuracy (title, attendees, start/end time, location, time_confidence)

See `DataProcessingModelTraining/EmailCalendar/Evaluation_Email.ipynb` for the exact metrics code and reporting.

## Docs

- `AWSOrchestration/Readme.MD` — architecture + deployment details
- `AWSOrchestration/Runbook.md` — operational commands and troubleshooting
- `AWSOrchestration/Meeting_Summarizer/README.md` — meeting summarizer tooling
- `DataProcessingModelTraining/MeetingSummarizer/README.md` — summarizer training/eval notes
- `DataProcessingModelTraining/Slack_Jira/readme.md` — Slack/Jira labeling + training

## Security

- Do not commit secrets (Slack tokens, Jira tokens, OAuth tokens).
- Prefer Lambda environment variables and Colab Secrets.
- Treat `GOOGLE_TOKEN_JSON` as a secret (it contains refresh credentials).

## License

No license is currently specified in this repository.
