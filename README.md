# Slack-Jira integration

## Overview
The Multi-Agent Autonomous Workforce Assistant is a project designed to streamline task management by integrating Slack messages with Jira issue tracking. This system utilizes natural language processing (NLP) techniques to extract actionable items from Slack conversations and create or update Jira tickets accordingly.

## Project Structure
- **DataPreprocessing.ipynb**: This notebook handles the preprocessing of Slack message data, including parsing XML files and cleaning the text for further analysis.
- **DataLabeling.ipynb**: This notebook uses the Gemini model to label Slack messages with task-related information, such as task summaries, assignees, and issue creation dates.
- **FinetuneLLM.ipynb**: This notebook is responsible for fine-tuning a language model (QLoRA) to extract task-related information from Slack messages and format it for Jira.
- **Requirements**: The project requires several Python packages, including `pandas`, `transformers`, `sentence-transformers`, `requests`, and others specified in the notebooks.

## Installation
To set up the project, ensure you have Python installed, then install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preprocessing**: Run the `DataPreprocessing.ipynb` notebook to parse Slack message XML files and save the cleaned messages to a CSV file.
2. **Data Labeling**: Use the `DataLabeling.ipynb` notebook to label Slack messages with task-related information using the Gemini model. The labeled data will be saved for further processing.
3. **Fine-tuning the Model**: Execute the `FinetuneLLM.ipynb` notebook to fine-tune the language model on the labeled data.
4. **Creating/Updating Jira Issues**: Use the provided functions to create or update Jira issues based on Slack messages. Ensure that your Jira credentials and project details are set in the environment variables.

## Environment Variables
Set the following environment variables for Jira integration:
- `JIRA_BASE_URL`: Your Jira instance URL (e.g., `https://your-domain.atlassian.net`)
- `JIRA_EMAIL`: Your Jira account email
- `JIRA_API_TOKEN`: Your Jira API token
- `JIRA_PROJECT_KEY`: The key of the Jira project where issues will be created
- `JIRA_ISSUE_TYPE`: The type of issue to create (e.g., Task, Bug)

## Example
To create a new Jira issue from a Slack message, use the following code snippet:

```python
slack_msg = {
    "timestamp": "2025-11-02T08:14:27.439600",
    "text": "<@User> Can you complete the task by tomorrow?"
}

new_key = create_issue(slack_msg)
print("Created:", new_key)
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


## License

No license is currently specified in this repository.
