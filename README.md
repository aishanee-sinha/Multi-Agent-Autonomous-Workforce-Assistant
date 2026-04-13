# Multi-Agent Autonomous Workforce Assistant (Chain-of-Debate)

A production-ready, multi-agent workplace automation system featuring **Human-in-the-loop** (HITL) task execution and **RLHF telemetry** powered by a self-hosted Qwen-2.5 14B model.

- **Slack → Jira**: Extract tasks from conversations, propose a ticket for approval, and create it on Jira.
- **Email → Google Calendar**: Detect meeting intent from Gmail, propose slots in Slack, and book the event.
- **Meeting Summarizer**: Automatically process meeting transcripts into summaries and action items.

---

## 🏛️ Deployment Architecture

This project is built for **scale and statelessness** using a hybrid cloud architecture:

1.  **AWS Lambda (The Orchestrator)**: Runs the LangGraph agent logic using a containerized Python 3.11 environment.
2.  **AWS EC2 (The Brain)**: A GPU-backed instance running:
    - **vLLM Engine (Port 8000)**: Serves the Qwen-2.5 model with custom LoRA adapters.
    - **ChromaDB (Port 8001)**: A vector database for RLHF telemetry and RAG-based prompt injection.
3.  **Upstash Redis (The Memory)**: Manages short-term session state to bypass Slack's 3KB interactive payload limit.

---

## 📁 Repository Structure

```text
.
├── src/                    # Core Production Code
│   ├── orchestrator.py     # Lambda entrypoint & Graph routing
│   ├── state.py            # Global Graph State & Config
│   ├── slack_agent.py      # Jira Ticket Subgraph
│   ├── calendar_agent.py   # Google Calendar Subgraph
│   ├── meeting_agent.py    # Meeting Summarizer Subgraph
│   ├── redis_store.py      # Session management & ChromaDB hooks
│   └── rag_retriever.py    # RAG logic for RLHF injection
├── rlhf/                   # Reinforcement Learning Pipeline
│   ├── setup_chromadb.sh   # Spin up Vector DB on EC2
│   ├── check_chroma_data.py # Utility to verify logging
│   ├── train_dpo.py        # Offline Direct Preference Optimization
│   └── build_preference_dataset.py
├── DataProcessingModelTraining/ # R&D Notebooks
│   ├── EmailCalendar/      # Notebooks for GCal model training
│   ├── MeetingSummarizer/  # Notebooks for Summarization model
│   └── Slack_Jira/         # Notebooks for Jira extraction model
├── docs/                   # Detailed Runbooks & Architecture
├── Dockerfile              # Lambda container definition
└── requirements.txt        # Production dependencies
```

---

## 🚀 Getting Started

### 1. EC2 Backend Setup (The Brain)
SSH into your GPU instance and start the two required services:
```bash
# Start ChromaDB (RLHF telemetry)
bash rlhf/setup_chromadb.sh --bg

# Start vLLM inference engine
# (See docs/Runbook.md for the full vllm start command)
```
*Note: Ensure Port 8000 (vLLM) and Port 8001 (ChromaDB) are open in your EC2 Security Groups.*

### 2. AWS Lambda Deployment (The Orchestration)
Build and push the Docker image to your Amazon ECR repository:
```bash
docker build -t slack-jira-agent .
# (Push to ECR and click "Deploy new image" in the AWS Lambda console)
```

### 3. Environment Variables
Inject these variables into your Lambda function:
| Variable | Required | Purpose |
|---|---|---|
| `EC2_IP` | Yes | Public IP of your GPU server |
| `REDIS_URL` | Yes | Your Upstash Redis connection string (`rediss://...`) |
| `SLACK_BOT_TOKEN` | Yes | Slack App Bot OAuth Token |
| `GOOGLE_TOKEN_JSON`| Yes | Serialized Google OAuth JSON for Calendar/Gmail |
| `JIRA_API_TOKEN` | Yes | Jira REST API token |

---

## 📊 RLHF Pipeline
The system natively records every user decision ("Approve" vs "Cancel") into ChromaDB. 

- **Monitor Telemetry**: Run `python3 rlhf/check_chroma_data.py` on your EC2.
- **Train Adapters**: Once enough data is collected, use the scripts in `rlhf/` to build a DPO dataset and fine-tune your model to improve its future suggestions.

---

## 📚 Detailed Documentation
- [Architecture Deep-Dive](docs/Readme.MD)
- [Operational Runbook (Ops)](docs/Runbook.md)
- [RLHF Next Steps](NEXT_STEPS.md)
zer/README.md` — meeting summarizer tooling
- `DataProcessingModelTraining/MeetingSummarizer/README.md` — summarizer training/eval notes
- `DataProcessingModelTraining/Slack_Jira/readme.md` — Slack/Jira labeling + training


## License

No license is currently specified in this repository.
