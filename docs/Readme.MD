# Multi-Agent Architecture & Implementation

This document provides a deep-dive into the technical architecture, graph logic, and deployment of the Multi-Agent Workforce Assistant.

---

## 🏛️ System Architecture

The system follows a **Modular Agentic** pattern built on **LangGraph**. It consists of a top-level Router and three specialized subgraphs.

### Top-Level Routing
```text
Lambda Entrypoint (src/orchestrator.py)
    │
    ├── parse_input        — Normalizes events (Slack, Gmail, Buttons)
    │
    └── router_agent       — Uses LLM to classify intent: "slack" | "email" | "meeting"
             │
             ├── [intent=slack]   → Slack Subgraph (Jira flow)
             ├── [intent=email]   → Calendar Subgraph (GCal flow)
             └── [intent=meeting] → Meeting Subgraph (Summarizer)
```

### Statelessness & Scaling (Redis)
Unlike simple bots, this system manages complex multi-step workflows. Because Slack limits button `value` fields to **3,000 characters**, we cannot store entire meeting proposals in the button itself.

Instead, we use **Upstash Redis**:
1.  **Selection**: The AI proposes a slot and generates a massive JSON payload.
2.  **Storage**: `src/redis_store.py` saves the payload to Redis and returns a 36-character UUID.
3.  **Handoff**: Only the UUID is placed in the Slack button.
4.  **Resumption**: When a human clicks the button, the Lambda uses the UUID to "hydrate" the session from Redis and continue execution.

---

## 📁 Repository structure (Production)

```text
src/
├── orchestrator.py     # Graph definition & Lambda entrypoint
├── slack_agent.py      # Node definitions for Jira workflows
├── calendar_agent.py   # Node definitions for Google Calendar workflows
├── meeting_agent.py    # Node definitions for Meeting Summarization
├── state.py            # OrchestratorState (Graph State) & LLM Factory
├── redis_store.py      # Redis session persistence & ChromaDB hooks
└── rag_retriever.py    # ChromaDB RAG logic for prompt injection
```

---

## 📊 RLHF & Telemetry

The system includes a native **Reinforcement Learning from Human Feedback (RLHF)** loop.

### 1. Synchronous Logging
Every time a user clicks **Approve** or **Cancel** in Slack:
- The Lambda captures the exact prompt used, the model's response, and the human's decision.
- It calculates a 1536-dimensional embedding using `onnxruntime` (cached in `/tmp`).
- It ships this telemetry to **ChromaDB** on your EC2 instance (Port 8001).

### 2. In-Context Learning (RAG)
On every new request, the system queries ChromaDB for **past successful interactions**. These are injected into the system prompt as "Few-Shot" examples, causing the model to learn your preferences instantly without training.

### 3. Offline Fine-Tuning (DPO)
Data in ChromaDB is used by the scripts in `rlhf/` to build preference pairs (`chosen` vs `rejected`) for **Direct Preference Optimization** training.

---

## 🔧 Infrastructure & Ports

### EC2 GPU Instance
| Port | Service | Purpose |
|---|---|---|
| 8000 | vLLM | OpenAI-compatible inference API |
| 8001 | ChromaDB | Vector database for RLHF |

### AWS Lambda
| Configuration | Value |
|---|---|
| Runtime | Python 3.11 (Container) |
| Timeout | 60 Seconds (Required for LLM latency) |
| Memory | 1024 MB |
| Env: `HOME` | `/tmp` (Required for Read-Only FS bypass) |

---

## 🛠️ Configuration Checklist

The following environment variables **must** be set in the Lambda console:

| Variable | Description |
|---|---|
| `EC2_IP` | Public IP of your EC2 instance |
| `REDIS_URL` | Upstash Redis connection string |
| `SLACK_BOT_TOKEN` | Bot User OAuth Token |
| `GOOGLE_TOKEN_JSON` | Google OAuth credentials (stringified JSON) |
| `JIRA_API_TOKEN` | Jira API Personal Access Token |
| `JIRA_EMAIL` | Your Jira login email |
| `JIRA_BASE_URL` | `https://yourdomain.atlassian.net` |

---

## 🧪 Local Testing

You can simulate a Slack event locally without deploying to AWS:

```python
from src.orchestrator import handler

# Mock Slack Message
event = {
    "body": '{"event": {"text": "Create a jira for ketki", "channel": "C123"}}'
}

response = handler(event, None)
print(response)
```
MemorySaver` or missing retry guard | Check CloudWatch for a second invocation without the retry header |

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