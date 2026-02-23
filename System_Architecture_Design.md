# 5. System Architecture Design

## AI-Powered Autonomous Workforce Assistants
**DATA 298B — MSDA Project II, Team 5**
**Aishanee Sinha, Debika Choudhury, Ketki Maddiwar, Leela Dammalapati, Utkarsh Tripathi**

---

## 5.1 System Requirements Analysis

### 5.1.1 System Boundary, Actors, and Use Cases

#### System Boundary

The **AI-Powered Autonomous Workforce Assistant** system boundary encompasses all AI-driven automation components that sit between enterprise communication platforms (Slack, Email, Google Calendar) and project management tools (Jira). The system ingests unstructured communication data, processes it through specialized AI agents, and produces structured outputs (calendar events, Jira tickets, summaries, bottleneck alerts). The boundary includes:

- **Inside the boundary:** AI Agent modules (Scheduler, Summarizer, Task Extractor, Bottleneck Detector), the Chain of Debate (CoD) orchestration layer, the data ingestion pipeline (Airflow DAGs, S3, Snowflake), fine-tuned LLM models, the vector database, the monitoring/logging dashboard, and the user-facing Slack bot / web interface.
- **Outside the boundary:** External platforms (Slack API, Gmail/SMTP, Google Calendar API, Jira API, Kaggle datasets, HuggingFace model hub), end-user devices, and third-party cloud infrastructure (AWS compute, Snowflake warehouse).

#### Actors

| Actor | Type | Description |
|-------|------|-------------|
| **Team Member / End User** | Primary | Sends messages in Slack, writes emails, attends meetings. Receives AI-generated summaries, calendar invites, and task notifications. Can accept, reject, or override AI suggestions. |
| **Project Manager** | Primary | Monitors dashboards for bottleneck alerts and workflow health. Reviews AI-generated Jira tasks and priority suggestions. Configures system preferences. |
| **System Administrator** | Primary | Deploys, configures, and maintains the system. Manages API keys, model versions, and pipeline schedules. |
| **Slack API** | External System | Provides real-time message streams and bot interaction endpoints. |
| **Gmail / Email System** | External System | Supplies email data for meeting extraction and scheduling. |
| **Google Calendar API** | External System | Receives generated .ics invites and provides availability data. |
| **Jira API** | External System | Receives structured task objects and provides workflow/issue data. |
| **AWS (S3, EC2)** | External System | Provides cloud storage (raw/processed data zones) and compute resources. |
| **Snowflake** | External System | Data warehouse for analytics queries and reporting. |
| **HuggingFace Hub** | External System | Hosts base and fine-tuned model weights (Mistral-7B, FLAN-T5, etc.). |

#### Use Cases

**UC-1: Automated Meeting Scheduling**
- **Actor:** Team Member
- **Precondition:** User sends a Slack message or email containing a meeting request (e.g., "Let's meet tomorrow at 3 PM to discuss the Sprint review").
- **Flow:** (1) System detects scheduling intent via the Scheduler Agent (Flan-T5 + BERT NER). (2) System extracts date/time, participants, and topic. (3) System queries Google Calendar API for participant availability. (4) System generates an optimal time slot and creates a structured .ics calendar invite. (5) System sends the invite via email/Slack and logs the event.
- **Postcondition:** Calendar event is created; participants are notified.
- **Success Metric:** Intent detection accuracy ≥ 90%; invite creation success rate measured via API logs.

**UC-2: Conversation & Document Summarization**
- **Actor:** Team Member, Project Manager
- **Precondition:** A lengthy Slack thread, email chain, or meeting transcript is available.
- **Flow:** (1) Summarizer Agent receives the text input. (2) Text is preprocessed (HTML stripping, normalization). (3) For long inputs, token-aware chunking with overlap is applied. (4) Fine-tuned FLAN-T5-Large (LoRA) generates an abstractive summary with action items. (5) Summary is delivered via Slack bot or web dashboard.
- **Postcondition:** Concise summary with highlighted action items is available.
- **Success Metric:** ROUGE-L > 0.85; human rating ≥ 4/5; G-Eval composite score.

**UC-3: Automated Task Extraction to Jira**
- **Actor:** Team Member, Project Manager
- **Precondition:** Slack conversations contain implicit or explicit action items.
- **Flow:** (1) Task Extractor Agent processes Slack messages. (2) Fine-tuned Mistral-7B (QLoRA) extracts task_summary, assignee, and issue_creation_date as structured JSON. (3) System validates extracted fields. (4) System calls Jira API to create issues automatically. (5) User is notified in Slack with a link to the created Jira ticket.
- **Postcondition:** Jira issue is created with correct metadata.
- **Success Metric:** Action Item Detection F1 ≥ 0.80; Metadata Extraction Accuracy; Integration Success Rate via API logs.

**UC-4: Workflow Bottleneck Detection & Alerts**
- **Actor:** Project Manager
- **Precondition:** Jira project has active tasks with workflow history.
- **Flow:** (1) Bottleneck Detector Agent ingests Jira workflow data. (2) Sentence embeddings (all-MiniLM-L6-v2) and time-series anomaly detection identify stuck or overloaded tasks. (3) System generates re-prioritization suggestions. (4) Alerts are surfaced on the monitoring dashboard and/or Slack. (5) Project Manager accepts or overrides suggestions.
- **Postcondition:** Bottleneck alerts are raised; workflow adjustments are suggested.
- **Success Metric:** Detection accuracy ≥ 85%; False positive rate monitored; User acceptance rate of suggestions.

**UC-5: Chain of Debate (CoD) Multi-Agent Collaboration**
- **Actor:** System (internal)
- **Precondition:** Multiple agents have produced independent outputs for the same context.
- **Flow:** (1) Individual agents propose their outputs. (2) CoD orchestrator initiates structured debate rounds. (3) Agents critique, refine, and converge on improved outputs. (4) A Judge Agent evaluates and selects the final output. (5) Final refined output is delivered to the user.
- **Postcondition:** Higher-quality, validated output compared to single-agent execution.
- **Success Metric:** Decision quality improvement vs. single-agent baseline; convergence rate; token efficiency.

**UC-6: System Monitoring & Logging**
- **Actor:** System Administrator, Project Manager
- **Precondition:** System is operational.
- **Flow:** (1) All agent activities, API calls, and errors are logged. (2) Monitoring dashboard displays real-time metrics (latency, throughput, error rates). (3) Admin can drill down into specific agent logs for debugging.
- **Postcondition:** Full audit trail is available; anomalies are flagged.

---

### 5.1.2 High-Level Data Analytics and Machine Learning Functions and Capabilities

#### Data Analytics Functions

| Function | Description | Techniques |
|----------|-------------|------------|
| **Text Preprocessing & Cleaning** | Normalizes raw Slack XML, email MIME, and meeting transcripts. Removes HTML, signatures, duplicates, and noise. | Regex, pandas, MIME parsing, HTML stripping, dateparser |
| **Semantic Embedding & Similarity Analysis** | Converts messages to vector representations for task-relevance scoring. | SentenceTransformer (all-MiniLM-L6-v2), cosine similarity |
| **Data Pipeline Orchestration** | End-to-end ETL from raw sources (Kaggle, GitHub, HuggingFace) → S3 Raw Zone → S3 Processed Zone → Snowflake. | Apache Airflow DAGs, AWS S3, Snowflake |
| **Exploratory Data Analysis** | Distribution analysis, data quality checks, score distributions, and split validation. | pandas, matplotlib, seaborn, descriptive statistics |
| **Time-Series Anomaly Detection** | Identifies abnormal patterns in task completion timelines and workflow metrics. | Keras time-series anomaly detection, statistical methods |
| **Workflow & Process Mining** | Analyzes Jira event logs to discover process patterns and predict bottlenecks. | Graph-based process mining, Graph Neural Networks (GNNs) |

#### Machine Learning Capabilities

| Capability | Model(s) | Technique | Purpose |
|------------|----------|-----------|---------|
| **Meeting Intent Detection & Scheduling** | Flan-T5 + BERT NER + spaCy | Instruction-following + Named Entity Recognition | Detect scheduling intent, extract date/time/participants, generate calendar events |
| **Abstractive Summarization** | FLAN-T5-Large (fine-tuned via LoRA, rank-32) | Few-shot Chain-of-Thought prompting, LoRA fine-tuning (3% params) | Generate structured meeting summaries with action items |
| **Summarization Baselines** | FLAN-T5-Base, BART-CNN, PEGASUS-XSUM | Transfer learning, pre-trained summarization | Baseline comparison for summarization quality |
| **Task Extraction (Slack → Jira)** | Mistral-7B-Instruct-v0.3 (QLoRA, 4-bit) | QLoRA fine-tuning (rank=16, alpha=32), SFTTrainer | Extract structured Jira tasks from Slack messages |
| **Email → Calendar Event Extraction** | Mistral-7B-Instruct-v0.2 (LoRA, 4-bit) | LoRA fine-tuning on [INST] templates, custom loss masking | Convert emails to structured JSON → .ics files |
| **Weak Supervision & Labeling** | Gemini Flash 2.5 + Manual Review | LLM-assisted labeling, human-in-the-loop validation | Label 100K+ Slack messages for task extraction training |
| **Zero-Shot Classification** | BART-large-MNLI, DistilBERT-MNLI, Flan-T5-small | Zero-shot text classification | Action item detection, intent classification |
| **Bottleneck Detection** | Sentence embeddings + Time-series Transformer | Anomaly detection, reinforcement learning | Identify stuck tasks and suggest re-prioritization |
| **Multi-Agent Debate (CoD)** | All agents + Judge Agent | Chain of Debate, tit-for-tat argumentation, consensus protocols | Collaborative output refinement across agents |
| **Continuous Learning** | RLHF / DPO | Reinforcement Learning from Human Feedback, Direct Preference Optimization | Adapt agent behavior based on user feedback over time |

#### Evaluation Framework

| Agent | Key Metrics |
|-------|------------|
| **Scheduler** | Intent Detection Accuracy, Slot Selection Success Rate, Invite Creation Success Rate |
| **Summarizer** | ROUGE-1/2/L, Coherence, Consistency, Fluency, Relevance, Action Quality (G-Eval), Flesch Reading Ease |
| **Task Extractor** | Action Item Detection F1, Metadata Extraction Accuracy, Integration Success Rate |
| **Bottleneck Detector** | Detection Accuracy (≥85%), False Positive Rate, Precision/Recall |
| **CoD System** | Decision Quality Improvement, Convergence Rate, Token Efficiency |
| **Overall System** | P95 Latency (<120ms), User Satisfaction (1-5), Override Rate, Suggestion Acceptance Rate |

---

## 5.2 System Design

### 5.2.1 System Architecture and Infrastructure with AI-Powered Function Components

#### High-Level Architecture Overview

The system follows a **layered microservices architecture** with four distinct tiers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                               │
│   ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐    │
│   │  Slack Bot    │  │  Web UI /    │  │  Monitoring & Logging     │    │
│   │  Interface    │  │  Dashboard   │  │  Dashboard                │    │
│   └──────┬───────┘  └──────┬───────┘  └─────────┬─────────────────┘    │
└──────────┼─────────────────┼────────────────────┼──────────────────────┘
           │                 │                    │
┌──────────┼─────────────────┼────────────────────┼──────────────────────┐
│          ▼                 ▼                    ▼                       │
│                    ORCHESTRATION LAYER                                  │
│   ┌────────────────────────────────────────────────────────────────┐   │
│   │              Chain of Debate (CoD) Orchestrator                │   │
│   │    ┌──────────┐  ┌──────────┐  ┌─────────┐  ┌─────────────┐  │   │
│   │    │  Debate   │  │  Judge   │  │ Routing │  │  Consensus  │  │   │
│   │    │  Manager  │  │  Agent   │  │ Engine  │  │  Protocol   │  │   │
│   │    └──────────┘  └──────────┘  └─────────┘  └─────────────┘  │   │
│   └────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
┌───────────────────────────────┼────────────────────────────────────────┐
│                               ▼                                        │
│                      AI AGENT LAYER                                    │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │  Scheduler   │ │  Summarizer  │ │    Task      │ │  Bottleneck  │  │
│  │    Agent     │ │    Agent     │ │  Extractor   │ │  Detector    │  │
│  │              │ │              │ │    Agent     │ │    Agent     │  │
│  │ Flan-T5 +   │ │ FLAN-T5-Large│ │ Mistral-7B   │ │ Embeddings + │  │
│  │ BERT NER    │ │ (LoRA)       │ │ (QLoRA)      │ │ Time-Series  │  │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘  │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                                   │                                    │
│                    ┌──────────────┼──────────────┐                     │
│                    │  Shared Services            │                     │
│                    │  • Vector DB (FAISS/Chroma) │                     │
│                    │  • Model Registry           │                     │
│                    │  • RLHF Feedback Loop        │                     │
│                    └─────────────────────────────┘                     │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
┌───────────────────────────────┼────────────────────────────────────────┐
│                               ▼                                        │
│                      DATA LAYER                                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │  AWS S3      │ │  Snowflake   │ │  PostgreSQL  │ │  Redis       │  │
│  │  (Raw +      │ │  (Analytics  │ │  (App State  │ │  (Cache +    │  │
│  │  Processed)  │ │  Warehouse)  │ │  + Metadata) │ │  Sessions)   │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
└───────────────────────────────┬────────────────────────────────────────┘
                                │
┌───────────────────────────────┼────────────────────────────────────────┐
│                               ▼                                        │
│                 EXTERNAL INTEGRATION LAYER                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ Slack API │  │ Gmail /  │  │ Google Cal   │  │ Jira API       │    │
│  │          │  │ SMTP     │  │ API          │  │                │    │
│  └──────────┘  └──────────┘  └──────────────┘  └────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

#### AI-Powered Function Components Detail

**1. Scheduler Agent**
- **Input:** Raw Slack messages or email text containing scheduling language.
- **Processing Pipeline:**
  - Intent detection via Flan-T5 instruction-following model.
  - Entity extraction (dates, times, attendees) using BERT-based NER + spaCy.
  - Availability check via Google Calendar API query.
  - Conflict resolution and optimal slot selection algorithm.
- **Output:** Structured JSON → .ics file → Google Calendar event creation.
- **Infrastructure:** GPU inference endpoint (AWS EC2 with NVIDIA GPU or Google Colab Pro for prototyping).

**2. Summarizer Agent**
- **Input:** Meeting transcripts (AMI corpus format), long Slack threads, or email chains.
- **Processing Pipeline:**
  - Text preprocessing and normalization.
  - Token-aware chunking with overlap for long documents.
  - Abstractive summarization via fine-tuned FLAN-T5-Large (LoRA rank-32, 3% trainable params).
  - Chain-of-Thought (CoT) prompting for structured output.
  - Post-processing: action item extraction and language formalization.
- **Output:** Structured summary with highlights + action items (CSV + formatted text).
- **Evaluation:** G-Eval composite score (Coherence 20%, Consistency 25%, Fluency 10%, Relevance 25%, Action Quality 20%).

**3. Task Extractor Agent**
- **Input:** Slack messages (105K+ messages from pythondev community, 100K labeled via Gemini).
- **Processing Pipeline:**
  - Semantic embedding via SentenceTransformer (all-MiniLM-L6-v2).
  - Task-relevance scoring using cosine similarity against reference task anchors.
  - Fine-tuned Mistral-7B-Instruct-v0.3 (QLoRA 4-bit, rank=16, alpha=32) for structured extraction.
  - Output validation and Jira API integration.
- **Output:** JSON with `task_summary`, `assignee`, `issue_creation_date` → Jira issue.

**4. Bottleneck Detector Agent**
- **Input:** Jira workflow data, task queues, and completion timelines.
- **Processing Pipeline:**
  - Zero-shot classification (BART-large-MNLI) for initial categorization.
  - Sentence embeddings for weak supervision.
  - Time-series anomaly detection (Keras-based) for stuck task identification.
  - Graph-based process mining for dependency analysis.
  - Re-prioritization suggestion generation.
- **Output:** Bottleneck alerts, priority adjustment recommendations, workflow health score.

**5. Chain of Debate (CoD) Orchestrator**
- **Purpose:** Coordinates multi-agent collaboration to improve output quality.
- **Components:**
  - **Debate Manager:** Initiates structured debate rounds between agents when cross-functional context is relevant.
  - **Judge Agent:** Evaluates competing proposals and selects the optimal output, addressing convergence-to-incorrect-answer failure modes (per Wynn et al.).
  - **Routing Engine:** Determines which agents need to participate based on input context.
  - **Consensus Protocol:** Implements GroupDebate (parallel sub-group debates) and Free-MAD (consensus-free) strategies for efficiency.
  - **Shared Knowledge Base:** Vector database (inspired by MADKE framework) for agents to share context.

---

### 5.2.2 Integration with Supporting Platforms, Frameworks, and Cloud Environment

#### Platform Integration Architecture

```
┌─────────────────────────────────────────────┐
│              CLOUD ENVIRONMENT (AWS)         │
│                                             │
│  ┌─────────┐    ┌──────────┐    ┌────────┐  │
│  │ EC2     │    │ S3       │    │ Lambda │  │
│  │ (GPU    │    │ Raw Zone │    │ (Event │  │
│  │ Inference│    │ Processed│    │ Triggers│  │
│  │ Server) │    │ Zone     │    │)       │  │
│  └────┬────┘    └────┬─────┘    └───┬────┘  │
│       │              │              │        │
│       └──────────────┼──────────────┘        │
│                      │                       │
│            ┌─────────▼──────────┐            │
│            │   Snowflake        │            │
│            │   Data Warehouse   │            │
│            └────────────────────┘            │
└──────────────────────┬──────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐  ┌──────▼────┐  ┌─────▼─────┐
   │ Apache  │  │ Docker    │  │ FastAPI   │
   │ Airflow │  │ Containers│  │ Backend   │
   │ (DAGs)  │  │           │  │ Server    │
   └─────────┘  └───────────┘  └───────────┘
```

#### Framework Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Framework** | PyTorch, Hugging Face Transformers | Model training, fine-tuning (LoRA/QLoRA), inference |
| **ML Training** | Hugging Face SFTTrainer, PEFT, bitsandbytes | Parameter-efficient fine-tuning, 4-bit quantization |
| **Agent Orchestration** | LangChain / AutoGen | Multi-agent conversation management, tool integration |
| **Data Pipeline** | Apache Airflow | DAG-based ETL orchestration with monitoring |
| **Data Processing** | pandas, scikit-learn, regex | Preprocessing, train/val/test splitting, feature engineering |
| **NLP Utilities** | spaCy, SentenceTransformers, dateparser | NER, embeddings, date normalization |
| **Vector Database** | FAISS / ChromaDB | Semantic search, shared knowledge base for CoD |
| **Backend API** | FastAPI (Python) | RESTful API for agent endpoints and webhook handlers |
| **Containerization** | Docker, Docker Compose | Reproducible deployment, service isolation |
| **Version Control** | GitHub | Source code management and collaboration |

#### Cloud Environment Design

| Service | Usage | Tier |
|---------|-------|------|
| **AWS S3** | Raw Zone (immutable ingested data) + Processed Zone (cleaned/transformed data) | Free tier / student credits |
| **AWS EC2** | GPU instances for model inference (p3.2xlarge or g4dn.xlarge) | Student credits |
| **AWS Lambda** | Event-driven triggers for new Slack messages/emails | Free tier |
| **Snowflake** | Analytics data warehouse for ad-hoc queries and reporting | Free tier |
| **Google Colab Pro** | Model training and experimentation (NVIDIA T4/A100) | Academic license |

#### API Integration Specifications

| External API | Integration Method | Auth | Data Flow |
|-------------|-------------------|------|-----------|
| **Slack API** | WebSocket (Real-Time Events) + REST (Bot) | OAuth 2.0, Bot Token | Inbound: messages, threads → System; Outbound: summaries, alerts → Slack channels |
| **Gmail API** | REST API + OAuth2 | OAuth 2.0, credentials.json + token.json | Inbound: emails → System; Outbound: formatted responses |
| **Google Calendar API** | REST API | OAuth 2.0 (already configured with credentials.json) | Bidirectional: read availability, create/update events |
| **Jira API** | REST API v3 | API Token / OAuth 2.0 | Outbound: create issues, update priorities; Inbound: workflow data, task statuses |
| **Kaggle API** | CLI / REST | API Key | Inbound: dataset downloads (Enron corpus) |
| **HuggingFace Hub** | Python SDK (transformers) | API Token | Inbound: base model weights; Outbound: fine-tuned model uploads |

---

### 5.2.3 Data Management Solution and Database Design

#### Data Flow Architecture

```
  DATA SOURCES                  INGESTION              STORAGE              PROCESSING           SERVING
  ───────────                  ─────────              ───────              ──────────           ───────

  Slack XML        ──┐
  (GitHub)           │     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                     ├────▶│  Apache      │───▶│  AWS S3      │───▶│  Transform   │───▶│  Snowflake   │
  Enron Emails     ──┤     │  Airflow     │    │  Raw Zone    │    │  & Clean     │    │  Warehouse   │
  (Kaggle)           │     │  DAG         │    │              │    │              │    │              │
                     ├────▶│              │    └──────┬───────┘    └──────┬───────┘    └──────────────┘
  AMI Corpus       ──┤     └──────────────┘           │                  │
  (HuggingFace)      │                                ▼                  ▼
                     │                         ┌──────────────┐   ┌──────────────┐
  Jira API         ──┤                         │  AWS S3      │   │  PostgreSQL  │
                     │                         │  Processed   │   │  App DB      │
  Google Cal API   ──┘                         │  Zone        │   │              │
                                               └──────────────┘   └──────────────┘
```

#### Database Schema Design

**PostgreSQL — Application Database**

```sql
-- Users and authentication
CREATE TABLE users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    display_name    VARCHAR(100),
    slack_user_id   VARCHAR(50),
    role            VARCHAR(20) DEFAULT 'member',  -- member, manager, admin
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Slack messages (ingested and processed)
CREATE TABLE slack_messages (
    message_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slack_ts        VARCHAR(50) UNIQUE,
    channel_id      VARCHAR(50),
    user_id         UUID REFERENCES users(user_id),
    raw_text        TEXT NOT NULL,
    cleaned_text    TEXT,
    embedding_vector BYTEA,  -- serialized embedding from all-MiniLM-L6-v2
    task_score      FLOAT,   -- combined similarity + heuristic score
    is_task         BOOLEAN DEFAULT FALSE,
    processed_at    TIMESTAMP DEFAULT NOW()
);

-- Extracted tasks (Slack → Jira mapping)
CREATE TABLE extracted_tasks (
    task_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_message_id UUID REFERENCES slack_messages(message_id),
    task_summary    TEXT,
    assignee        VARCHAR(100),
    priority        VARCHAR(20),        -- low, medium, high, critical
    issue_creation_date DATE,
    jira_issue_key  VARCHAR(20),        -- e.g., PROJ-123
    jira_sync_status VARCHAR(20) DEFAULT 'pending',  -- pending, synced, failed
    confidence_score FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Email records
CREATE TABLE emails (
    email_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject         VARCHAR(500),
    body            TEXT,
    sender          VARCHAR(255),
    recipients      TEXT[],             -- array of recipient emails
    email_date      TIMESTAMP,
    has_meeting     BOOLEAN DEFAULT FALSE,
    processed       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Calendar events (extracted from emails)
CREATE TABLE calendar_events (
    event_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_email_id UUID REFERENCES emails(email_id),
    title           VARCHAR(500),
    start_time      TIMESTAMP WITH TIME ZONE,
    end_time        TIMESTAMP WITH TIME ZONE,
    location        VARCHAR(500),
    attendees       TEXT[],
    ics_file_path   VARCHAR(500),
    gcal_event_id   VARCHAR(100),       -- Google Calendar event ID
    sync_status     VARCHAR(20) DEFAULT 'pending',
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Meeting summaries
CREATE TABLE meeting_summaries (
    summary_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id      VARCHAR(50),
    transcript_path VARCHAR(500),
    summary_text    TEXT,
    action_items    JSONB,              -- [{owner, task, due_date}]
    model_used      VARCHAR(50),        -- e.g., flan-t5-large-lora
    coherence_score FLOAT,
    consistency_score FLOAT,
    fluency_score   FLOAT,
    relevance_score FLOAT,
    action_quality_score FLOAT,
    production_score FLOAT,             -- weighted composite
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Bottleneck alerts
CREATE TABLE bottleneck_alerts (
    alert_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jira_issue_key  VARCHAR(20),
    alert_type      VARCHAR(50),        -- stuck, overloaded, dependency_blocked
    severity        VARCHAR(20),        -- low, medium, high, critical
    description     TEXT,
    suggested_action TEXT,
    user_response   VARCHAR(20),        -- accepted, rejected, ignored
    resolved_at     TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Agent activity logs (for monitoring & RLHF)
CREATE TABLE agent_logs (
    log_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name      VARCHAR(50),        -- scheduler, summarizer, extractor, bottleneck, cod
    action_type     VARCHAR(50),
    input_hash      VARCHAR(64),
    output_summary  TEXT,
    latency_ms      INTEGER,
    success         BOOLEAN,
    error_message   TEXT,
    user_feedback   INTEGER,            -- 1-5 rating (for RLHF)
    created_at      TIMESTAMP DEFAULT NOW()
);

-- CoD debate sessions
CREATE TABLE debate_sessions (
    session_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trigger_context TEXT,
    participating_agents TEXT[],
    num_rounds      INTEGER,
    final_output    JSONB,
    consensus_reached BOOLEAN,
    judge_reasoning TEXT,
    total_tokens_used INTEGER,
    created_at      TIMESTAMP DEFAULT NOW()
);
```

#### Data Storage Strategy

| Data Category | Storage | Format | Retention |
|---------------|---------|--------|-----------|
| Raw ingested data (Slack XML, emails, transcripts) | AWS S3 Raw Zone | XML, CSV, TXT | Immutable, indefinite |
| Cleaned/preprocessed data | AWS S3 Processed Zone | CSV, JSONL | Versioned, 1 year |
| Training datasets (labeled) | AWS S3 + local | JSONL (train/val/test splits) | Versioned with model |
| Model weights (base + fine-tuned) | HuggingFace Hub + S3 | Safetensors / PyTorch | Per model version |
| Embeddings & vectors | FAISS / ChromaDB | Binary index files | Rebuilt per model update |
| Application state | PostgreSQL | Relational | Indefinite |
| Analytics & reporting | Snowflake | Columnar | 2 years |
| Cache & sessions | Redis | Key-Value | TTL-based (24h) |

---

### 5.2.4 User Interface and Data Visualization Design

#### User Interface Design

The system provides two primary user interfaces:

**1. Slack Bot Interface (Primary Interaction Channel)**

The Slack bot provides conversational AI interaction directly within the user's workflow:

```
┌─────────────────────────────────────────────────────┐
│  #project-alpha                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  👤 John: Can we schedule a meeting tomorrow at 3PM  │
│     to discuss the sprint review? @sarah @mike       │
│                                                      │
│  🤖 WorkforceBot:                                    │
│  ┌─────────────────────────────────────────────┐    │
│  │ 📅 Meeting Scheduled                         │    │
│  │                                              │    │
│  │ Title: Sprint Review Discussion              │    │
│  │ Time: Feb 24, 2026 3:00 PM – 4:00 PM PST   │    │
│  │ Attendees: John, Sarah, Mike                 │    │
│  │ Calendar: ✅ Added to Google Calendar         │    │
│  │                                              │    │
│  │ [✅ Confirm] [✏️ Edit] [❌ Cancel]            │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  🤖 WorkforceBot:                                    │
│  ┌─────────────────────────────────────────────┐    │
│  │ 📋 Task Detected                             │    │
│  │                                              │    │
│  │ Summary: Prepare sprint review presentation  │    │
│  │ Assignee: @john                              │    │
│  │ Due: Feb 24, 2026                            │    │
│  │ Jira: PROJ-456 (created)                     │    │
│  │                                              │    │
│  │ [👍 Accept] [✏️ Edit] [🚫 Dismiss]           │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
└─────────────────────────────────────────────────────┘
```

- **Commands:** `/summarize [channel/thread]`, `/extract-tasks`, `/bottlenecks`, `/schedule [details]`
- **Notifications:** Proactive alerts for bottlenecks, summary digests, task reminders.
- **Feedback:** Inline buttons for Accept/Edit/Dismiss → feeds directly into RLHF loop.

**2. Web Dashboard (Monitoring & Analytics)**

```
┌──────────────────────────────────────────────────────────────────────┐
│  🏠 AI Workforce Assistant Dashboard            [Admin ▼] [⚙️]  │
├──────────┬───────────────────────────────────────────────────────────┤
│          │                                                           │
│ 📊 Overview│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │
│ 📅 Schedule│  │ Tasks       │ │ Meetings    │ │ Bottlenecks │      │
│ 📋 Tasks  │  │ Extracted   │ │ Scheduled   │ │ Detected    │      │
│ 📝 Summaries│ │   47 today  │ │   12 today  │ │   3 active  │      │
│ ⚠️ Alerts │  │ ↑ 12%       │ │ ↑ 5%        │ │ ↓ 20%       │      │
│ 📈 Analytics│ └─────────────┘ └─────────────┘ └─────────────┘      │
│ ⚙️ Settings│                                                        │
│          │  ┌──────────────────────────────────────────────────┐    │
│          │  │  Agent Activity Timeline              (7 days)  │    │
│          │  │  ████████████████░░░░░░░░░░░░░░░░░░░░░          │    │
│          │  │  Scheduler ━━━━━━ Summarizer ━━━━━━             │    │
│          │  │  Extractor ━━━━━━ Bottleneck ━━━━━━             │    │
│          │  └──────────────────────────────────────────────────┘    │
│          │                                                           │
│          │  ┌──────────────────────┐ ┌────────────────────────┐    │
│          │  │ Recent Summaries     │ │ Bottleneck Alerts      │    │
│          │  │                      │ │                        │    │
│          │  │ • Sprint Review      │ │ ⚠️ PROJ-234: Stuck    │    │
│          │  │   Feb 22 | Score:8.4 │ │   5 days, High        │    │
│          │  │ • Design Sync        │ │ ⚠️ PROJ-289: Blocked  │    │
│          │  │   Feb 21 | Score:7.9 │ │   3 days, Medium      │    │
│          │  │ • Standup            │ │ ✅ PROJ-301: Resolved  │    │
│          │  │   Feb 21 | Score:9.1 │ │   Auto-reprioritized  │    │
│          │  └──────────────────────┘ └────────────────────────┘    │
│          │                                                           │
└──────────┴───────────────────────────────────────────────────────────┘
```

#### Data Visualization Components

| Visualization | Type | Purpose | Library |
|---------------|------|---------|---------|
| **Agent Activity Timeline** | Multi-line time series | Track agent invocations and throughput over time | Plotly / D3.js |
| **Task Extraction Funnel** | Funnel chart | Show conversion: raw messages → scored → extracted → synced to Jira | Plotly |
| **Bottleneck Heatmap** | Heatmap / Treemap | Visualize task blockages by project, assignee, and severity | Plotly / Seaborn |
| **Summary Quality Radar** | Radar/Spider chart | Display G-Eval metrics (Coherence, Consistency, Fluency, Relevance, Action Quality) per summary | Chart.js / Plotly |
| **Similarity Score Distribution** | Histogram | Distribution of task-relevance scores across Slack messages | matplotlib / Plotly |
| **Model Performance Comparison** | Grouped bar chart | Compare base vs. fine-tuned models on ROUGE, latency, GPU memory | matplotlib |
| **Pipeline Health Monitor** | Status cards + Gantt | Real-time Airflow DAG status, data freshness indicators | Custom React + Airflow API |
| **CoD Debate Visualization** | Flow/Sankey diagram | Show agent debate rounds, proposals, critiques, and final decisions | D3.js |
| **User Feedback Distribution** | Pie/Donut chart | Breakdown of user responses (Accepted/Edited/Dismissed) per agent | Chart.js |
| **Latency Dashboard** | Gauge + Line chart | P50/P95/P99 latency per agent, with 120ms SLA line | Grafana / Plotly |

#### Technology Stack for UI

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Frontend Framework** | React.js with TypeScript | Component-based architecture; rich ecosystem for dashboards |
| **UI Component Library** | Material UI (MUI) or Ant Design | Professional enterprise-grade components |
| **Charting** | Plotly.js + D3.js | Interactive, publication-quality visualizations |
| **State Management** | React Query + Zustand | Server-state caching + lightweight client state |
| **Slack Bot** | Bolt for Python (slack-bolt) | Official Slack SDK with event handling and interactive messages |
| **Real-time Updates** | WebSocket (Socket.IO) | Live dashboard updates for agent activity and alerts |
| **Backend API** | FastAPI (Python) | High-performance async API; auto-generated OpenAPI docs |
| **Authentication** | OAuth 2.0 (Slack SSO + Google) | Seamless enterprise authentication |

---

## Appendix: Technology Stack Summary

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+, TypeScript, SQL |
| **ML/AI** | PyTorch, Hugging Face (Transformers, PEFT, bitsandbytes), SentenceTransformers, spaCy |
| **Models** | Mistral-7B-Instruct (v0.2, v0.3), FLAN-T5 (Base, Large), BART-CNN, PEGASUS-XSUM, Gemini Flash 2.5 |
| **Fine-tuning** | LoRA, QLoRA (4-bit), SFTTrainer, AdamW, Cosine LR scheduler |
| **Orchestration** | LangChain / AutoGen, Chain of Debate (custom) |
| **Data Pipeline** | Apache Airflow, pandas, scikit-learn |
| **Cloud** | AWS (S3, EC2, Lambda), Snowflake, Google Colab Pro |
| **Databases** | PostgreSQL, Redis, FAISS/ChromaDB, Snowflake |
| **APIs** | Slack API, Gmail API, Google Calendar API, Jira API, Kaggle API, HuggingFace Hub |
| **Deployment** | Docker, Docker Compose, FastAPI |
| **Frontend** | React.js, Plotly.js, D3.js, Material UI |
| **Monitoring** | Grafana, custom logging dashboard, Apache Airflow UI |
| **Version Control** | GitHub |
