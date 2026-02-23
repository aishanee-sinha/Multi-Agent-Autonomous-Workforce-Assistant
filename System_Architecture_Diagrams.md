# System Architecture Diagrams (Updated)

**AI-Powered Autonomous Workforce Assistant**
DATA 298B — MSDA Project II, Team 5

---

## 1. Use Case Diagram
**Description:** Defines the system boundary, primary actors, and core functional requirements.

```mermaid
graph LR
    subgraph ExternalSystems["External Systems"]
        SLACK["Slack API"]
        GCAL["Google Calendar API"]
        GMAIL["Gmail API"]
        JIRA["Jira API"]
    end

    subgraph SystemBoundary["AI-Powered Autonomous Workforce Assistant"]
        UC1["UC-1: Schedule Meeting"]
        UC2["UC-2: Summarize Conversation"]
        UC3["UC-3: Extract Tasks to Jira"]
        UC4["UC-4: Detect Bottlenecks"]
        UC5["UC-5: Chain of Debate\n(Refinement)"]
        UC6["UC-6: Collect Feedback\n(RLHF)"]
    end

    TM["Team Member"]
    PM["Project Manager"]
    SA["System Admin"]

    TM --> UC1
    TM --> UC2
    TM --> UC3
    PM --> UC2
    PM --> UC4
    PM --> UC6
    SA --> UC6

    UC1 --> GCAL
    UC1 --> GMAIL
    UC2 --> SLACK
    UC3 --> JIRA
    UC4 --> JIRA
    UC4 --> GCAL

    style SystemBoundary fill:#fdf4ff,stroke:#d946ef,stroke-width:2px
    style UC5 fill:#f0f9ff,stroke:#0ea5e9
    style UC6 fill:#f0fdf4,stroke:#22c55e
```

---

## 2. Workflow Diagram
**Description:** Step-by-step sequence from user request in Slack to final execution and feedback.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Slack as Slack Interface
    participant App as Backend (Cloud Run)
    participant Qwen as Qwen-14B (Vertex AI)
    participant DB as Cloud SQL / ChromaDB
    participant Ext as External APIs

    User->>Slack: Sends Message
    Slack->>App: Webhook / Event
    App->>Qwen: Route Intent
    Qwen-->>App: Intent (e.g. Task)
    
    rect rgb(240, 240, 240)
        Note over Qwen, DB: Chain of Debate
        App->>Qwen: Generate Proposal
        Qwen-->>App: Proposal (Draft)
        App->>DB: Fetch Context (ChromaDB)
        App->>Qwen: Criticize Proposal
        Qwen-->>App: Critique / Flaws
        App->>Qwen: Refine & Judge
        Qwen-->>App: Final Output
    end

    App->>Ext: Execute Action (Jira/Cal)
    App->>Slack: Posted Result w/ Buttons
    User->>Slack: Clicks [Accept/Edit/Dismiss]
    Slack->>App: Feedback Event
    App->>DB: Store in RLHF Log (Cloud SQL)
```

---

## 3. Data Flow Diagram
**Description:** Illustrates the movement of data during training (298A) and real-time production (298B).

```mermaid
flowchart TD
    subgraph TrainingPhase["Training Data Flow (298A)"]
        DS1["Github Slack Logs"] --> CL["Preprocess & Clean"]
        DS2["Enron Emails"] --> CL
        DS3["AMI Transcripts"] --> CL
        CL --> LAB["Labeling (Gemini 2.5)"]
        LAB --> FT["LoRA Fine-Tuning"]
        FT --> QM["Qwen-14B LoRA Weights"]
    end

    subgraph ProductionPhase["Real-Time Production Flow (298B)"]
        MS["Incoming Slack event"] --> API["FastAPI Endpoint"]
        API --> INF["Qwen-14B (Vertex AI)"]
        INF --> ACT["Execute Action"]
        ACT --> JIR["Jira API"]
        ACT --> GCL["Google Calendar"]
        ACT --> SLK["Slack Response"]
    end

    subgraph Storage["Persistence & Search"]
        HIST["Historical Tasks"] --> EMB["MiniLM Embedder"]
        EMB --> VDB["ChromaDB (Vector)"]
        VDB -.->|Search Context| INF
        SLK --> FEED["User Feedback"]
        FEED --> PSQL["Cloud SQL (RLHF)"]
    end

    style TrainingPhase fill:#fff7ed,stroke:#fb923c
    style ProductionPhase fill:#f0fdf4,stroke:#4ade80
```

---

## 4. Agents & Orchestration
**Description:** Detailed view of the multi-agent system powered by a single Qwen-14B model.

```mermaid
graph TB
    subgraph Model["Vertex AI Model Instance"]
        QWEN["Qwen-14B (LoRA Fine-tuned)"]
    end

    subgraph LangGraph["Orchestration Logic"]
        direction TB
        RTR["Router Agent"]
        
        subgraph Specialists["Specialist Agents"]
            SCH["Scheduler"]
            SUM["Summarizer"]
            TSK["Task Extractor"]
            BTN["Bottleneck Detector"]
        end

        CRT["Critic Agent"]
        JDG["Judge Agent"]

        RTR --> Specialists
        Specialists --> CRT
        CRT -->|Conflict| Specialists
        CRT -->|Consensus| JDG
    end

    subgraph Knowledge["Retrieval Layer"]
        VDB[("ChromaDB")]
        EMB["MiniLM-L6-v2"]
    end

    subgraph Tools["External Tools"]
        GC["Google Cal API"]
        GM["Gmail API"]
        JR["Jira API"]
    end

    RTR --- QWEN
    Specialists --- QWEN
    CRT --- QWEN
    JDG --- QWEN

    CRT -- Search --> EMB
    EMB -- Query --> VDB
    SCH --- GC
    SCH --- GM
    TSK --- JR
    BTN --- JR
    BTN --- GC

    style Model fill:#eff6ff,stroke:#3b82f6,stroke-dasharray: 5 5
    style LangGraph fill:#ffffff,stroke:#333
```

---

## 5. ML Pipeline & Fine-Tuning
**Description:** The lineage of the model from base weights to fine-tuned production state.

```mermaid
flowchart LR
    Base["Qwen-14B Base Model\n(HuggingFace)"]
    Data["Labeled Curated Data\n(Slack, Email, Meetings)"]
    
    subgraph FineTuning["Fine-Tuning Process"]
        PEFT["PEFT / LoRA"]
        TRAIN["Supervised Fine-Tuning\n(SFT)"]
    end

    Weights["Fine-tuned LoRA Adapters"]
    Deployment["Vertex AI Multi-Model Endpoint"]

    Base --> FineTuning
    Data --> FineTuning
    FineTuning --> Weights
    Weights --> Deployment
    
    subgraph Continuous["Continuous Learning Loop"]
        User["User Feedback"]
        Store["RLHF Preference Store"]
        DPO["DPO Tuning"]
    end

    Deployment --> User
    User --> Store
    Store --> DPO
    DPO --> Weights

    style FineTuning fill:#fefce8,stroke:#eab308
    style Continuous fill:#ecfdf5,stroke:#10b981
```

---

## 6. End-to-End Architecture
**Description:** Full system integration within the Google Cloud Platform ecosystem.

```mermaid
graph TB
    subgraph Users["User Layer"]
        SLK_CL["Slack User"]
        DASH["Web Dashboard"]
    end

    subgraph GCP["Google Cloud Platform"]
        subgraph Compute["Application Logic"]
            CR["Cloud Run (FastAPI)"]
            LG["LangGraph Orchestrator"]
            CR --- LG
        end

        subgraph AI["AI & Model Layer"]
            VAI["Vertex AI Service"]
            VAI --- QWEN["Qwen-14B (LoRA)"]
            EMB["MiniLM Service"]
        end

        subgraph Data["Persistence Layer"]
            SQL[("Cloud SQL - Postgre")]
            GCS[("Cloud Storage")]
            VDB[("ChromaDB / Vector")]
        end
    end

    subgraph Integrations["Integration Layer"]
        S_API["Slack API"]
        GC_API["Google Cal API"]
        GM_API["Gmail API"]
        JR_API["Jira API"]
    end

    Users <--> S_API
    S_API <--> CR
    CR <--> VAI
    VAI <--> Data
    CR <--> Integrations
    CR <--> SQL

    style GCP fill:#f8fafc,stroke:#64748b,stroke-width:2px
    style AI fill:#eff6ff,stroke:#3b82f6
    style Compute fill:#f0fdf4,stroke:#22c55e
```
