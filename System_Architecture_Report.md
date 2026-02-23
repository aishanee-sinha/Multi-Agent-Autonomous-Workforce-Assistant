# 5. System Architecture Design

## 5.1 System Requirements Analysis

### 5.1.1 System Boundary, Actors, and Use Cases

#### System Boundary
The **AI-Powered Autonomous Workforce Assistant** defines its boundary between the enterprise communication and productivity ecosystem (Slack, Gmail, Google Calendar, Jira) and the internal AI orchestration engine. 
- **Inside the Boundary:** The multi-agent orchestration layer (LangGraph), the fine-tuned LLM backbone (Qwen-14B), the vector database (ChromaDB), the application logic (FastAPI), and the feedback storage (Cloud SQL).
- **Outside the Boundary:** External SaaS platforms (Slack API, Jira API, Google Workspace), and the underlying cloud infrastructure (GCP Vertex AI, Cloud Run).

#### Actors
1.  **Team Member (End User):** The primary consumer of AI services. Interacts via Slack to initiate scheduling, request summaries, or confirm task extractions.
2.  **Project Manager (PM):** Uses the system to monitor team health, receive bottleneck alerts, and visualize project velocity via the dashboard.
3.  **System Administrator:** Responsible for system configuration, managing API credentials, and monitoring the health of the ML pipeline.
4.  **External Systems (Machine Actors):** Slack (Trigger/Interface), Jira (Task Store), Google Calendar (Schedule Storage), and Gmail (Data Source).

#### Use Cases
- **UC-1: Automated Meeting Scheduling:** Detects scheduling intent, checks participant availability via Google Calendar API, and generates .ics invites.
- **UC-2: Conversation & Document Summarization:** Provides abstractive summaries of Slack threads or transcripts, highlighting key decisions and action items.
- **UC-3: Automated Task Extraction to Jira:** Identifies tasks in Slack messages and converts them into structured Jira issues with assignees and priorities.
- **UC-4: Workflow Bottleneck Detection:** Periodically analyzes Jira status transitions and team workloads to flag stuck tasks and over-allocated members.
- **UC-5: Chain of Debate (CoD) Orchestration:** An internal multi-agent process where "Critic" and "Judge" agents refine the proposals of specialist agents before delivery.
- **UC-6: Learning from Feedback (RLHF):** Captures user corrections (Accept/Edit/Dismiss) to create preference datasets for model alignment.

---

### 5.1.2 High-Level Data Analytics and Machine Learning Functions and Capabilities

#### Machine Learning Functions
- **Intent Classification (Routing):** A sequence-to-sequence task mapping raw input to specific agent tool-chains (e.g., "Summarize" mapping to the Summarizer Agent).
- **LoRA-Augmented Text Generation:** Utilizes fine-tuned Low-Rank Adapters (LoRA) on the Qwen-14B base model for high-accuracy domain-specific extraction and reasoning.
- **Named Entity Recognition (NER):** Specialized prompts to extract temporal expressions (dates/times), person names, and task descriptions from unstructured text.
- **Reflective Critique (CoD):** A multi-turn reasoning process where the model critiques its own initial drafts based on historical context and project constraints.

#### Data Analytics Capabilities
- **Semantic Retrieval (RAG):** Uses the `all-MiniLM-L6-v2` embedding model to transform historical project data into a vector space, enabling similarity-based search for the Critic agent.
- **Task Anomaly Detection:** Analyzing Jira "Time-in-Status" metrics to identify statistical outliers representing potential workflow bottlenecks.
- **Preference Analytics:** Analyzing RLHF feedback logs to track model performance trends and user alignment scores over time.

---

## 5.2 System Design

### 5.2.1 System Architecture and Infrastructure with AI-Powered Function Components

The system employs a **Layered Multi-Agent Architecture** orchestrated via **LangGraph**. The design consolidates multiple domain-specific agents into a single unified backbone.

1.  **Orchestration Layer:** Built on **LangGraph**, it manages the state of the conversation and the "Chain of Debate" cycle. It uses a **Router Agent** to direct requests and a **Judge Agent** to finalize outputs.
2.  **AI Function Layer:** Powered by **Qwen-14B (LoRA fine-tuned)**. This model wears multiple "hats" by switching context-specific system prompts. It performs the core reasoning for scheduling, summarization, and task extraction.
3.  **Infrastructure:** 
    - **Vertex AI:** Hosts the Qwen-14B endpoint for scalable, low-latency GPU inference.
    - **Cloud Run:** Hosts the FastAPI-based application logic in a serverless, auto-scaling container environment.

---

### 5.2.2 Integration with Supporting Platforms, Frameworks, and Cloud Environment

The system is natively integrated into the **Google Cloud Platform (GCP)** and connects to the enterprise stack via secure API protocols.

- **Platform Integration:**
    - **Slack:** Communicates via Socket Mode/Webhooks for real-time bidirectional messaging.
    - **Google Workspace:** Uses OAuth 2.0 to securely access Gmail and Google Calendar APIs.
    - **Jira:** Utilizes the Jira REST API for automated CRUD operations on project issues.
- **Framework Integration:** Uses **LangChain/LangGraph** for multi-agent state management and **bitsandbytes/PEFT** for efficient LoRA adapter management.
- **Cloud Ecosystem:** Leverages **Cloud Storage (GCS)** for dataset persistence and **Artifact Registry** for container image management.

---

### 5.2.3 Data Management Solution and Database

The system uses a **Hybrid Database Strategy** to handle both relational state and semantic context.

1.  **Vector Management (ChromaDB):**
    - Stores high-dimensional embeddings of 100K+ Slack messages and Jira tickets.
    - Enables the **Critic Agent** to perform context-aware reviews by retrieving similar past project scenarios.
2.  **Relational Management (Cloud SQL - PostgreSQL):**
    - Acts as the system of record for **RLHF Logs** (storing original outputs vs. user-edited versions).
    - Manages user identity mappings between Slack IDs, Emails, and Jira usernames.
3.  **Data Lifecycle:** Raw data from GitHub/Kaggle is preprocessed, labeled via Gemini Flash 2.5, and stored in GCS before being used for LoRA fine-tuning and vector indexing.

---

### 5.2.4 User Interface and Data Visualization

1.  **Conversational UI (Slack Bot):**
    - The primary interface for all actors. It utilizes **Slack Block Kit** for rich interaction, providing interactive buttons ([Accept], [Edit], [Dismiss]) that simplify human-in-the-loop validation.
    - Proactively sends "Bottleneck Alerts" and "Digest Summaries" directly to users.
2.  **Web Dashboard (Visualization):**
    - A secondary interface for Project Managers to visualize system performance.
    - **Key Visuals:** Latency gauges, task extraction funnel (showing detection vs. creation rates), and team velocity heatmaps.
    - Displays **G-Eval scores** (Consistency, Fluency, Relevance) to quantify the quality improvement achieved through the Chain of Debate.
