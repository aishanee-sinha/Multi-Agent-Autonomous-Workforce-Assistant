"""
state.py — Shared state, config, and LLM factory
Used by orchestrator.py, slack_agent.py, calendar_agent.py, and meeting_agent.py

Changes vs peers' original:
  - intent Literal extended with "meeting_transcript"
  - slack_action_id comment updated with confirm_summary / cancel_summary
  - Meeting summarizer env vars added (bottom of config block)
  - "meeting" added to _llm() valid model names
  - Meeting transcript fields added to OrchestratorState
"""

import os, json, logging, operator
from typing import Optional, Literal, Annotated

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — peers' original vars
# ─────────────────────────────────────────────────────────────────────────────
JIRA_BASE_URL    = os.environ.get("JIRA_BASE_URL")
JIRA_EMAIL       = os.environ.get("JIRA_EMAIL")
JIRA_API_TOKEN   = os.environ.get("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "KAN")
JIRA_ISSUE_TYPE  = os.environ.get("JIRA_ISSUE_TYPE", "Task")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN")
SLACK_NOTIFY_CHANNEL = os.environ.get("SLACK_NOTIFY_CHANNEL")
EC2_IP           = os.environ.get("EC2_IP")
TEAM_MAP: dict   = json.loads(os.environ.get("TEAM_MAP_JSON", "{}"))

GROUP_EMAILS = json.loads(os.environ.get("GROUP_EMAILS_JSON", "[]"))
GOOGLE_TOKEN = os.environ.get("GOOGLE_TOKEN_JSON", "")
CALENDAR_TOKENS: dict = json.loads(os.environ.get("CALENDAR_TOKENS_JSON", "{}"))

# ── Redis session store ────────────────────────────────────────────────────────
REDIS_URL           = os.environ.get("REDIS_URL", "redis://localhost:6379")
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))

# ── Meeting Summarizer env vars (added) ───────────────────────────────────────
SES_FROM_EMAIL     = os.environ.get("SES_FROM_EMAIL", "")
PARTICIPANT_EMAILS = [
    e.strip() for e in os.environ.get("PARTICIPANT_EMAILS", "").split(",") if e.strip()
]
# Email agent inbox — receives meeting summaries for conflict detection + participant forwarding
EMAIL_AGENT_INBOX  = os.environ.get("EMAIL_AGENT_INBOX", "msadi.finalproject@gmail.com")
S3_BUCKET          = os.environ.get("S3_BUCKET", "qwen-lora-weights")
S3_PREFIX          = os.environ.get("S3_TRANSCRIPT_PREFIX", "transcript_summarizer")
VLLM_MODEL_NAME    = os.environ.get("VLLM_MODEL_NAME", "meeting")
WEBHOOK_SECRET     = os.environ.get("WEBHOOK_SECRET", "meeting-summarizer-secret-2026")
GOOGLE_DRIVE_FOLDER_ID = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM factory
# "meeting" added alongside peers' "slack" and "email" adapters
# ─────────────────────────────────────────────────────────────────────────────
def _llm(structured_output=None, model_name=None) -> ChatOpenAI:
    assert model_name in [
        "Qwen/Qwen2.5-14B-Instruct-AWQ", "slack", "email", "meeting"
    ], f"Invalid model name: {model_name}"

    base = ChatOpenAI(
        model=model_name,
        openai_api_base=f"http://{EC2_IP}:8000/v1",
        openai_api_key="none",
        timeout=120,
        max_retries=0,
    )
    return base.with_structured_output(structured_output) if structured_output else base


# ─────────────────────────────────────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────────────────────────────────────
class OrchestratorState(BaseModel):
    """
    Master state that flows through the entire graph.
    All three subgraphs (slack, calendar, meeting) read/write to this object.
    """
    # ── Raw input ──────────────────────────────────────────────────────────
    raw_event: dict = Field(default_factory=dict)

    # ── Router decision ────────────────────────────────────────────────────
    # "meeting_transcript" added for meeting summarizer trigger + button clicks
    intent: Literal["slack", "email", "none", "unknown", "meeting_transcript"] = "unknown"
    intent_reason: Optional[str] = None

    # ── Slack-specific (Jira flow) ─────────────────────────────────────────
    slack_event_type: Literal["message", "interactivity", "url_verification", "unknown"] = "unknown"
    channel_id: Optional[str] = None
    user_text: Optional[str] = None
    message_ts: Optional[str] = None
    preview_ts: Optional[str] = None
    is_retry: bool = False
    is_bot: bool = False
    slack_ticket_summary: Optional[str] = None
    slack_ticket_assignee: Optional[str] = None
    slack_no_action: bool = False
    # action_ids: create_jira | cancel_jira | create_meeting | cancel_meeting
    #             confirm_summary | cancel_summary  ← added for meeting summarizer
    slack_action_id: Optional[str] = None
    slack_action_value: Optional[dict] = None
    jira_account_id: Optional[str] = None
    jira_key: Optional[str] = None

    # ── Email/Calendar-specific ────────────────────────────────────────────
    email_source: Literal["pubsub", "direct", "unknown"] = "unknown"
    email_data: Optional[dict] = None
    is_meeting: bool = False
    meeting_title: Optional[str] = None
    meeting_start: Optional[str] = None
    meeting_end: Optional[str] = None
    meeting_location: Optional[str] = None
    meeting_attendees: list[str] = Field(default_factory=list)
    time_confidence: Optional[str] = None
    calendar_link: Optional[str] = None
    pending_meeting: Optional[dict] = None

    # ── CoD top-3 slots — set by slot_cod, consumed by email_post_slack_preview ─
    cod_top_slots: list[dict] = Field(default_factory=list)

    # ── Selected slot — set by parse_input when a slot button is clicked ──────
    selected_slot: Optional[dict] = None

    # ── Redis session IDs — carried from parse_input to create/cancel nodes ───
    # For email flow: comma-joined string of up to 3 session IDs (one per slot)
    session_id: Optional[str] = None

    # ── Meeting Transcript-specific (added) ───────────────────────────────
    transcript_file_id:   Optional[str]   = None  # Google Drive file ID
    transcript_file_name: Optional[str]   = None  # e.g. "demo_meeting.txt"
    transcript_text:      Optional[str]   = None  # raw downloaded text
    transcript_processed: Optional[str]   = None  # after preprocessor
    meeting_summary_parsed: Optional[dict] = None  # parsed model output dict
    meeting_triage:       Optional[list]  = None  # CoD triage results per action item
    meeting_s3_key:       Optional[str]   = None  # S3 prefix for all artifacts
    meeting_ics_bytes:    Optional[bytes] = None  # ICS calendar bytes
    meeting_csv_bytes:    Optional[bytes] = None  # CSV action items bytes
    meeting_jira_proposals:     Optional[list] = None  # CoD-detected Jira tickets to propose
    meeting_jira_queue_session: Optional[str]  = None  # Redis session ID for pending Jira proposal queue

    # ── Shared control ─────────────────────────────────────────────────────
    error: Optional[str] = None
    messages: Annotated[list[BaseMessage], operator.add] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True