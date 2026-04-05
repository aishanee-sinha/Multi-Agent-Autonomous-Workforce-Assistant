"""
state.py — Shared state, config, and LLM factory
Used by orchestrator.py, slack_agent.py, and calendar_agent.py
"""

import os, json, logging, operator
from typing import Optional, Literal, Annotated

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
JIRA_BASE_URL    = os.environ.get("JIRA_BASE_URL")
JIRA_EMAIL       = os.environ.get("JIRA_EMAIL")
JIRA_API_TOKEN   = os.environ.get("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "KAN")
JIRA_ISSUE_TYPE  = os.environ.get("JIRA_ISSUE_TYPE", "Task")
SLACK_BOT_TOKEN  = os.environ.get("SLACK_BOT_TOKEN")
SLACK_NOTIFY_CHANNEL = os.environ.get("SLACK_NOTIFY_CHANNEL")  # channel to post meeting previews
EC2_IP           = os.environ.get("EC2_IP")
TEAM_MAP: dict   = json.loads(os.environ.get("TEAM_MAP_JSON", "{}"))

GROUP_EMAILS = json.loads(os.environ.get("GROUP_EMAILS_JSON", "[]"))
GOOGLE_TOKEN = os.environ.get("GOOGLE_TOKEN_JSON", "")

# Maps email address → token filename stem
# e.g. {"msadi.finalproject@gmail.com": "Agent", "aishanee.sinha@sjsu.edu": "Aishanee"}
# Maps email address → refresh_token string
# e.g. {"aishanee.sinha@sjsu.edu": "1//0abc...", "sohan.juetce@gmail.com": "1//0def..."}
CALENDAR_TOKENS: dict = json.loads(os.environ.get("CALENDAR_TOKENS_JSON", "{}"))


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM factory
# ─────────────────────────────────────────────────────────────────────────────
def _llm(structured_output=None, model_name=None) -> ChatOpenAI:
    assert model_name in ["Qwen/Qwen2.5-14B-Instruct-AWQ", "slack", "email"], f"Invalid model name: {model_name}"

    base = ChatOpenAI(
        model=model_name,
        openai_api_base=f"http://{EC2_IP}:8000/v1",
        openai_api_key="none",
        timeout=45,
        max_retries=0,
    )
    return base.with_structured_output(structured_output) if structured_output else base


# ─────────────────────────────────────────────────────────────────────────────
# Shared State
# ─────────────────────────────────────────────────────────────────────────────
class OrchestratorState(BaseModel):
    """
    Master state that flows through the entire graph.
    Both subgraphs read/write to this same object.
    """
    # ── Raw input ──────────────────────────────────────────────────────────
    raw_event: dict = Field(default_factory=dict)

    # ── Router decision ────────────────────────────────────────────────────
    intent: Literal["slack", "email", "none", "unknown"] = "unknown"
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
    slack_action_id: Optional[str] = None   # "create_jira" | "cancel_jira" | "create_meeting" | "cancel_meeting"
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

    # ── Pending meeting — passed via Slack button value, no DynamoDB ──────
    pending_meeting: Optional[dict] = None

    # ── Shared control ─────────────────────────────────────────────────────
    error: Optional[str] = None
    messages: Annotated[list[BaseMessage], operator.add] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
