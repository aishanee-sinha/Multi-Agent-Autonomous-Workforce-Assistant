"""
slack_agent.py — Slack + Jira subgraph
=======================================
Nodes:
  slack_extract_ticket   — LLM extracts a Jira ticket from the Slack message
  slack_post_preview     — Posts proposed ticket card with Approve/Cancel buttons
  slack_resolve_assignee — Maps raw assignee string → Jira account ID via TEAM_MAP
  slack_create_jira      — Calls the Jira REST API to create the issue
  slack_post_result      — Updates the Slack message with the final outcome
  slack_post_cancel      — Updates the Slack message with a cancellation notice

Entry routing (route_slack_entry):
  - url_verification  → __end__  (handled at orchestrator level)
  - interactivity + create_jira → slack_resolve_assignee
  - interactivity + cancel_jira → slack_post_cancel
  - message           → slack_extract_ticket
"""

import json, logging

import requests
from requests.auth import HTTPBasicAuth
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import (
    OrchestratorState,
    JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN,
    JIRA_PROJECT_KEY, JIRA_ISSUE_TYPE,
    SLACK_BOT_TOKEN, TEAM_MAP,
    _llm,
)
from feedback_logger import log_feedback
from rag_retriever import get_similar_approved, format_as_few_shot

logger = logging.getLogger(__name__)

from functools import wraps
import logging

logger = logging.getLogger(__name__)

def _logged_slack(client: WebClient) -> WebClient:
    """Wrap every Slack API method to log the call and args before executing."""
    original_api_call = client.api_call

    @wraps(original_api_call)
    def logged_api_call(api_method, *args, **kwargs):
        # Log the method name and all arguments
        safe_kwargs = {
            k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
            for k, v in kwargs.items()
            if k not in ("token",)  # never log the token
        }
        logger.info(f"SLACK API CALL  : {api_method}")
        logger.info(f"SLACK API ARGS  : {safe_kwargs}")
        result = original_api_call(api_method, *args, **kwargs)
        logger.info(f"SLACK API RESULT: ok={result.get('ok')} error={result.get('error','none')}")
        return result

    client.api_call = logged_api_call
    return client
# ─────────────────────────────────────────────────────────────────────────────
# Structured output schema
# ─────────────────────────────────────────────────────────────────────────────
class JiraTicket(BaseModel):
    task_summary: str = Field(description="Brief summary of the task")
    assignee: str = Field(default="Unassigned", description="Slack User ID or name mentioned")
    no_action: bool = Field(default=False, description="True if no actionable task found")


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def slack_extract_ticket(state: OrchestratorState) -> OrchestratorState:
    """LLM extracts a structured Jira ticket from the Slack message text."""
    # ── RAG: retrieve similar approved examples ─────────────────────────
    examples = get_similar_approved("slack", state.user_text, n_results=2)
    few_shot = format_as_few_shot(examples)

    sys_msg = (
        "You are a Jira assistant. Extract task details from the user's Slack message. "
        "For 'assignee', return the Slack User ID or name. "
        "Set no_action=True if the message contains no actionable task."
    )
    if few_shot:
        sys_msg += "\n" + few_shot

    llm = _llm(structured_output=JiraTicket, model_name="slack")
    try:
        human_msg = state.user_text
        ticket: JiraTicket = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=human_msg)])

        # ── Save prompt+response for feedback logging later ─────────────
        prompt_log = json.dumps([{"role": "system", "content": sys_msg},
                                  {"role": "user", "content": human_msg}])
        response_log = json.dumps(ticket.dict())

        return state.model_copy(update={
            "slack_ticket_summary": ticket.task_summary,
            "slack_ticket_assignee": ticket.assignee,
            "slack_no_action": ticket.no_action,
            "llm_prompt_log": prompt_log,
            "llm_response_log": response_log,
        })
    except Exception as e:
        logger.error(f"Slack extract error: {e}")
        return state.model_copy(update={"error": str(e)})


def slack_post_preview(state: OrchestratorState) -> OrchestratorState:
    """Post the proposed ticket card with Approve / Cancel buttons to Slack."""
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"🎫 *Proposed Jira Task*\n"
                    f"*Summary:* {state.slack_ticket_summary}\n"
                    f"*Assignee:* <@{state.slack_ticket_assignee}>"
                ),
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ Create Ticket"},
                    "style": "primary",
                    "value": json.dumps({"s": state.slack_ticket_summary, "a": state.slack_ticket_assignee}),
                    "action_id": "create_jira",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "❌ Cancel"},
                    "style": "danger",
                    "action_id": "cancel_jira",
                },
            ],
        },
    ]
    try:
        resp = client.chat_postMessage(
            channel=state.channel_id,
            blocks=blocks,
            thread_ts=state.message_ts,
            text=f"Proposed Jira Task: {state.slack_ticket_summary}",
        )
        return state.model_copy(update={"preview_ts": resp["ts"]})
    except SlackApiError as e:
        return state.model_copy(update={"error": str(e)})


def slack_resolve_assignee(state: OrchestratorState) -> OrchestratorState:
    """Map the raw assignee string → Jira account ID using TEAM_MAP."""
    raw = (state.slack_action_value or {}).get("a", state.slack_ticket_assignee or "")
    logger.info(f"Resolving assignee: raw='{raw}'")
    jira_id = TEAM_MAP.get(raw)
    if not jira_id:
        for name, acc_id in TEAM_MAP.items():
            if name.lower() in raw.lower():
                jira_id = acc_id
                break
    return state.model_copy(update={"jira_account_id": jira_id})


def slack_create_jira(state: OrchestratorState) -> OrchestratorState:
    """Call the Jira REST API to create the issue."""
    summary = (state.slack_action_value or {}).get("s", state.slack_ticket_summary or "")
    fields: dict = {
        "project": {"key": JIRA_PROJECT_KEY},
        "summary": summary,
        "description": {
            "type": "doc", "version": 1,
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Created via Slack Agent"}]}],
        },
        "issuetype": {"name": JIRA_ISSUE_TYPE},
    }
    if state.jira_account_id:
        fields["assignee"] = {"id": state.jira_account_id}

    try:
        jira_endpoint = f"{JIRA_BASE_URL}/rest/api/3/issue"
        safe_summary = (summary[:120] + "...") if len(summary) > 120 else summary
        logger.info(
            "Jira create request: endpoint=%s project=%s issue_type=%s assignee_id=%s summary=%s",
            jira_endpoint,
            JIRA_PROJECT_KEY,
            JIRA_ISSUE_TYPE,
            state.jira_account_id or "none",
            safe_summary,
        )

        resp = requests.post(
            jira_endpoint,
            data=json.dumps({"fields": fields}),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
            timeout=30,
        )
        logger.info("Jira create response status=%s", resp.status_code)
        try:
            result = resp.json()
        except ValueError:
            result = {"raw_body": resp.text}

        logger.info("Jira create response body=%s", json.dumps(result, ensure_ascii=True))

        if not resp.ok:
            error_messages = result.get("errorMessages", []) if isinstance(result, dict) else []
            error_fields = result.get("errors", {}) if isinstance(result, dict) else {}
            logger.error(
                "Jira create failed: status=%s errorMessages=%s errors=%s",
                resp.status_code,
                error_messages,
                error_fields,
            )
            if any("permission" in msg.lower() for msg in error_messages if isinstance(msg, str)):
                logger.error(
                    "Jira permission issue detected for project=%s user=%s",
                    JIRA_PROJECT_KEY,
                    JIRA_EMAIL,
                )

        if "key" in result:
            return state.model_copy(update={"jira_key": result["key"]})
        return state.model_copy(update={"error": json.dumps({"status": resp.status_code, "body": result})})
    except Exception as e:
        logger.exception("Jira create exception")
        return state.model_copy(update={"error": str(e)})


def slack_post_result(state: OrchestratorState) -> OrchestratorState:
    """Update the Slack preview message with the final outcome."""
    # ── RLHF: log approved feedback ─────────────────────────────────────
    if state.jira_key and state.llm_prompt_log:
        try:
            log_feedback(
                flow="slack",
                prompt=state.llm_prompt_log,
                response=state.llm_response_log or "",
                human_decision="approved",
                metadata={"channel": state.channel_id, "jira_key": state.jira_key},
            )
        except Exception as e:
            logger.warning("RLHF feedback log failed (approved): %s", e)

    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))
    msg = (
        f"✅ *Ticket Created:* <{JIRA_BASE_URL}/browse/{state.jira_key}|{state.jira_key}>"
        if state.jira_key
        else f"⚠️ Failed: {state.error or 'Unknown error'}"
    )
    try:
        client.chat_update(channel=state.channel_id, ts=state.preview_ts, text=msg, blocks=None)
    except SlackApiError as e:
        logger.error(f"Slack update error: {e}")
    return state


def slack_post_cancel(state: OrchestratorState) -> OrchestratorState:
    """Update the Slack message to show cancellation."""
    # ── RLHF: log cancelled feedback ────────────────────────────────────
    if state.llm_prompt_log:
        try:
            log_feedback(
                flow="slack",
                prompt=state.llm_prompt_log,
                response=state.llm_response_log or "",
                human_decision="cancelled",
                metadata={"channel": state.channel_id},
            )
        except Exception as e:
            logger.warning("RLHF feedback log failed (cancelled): %s", e)

    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))
    try:
        client.chat_update(
            channel=state.channel_id,
            ts=state.preview_ts,
            text="❌ Ticket creation cancelled.",
            blocks=None,
        )
    except SlackApiError as e:
        logger.error(f"Slack cancel error: {e}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────
def route_slack_entry(state: OrchestratorState) -> Literal[
    "slack_extract_ticket", "slack_resolve_assignee", "slack_post_cancel", "__end__"
]:
    if state.slack_event_type == "url_verification":
        return "__end__"
    if state.slack_event_type == "interactivity":
        if state.slack_action_id == "create_jira":
            return "slack_resolve_assignee"
        return "slack_post_cancel"
    return "slack_extract_ticket"


def route_after_extract(state: OrchestratorState) -> Literal["slack_post_preview", "__end__"]:
    if state.error or state.slack_no_action:
        return "__end__"
    return "slack_post_preview"


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_slack_subgraph() -> StateGraph:
    b = StateGraph(OrchestratorState)
    b.add_node("slack_extract_ticket",   slack_extract_ticket)
    b.add_node("slack_post_preview",     slack_post_preview)
    b.add_node("slack_resolve_assignee", slack_resolve_assignee)
    b.add_node("slack_create_jira",      slack_create_jira)
    b.add_node("slack_post_result",      slack_post_result)
    b.add_node("slack_post_cancel",      slack_post_cancel)

    b.add_conditional_edges(START, route_slack_entry, {
        "slack_extract_ticket":   "slack_extract_ticket",
        "slack_resolve_assignee": "slack_resolve_assignee",
        "slack_post_cancel":      "slack_post_cancel",
        "__end__":                END,
    })
    b.add_conditional_edges("slack_extract_ticket", route_after_extract, {
        "slack_post_preview": "slack_post_preview",
        "__end__":            END,
    })
    b.add_edge("slack_post_preview",     END)
    b.add_edge("slack_resolve_assignee", "slack_create_jira")
    b.add_edge("slack_create_jira",      "slack_post_result")
    b.add_edge("slack_post_result",      END)
    b.add_edge("slack_post_cancel",      END)
    return b.compile()
