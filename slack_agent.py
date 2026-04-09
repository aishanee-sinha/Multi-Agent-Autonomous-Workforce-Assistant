"""
slack_agent.py — Slack + Jira subgraph
=======================================
Nodes:
  slack_extract_ticket   — LLM extracts a Jira action from the Slack message
  slack_post_preview     — Posts proposed Jira action with Approve/Cancel buttons
  slack_resolve_assignee — Maps raw assignee string → Jira account ID via TEAM_MAP
  slack_create_jira      — Calls the Jira REST API to create the issue
  slack_update_jira      — Updates an existing Jira issue
  slack_close_jira       — Closes an existing Jira issue via transitions API
  slack_post_result      — Updates the Slack message with the final outcome
  slack_post_cancel      — Updates the Slack message with a cancellation notice

Entry routing (route_slack_entry):
  - url_verification  → __end__
  - interactivity + create_jira → slack_resolve_assignee
  - interactivity + cancel_jira → slack_post_cancel
  - message           → slack_extract_ticket
"""

import json
import logging
from functools import wraps
from typing import Literal

import requests
from requests.auth import HTTPBasicAuth
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import (
    OrchestratorState,
    JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN,
    JIRA_PROJECT_KEY, JIRA_ISSUE_TYPE,
    SLACK_BOT_TOKEN, TEAM_MAP,
    _llm,
)

logger = logging.getLogger(__name__)


def _logged_slack(client: WebClient) -> WebClient:
    """Wrap every Slack API method to log the call and args before executing."""
    original_api_call = client.api_call

    @wraps(original_api_call)
    def logged_api_call(api_method, *args, **kwargs):
        safe_kwargs = {
            k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
            for k, v in kwargs.items()
            if k not in ("token",)
        }
        logger.info(f"SLACK API CALL  : {api_method}")
        logger.info(f"SLACK API ARGS  : {safe_kwargs}")
        result = original_api_call(api_method, *args, **kwargs)
        logger.info(f"SLACK API RESULT: ok={result.get('ok')} error={result.get('error', 'none')}")
        return result

    client.api_call = logged_api_call
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Structured output schema
# ─────────────────────────────────────────────────────────────────────────────
class JiraTicket(BaseModel):
    action: Literal["create", "update", "close", "no_action"] = "no_action"
    task_summary: str | None = Field(default=None, description="Brief summary of the task")
    assignee: str | None = Field(default="Unassigned", description="Slack User ID or name mentioned")
    issue_key: str | None = Field(default=None, description="Existing Jira issue key like KAN-123")
    status: str | None = Field(default=None, description="Updated status if mentioned")
    comment: str | None = Field(default=None, description="Comment or update note if mentioned")
    no_action: bool = Field(default=False, description="True if no actionable task found")


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def slack_extract_ticket(state: OrchestratorState) -> OrchestratorState:
    """LLM extracts a structured Jira action from the Slack message text."""
    sys_msg = """
You are a Jira assistant.

Classify the user's Slack message into one of these actions:
- create: create a new Jira ticket
- update: update an existing Jira ticket
- close: close or resolve an existing Jira ticket
- no_action: no Jira action needed

Rules:
- If the message contains an issue key like KAN-123, prefer update or close instead of create.
- If the message says close, resolve, done, completed, fixed, mark done -> action = close
- If the message says update, change, modify, assign, add comment -> action = update
- If the message describes a new issue/task without an issue key -> action = create
- If nothing actionable is present -> action = no_action and no_action = true

Extract:
- action
- task_summary
- assignee
- issue_key
- status
- comment
- no_action
"""
    llm = _llm(structured_output=JiraTicket, model_name="slack")
    try:
        ticket: JiraTicket = llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=state.user_text)
        ])
        return state.model_copy(update={
            "slack_ticket_summary": ticket.task_summary,
            "slack_ticket_assignee": ticket.assignee,
            "slack_no_action": ticket.no_action,
            "slack_action_type": ticket.action,
            "slack_issue_key": ticket.issue_key,
            "slack_update_status": ticket.status,
            "slack_comment": ticket.comment,
        })
    except Exception as e:
        logger.error(f"Slack extract error: {e}")
        return state.model_copy(update={"error": str(e)})


def slack_post_preview(state: OrchestratorState) -> OrchestratorState:
    """Post the proposed Jira action card with Approve / Cancel buttons to Slack."""
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))

    action_label = (state.slack_action_type or "create").replace("_", " ").title()

    details = [
        f"🎫 *Proposed Jira Action*",
        f"*Action:* {action_label}",
        f"*Summary:* {state.slack_ticket_summary or 'N/A'}",
        f"*Issue Key:* {state.slack_issue_key or 'New Ticket'}",
        f"*Assignee:* {state.slack_ticket_assignee or 'Unassigned'}",
    ]

    if state.slack_update_status:
        details.append(f"*Status:* {state.slack_update_status}")
    if state.slack_comment:
        details.append(f"*Comment:* {state.slack_comment}")

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(details),
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": f"✅ Confirm {action_label}"},
                    "style": "primary",
                    "value": json.dumps({
                        "s": state.slack_ticket_summary,
                        "a": state.slack_ticket_assignee,
                        "issue_key": state.slack_issue_key,
                        "status": state.slack_update_status,
                        "comment": state.slack_comment,
                        "action": state.slack_action_type,
                    }),
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
            text=f"Proposed Jira Action: {action_label}",
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
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": "Created via Slack Agent"}]
                }
            ],
        },
        "issuetype": {"name": JIRA_ISSUE_TYPE},
    }

    if state.jira_account_id:
        fields["assignee"] = {"id": state.jira_account_id}

    try:
        jira_endpoint = f"{JIRA_BASE_URL}/rest/api/3/issue"
        resp = requests.post(
            jira_endpoint,
            json={"fields": fields},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
            timeout=30,
        )

        try:
            result = resp.json()
        except ValueError:
            result = {"raw_body": resp.text}

        if "key" in result:
            return state.model_copy(update={"jira_key": result["key"]})

        return state.model_copy(update={"error": json.dumps({"status": resp.status_code, "body": result})})

    except Exception as e:
        logger.exception("Jira create exception")
        return state.model_copy(update={"error": str(e)})


def slack_update_jira(state: OrchestratorState) -> OrchestratorState:
    """Update an existing Jira issue."""
    issue_key = state.slack_issue_key or (state.slack_action_value or {}).get("issue_key")
    summary = state.slack_ticket_summary or (state.slack_action_value or {}).get("s")
    status = state.slack_update_status or (state.slack_action_value or {}).get("status")
    comment = state.slack_comment or (state.slack_action_value or {}).get("comment")

    if not issue_key:
        return state.model_copy(update={"error": "Missing issue key for update"})

    try:
        # 1) Update standard editable fields
        fields = {}
        if summary:
            fields["summary"] = summary

        if fields:
            update_resp = requests.put(
                f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}",
                json={"fields": fields},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
                timeout=30,
            )

            if update_resp.status_code not in (200, 204):
                try:
                    err_body = update_resp.json()
                except ValueError:
                    err_body = update_resp.text
                return state.model_copy(update={
                    "error": f"Jira update failed ({update_resp.status_code}): {err_body}"
                })

        # 2) Add comment if provided
        if comment:
            comment_payload = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": comment}]
                        }
                    ],
                }
            }

            comment_resp = requests.post(
                f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/comment",
                json=comment_payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
                timeout=30,
            )

            if comment_resp.status_code not in (200, 201):
                try:
                    err_body = comment_resp.json()
                except ValueError:
                    err_body = comment_resp.text
                return state.model_copy(update={
                    "error": f"Jira comment failed ({comment_resp.status_code}): {err_body}"
                })

        # 3) Transition status if provided
        if status:
            transitions_resp = requests.get(
                f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/transitions",
                headers={"Accept": "application/json"},
                auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
                timeout=30,
            )

            if transitions_resp.status_code != 200:
                try:
                    err_body = transitions_resp.json()
                except ValueError:
                    err_body = transitions_resp.text
                return state.model_copy(update={
                    "error": f"Jira transitions fetch failed ({transitions_resp.status_code}): {err_body}"
                })

            transitions = transitions_resp.json().get("transitions", [])
            matched = next(
                (t for t in transitions if t.get("name", "").strip().lower() == status.strip().lower()),
                None
            )

            if not matched:
                available = [t.get("name") for t in transitions]
                return state.model_copy(update={
                    "error": f"No Jira transition found for status '{status}'. Available: {available}"
                })

            transition_resp = requests.post(
                f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/transitions",
                json={"transition": {"id": matched["id"]}},
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
                timeout=30,
            )

            if transition_resp.status_code not in (200, 204):
                try:
                    err_body = transition_resp.json()
                except ValueError:
                    err_body = transition_resp.text
                return state.model_copy(update={
                    "error": f"Jira transition failed ({transition_resp.status_code}): {err_body}"
                })

        return state

    except Exception as e:
        logger.exception("Jira update exception")
        return state.model_copy(update={"error": str(e)})


def slack_close_jira(state: OrchestratorState) -> OrchestratorState:
    """Close an existing Jira issue using workflow transitions."""
    issue_key = state.slack_issue_key or (state.slack_action_value or {}).get("issue_key")
    comment = state.slack_comment or (state.slack_action_value or {}).get("comment")
    target_status = (
        state.slack_update_status
        or (state.slack_action_value or {}).get("status")
        or "Done"
    )

    if not issue_key:
        return state.model_copy(update={"error": "Missing issue key for close"})

    try:
        # 1) Add optional comment before closing
        if comment:
            comment_payload = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": comment}]
                        }
                    ],
                }
            }

            comment_resp = requests.post(
                f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/comment",
                json=comment_payload,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
                timeout=30,
            )

            if comment_resp.status_code not in (200, 201):
                try:
                    err_body = comment_resp.json()
                except ValueError:
                    err_body = comment_resp.text
                return state.model_copy(update={
                    "error": f"Jira close comment failed ({comment_resp.status_code}): {err_body}"
                })

        # 2) Fetch available transitions
        transitions_resp = requests.get(
            f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/transitions",
            headers={"Accept": "application/json"},
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
            timeout=30,
        )

        if transitions_resp.status_code != 200:
            try:
                err_body = transitions_resp.json()
            except ValueError:
                err_body = transitions_resp.text
            return state.model_copy(update={
                "error": f"Jira transitions fetch failed ({transitions_resp.status_code}): {err_body}"
            })

        transitions = transitions_resp.json().get("transitions", [])

        preferred_names = [
            target_status,
            "Done",
            "Closed",
            "Resolved",
            "Resolve Issue",
        ]

        matched = None
        for name in preferred_names:
            matched = next(
                (t for t in transitions if t.get("name", "").strip().lower() == name.strip().lower()),
                None
            )
            if matched:
                break

        if not matched:
            available = [t.get("name") for t in transitions]
            return state.model_copy(update={
                "error": f"No close transition found for issue {issue_key}. Available: {available}"
            })

        # 3) Perform transition
        transition_resp = requests.post(
            f"{JIRA_BASE_URL}/rest/api/3/issue/{issue_key}/transitions",
            json={"transition": {"id": matched["id"]}},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
            timeout=30,
        )

        if transition_resp.status_code not in (200, 204):
            try:
                err_body = transition_resp.json()
            except ValueError:
                err_body = transition_resp.text
            return state.model_copy(update={
                "error": f"Jira close transition failed ({transition_resp.status_code}): {err_body}"
            })

        return state

    except Exception as e:
        logger.exception("Jira close exception")
        return state.model_copy(update={"error": str(e)})


def slack_post_result(state: OrchestratorState) -> OrchestratorState:
    """Update the Slack preview message with the final outcome."""
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))

    issue_key = state.slack_issue_key or (state.slack_action_value or {}).get("issue_key")
    action = state.slack_action_type or (state.slack_action_value or {}).get("action")

    if state.error:
        msg = f"⚠️ Failed: {state.error}"
    elif action == "create":
        msg = f"✅ *Ticket Created:* <{JIRA_BASE_URL}/browse/{state.jira_key}|{state.jira_key}>"
    elif action == "update":
        msg = f"✏️ *Ticket Updated:* {issue_key}"
    elif action == "close":
        msg = f"✅ *Ticket Closed:* {issue_key}"
    else:
        msg = "⚠️ Unknown Jira action result"

    try:
        client.chat_update(channel=state.channel_id, ts=state.preview_ts, text=msg, blocks=None)
    except SlackApiError as e:
        logger.error(f"Slack update error: {e}")
    return state


def slack_post_cancel(state: OrchestratorState) -> OrchestratorState:
    """Update the Slack message to show cancellation."""
    client = _logged_slack(WebClient(token=SLACK_BOT_TOKEN))
    try:
        client.chat_update(
            channel=state.channel_id,
            ts=state.preview_ts,
            text="❌ Jira action cancelled.",
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


def route_jira_action(state: OrchestratorState) -> Literal[
    "slack_create_jira", "slack_update_jira", "slack_close_jira", "slack_post_cancel"
]:
    action = state.slack_action_type or (state.slack_action_value or {}).get("action")

    if action == "create":
        return "slack_create_jira"
    if action == "update":
        return "slack_update_jira"
    if action == "close":
        return "slack_close_jira"
    return "slack_post_cancel"


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_slack_subgraph() -> StateGraph:
    b = StateGraph(OrchestratorState)
    b.add_node("slack_extract_ticket",   slack_extract_ticket)
    b.add_node("slack_post_preview",     slack_post_preview)
    b.add_node("slack_resolve_assignee", slack_resolve_assignee)
    b.add_node("slack_create_jira",      slack_create_jira)
    b.add_node("slack_update_jira",      slack_update_jira)
    b.add_node("slack_close_jira",       slack_close_jira)
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
    b.add_edge("slack_post_preview", END)

    b.add_conditional_edges("slack_resolve_assignee", route_jira_action, {
        "slack_create_jira": "slack_create_jira",
        "slack_update_jira": "slack_update_jira",
        "slack_close_jira": "slack_close_jira",
        "slack_post_cancel": "slack_post_cancel",
    })

    b.add_edge("slack_create_jira", "slack_post_result")
    b.add_edge("slack_update_jira", "slack_post_result")
    b.add_edge("slack_close_jira", "slack_post_result")
    b.add_edge("slack_post_result", END)
    b.add_edge("slack_post_cancel", END)
    return b.compile()