"""
orchestrator.py — Top-level graph + Lambda entry point
=======================================================
Architecture:
  Lambda handler → parse_input → router_agent (Qwen LLM)
                                      │
                      ┌───────────────┴────────────────┐
                      ▼                                 ▼
              [SLACK SUBGRAPH]               [CALENDAR SUBGRAPH]
         Jira ticket creation             Autonomous email → meeting

Autonomous email flow:
  1. Gmail receives email → pushes to Pub/Sub → triggers Lambda
  2. parse_input detects pubsub format → intent=email
  3. calendar agent classifies with Qwen → posts Slack card
  4. User clicks ✅ Create Meeting → Slack sends interactivity to Lambda
  5. parse_input detects create_meeting action → intent=email
  6. calendar agent creates Google Calendar event
"""

import base64, json, logging, urllib.parse
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import OrchestratorState, _llm
from slack_agent import build_slack_subgraph
from calendar_agent import build_calendar_subgraph

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Router schema
# ─────────────────────────────────────────────────────────────────────────────
class RouterDecision(BaseModel):
    intent: Literal["slack", "email", "none"] = Field(
        description="'slack' if Slack/Jira task, 'email' if meeting invite, 'none' if neither"
    )
    reason: str = Field(description="One-sentence explanation")


# ─────────────────────────────────────────────────────────────────────────────
# parse_input
# ─────────────────────────────────────────────────────────────────────────────
def parse_input(state: OrchestratorState) -> OrchestratorState:
    event   = state.raw_event
    headers = {k.lower(): v for k, v in event.get("headers", {}).items()}

    # ── Slack retry guard ───────────────────────────────────────────────────
    if int(headers.get("x-slack-retry-num", 0)) > 0:
        return state.model_copy(update={"intent": "none", "is_retry": True})

    body_raw = event.get("body", "")
    if event.get("isBase64Encoded"):
        body_raw = base64.b64decode(body_raw).decode("utf-8")

    # ── Slack interactivity (button press) ──────────────────────────────────
    if body_raw.startswith("payload="):
        payload_str = urllib.parse.unquote_plus(body_raw.split("payload=")[1])
        payload     = json.loads(payload_str)
        action      = payload["actions"][0]
        action_id   = action["action_id"]
        raw_value   = urllib.parse.unquote_plus(action.get("value", "{}"))
        value_dict  = json.loads(raw_value) if raw_value and raw_value != "{}" else {}

        # ── Calendar meeting buttons ────────────────────────────────────────
        if action_id in ("create_meeting", "cancel_meeting"):
            return state.model_copy(update={
                "intent":             "email",
                "slack_event_type":   "interactivity",
                "slack_action_id":    action_id,
                "pending_meeting":    value_dict if action_id == "create_meeting" else None,
                "channel_id":         payload["channel"]["id"],
                "preview_ts":         payload["container"]["message_ts"],
            })

        # ── Jira ticket buttons ─────────────────────────────────────────────
        return state.model_copy(update={
            "intent":             "slack",
            "slack_event_type":   "interactivity",
            "slack_action_id":    action_id,
            "slack_action_value": value_dict,
            "channel_id":         payload["channel"]["id"],
            "preview_ts":         payload["container"]["message_ts"],
        })

    body_str = body_raw.strip()
    if not body_str:
        return state.model_copy(update={"intent": "none"})

    try:
        body = json.loads(body_str)
    except json.JSONDecodeError:
        return state.model_copy(update={"intent": "none", "error": "Unparseable body"})

    # ── Slack URL verification ───────────────────────────────────────────────
    if body.get("type") == "url_verification":
        return state.model_copy(update={"intent": "slack", "slack_event_type": "url_verification"})

    # ── Gmail Pub/Sub push — autonomous email trigger ────────────────────────
    if body.get("message", {}).get("data"):
        try:
            decoded = json.loads(base64.b64decode(body["message"]["data"]).decode("utf-8"))
            if "emailAddress" in decoded:
                return state.model_copy(update={
                    "intent":       "email",
                    "email_source": "pubsub",
                    "email_data":   body,
                })
        except Exception:
            pass

    # ── Slack message event ──────────────────────────────────────────────────
    if "event" in body:
        slack_event = body["event"]
        is_bot      = bool(slack_event.get("bot_id") or slack_event.get("subtype") == "bot_message")
        clean_text  = slack_event.get("text", "").replace("<@", "").replace(">", "")
        return state.model_copy(update={
            "intent":           "unknown",
            "slack_event_type": "message",
            "is_bot":           is_bot,
            "channel_id":       slack_event.get("channel"),
            "user_text":        clean_text,
            "message_ts":       slack_event.get("ts"),
        })

    # ── Plain email dict (direct / test invocation) ──────────────────────────
    if "from_email" in body or "subject" in body:
        return state.model_copy(update={
            "intent":       "email",
            "email_source": "direct",
            "email_data":   body,
        })

    return state.model_copy(update={"intent": "unknown"})


# ─────────────────────────────────────────────────────────────────────────────
# router_agent
# ─────────────────────────────────────────────────────────────────────────────
def router_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Only called when intent is still 'unknown' (Slack message events).
    All other intents are set definitively by parse_input.
    """
    if state.intent in ("slack", "email", "none"):
        return state

    if state.is_bot:
        return state.model_copy(update={"intent": "none"})

    context_parts = []
    if state.user_text:
        context_parts.append(f"Slack message: \"{state.user_text}\"")
    if state.email_data:
        ed = state.email_data
        context_parts.append(
            f"Email from {ed.get('from_email', '')} "
            f"subject: \"{ed.get('subject', '')}\" "
            f"body: \"{ed.get('snippet', ed.get('body', ''))[:300]}\""
        )

    if not context_parts:
        return state.model_copy(update={"intent": "none"})

    sys_msg = (
        "You are an intent router for a workplace automation system.\n"
        "  'slack'  — Slack message asking to create a Jira task\n"
        "  'email'  — email containing meeting scheduling intent\n"
        "  'none'   — not actionable\n"
        "Return a RouterDecision JSON."
    )

    llm = _llm(structured_output=RouterDecision)
    try:
        decision: RouterDecision = llm.invoke(
            [SystemMessage(content=sys_msg), HumanMessage(content="\n".join(context_parts))]
        )
        logger.info(f"Router: {decision.intent} — {decision.reason}")
        return state.model_copy(update={"intent": decision.intent, "intent_reason": decision.reason})
    except Exception as e:
        logger.error(f"Router LLM error: {e}")
        return state.model_copy(update={"intent": "none", "error": str(e)})


def route_to_agent(state: OrchestratorState) -> Literal["slack_subgraph", "calendar_subgraph", "__end__"]:
    if state.intent == "slack":
        return "slack_subgraph"
    if state.intent == "email":
        return "calendar_subgraph"
    return "__end__"


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────
def build_orchestrator(checkpointer=None) -> StateGraph:
    slack_sg    = build_slack_subgraph()
    calendar_sg = build_calendar_subgraph()

    b = StateGraph(OrchestratorState)
    b.add_node("parse_input",       parse_input)
    b.add_node("router_agent",      router_agent)
    b.add_node("slack_subgraph",    slack_sg)
    b.add_node("calendar_subgraph", calendar_sg)

    b.add_edge(START, "parse_input")
    b.add_edge("parse_input", "router_agent")
    b.add_conditional_edges("router_agent", route_to_agent, {
        "slack_subgraph":    "slack_subgraph",
        "calendar_subgraph": "calendar_subgraph",
        "__end__":           END,
    })
    b.add_edge("slack_subgraph",    END)
    b.add_edge("calendar_subgraph", END)

    return b.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# Lambda entry point
# ─────────────────────────────────────────────────────────────────────────────
_checkpointer = MemorySaver()
_graph        = build_orchestrator(checkpointer=_checkpointer)


def handler(event: dict, context) -> dict:
    logger.info(f"Event: {json.dumps(event)}")
    initial = OrchestratorState(raw_event=event)
    config  = {"configurable": {"thread_id": "global"}}

    try:
        result = _graph.invoke(initial, config=config)
        final  = OrchestratorState(**result) if isinstance(result, dict) else result
    except Exception as e:
        logger.error(f"Graph error: {e}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    # Slack URL verification
    if final.slack_event_type == "url_verification":
        body = json.loads(event.get("body", "{}"))
        return {"statusCode": 200, "body": json.dumps({"challenge": body.get("challenge")})}

    return {"statusCode": 200, "body": "ok"}
