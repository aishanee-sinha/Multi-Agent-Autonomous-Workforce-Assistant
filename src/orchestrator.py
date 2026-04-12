import base64
import json
import logging
import os
import re
import requests
import urllib.parse
from datetime import datetime, timezone, timedelta, date
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from redis_store import load_session

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import OrchestratorState, _llm, WEBHOOK_SECRET

from slack_agent import build_slack_subgraph
from calendar_agent import build_calendar_subgraph_cod
from meeting_agent import build_meeting_subgraph

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"

# ─────────────────────────────────────────────────────────────────────────────
# Routing logic
# ─────────────────────────────────────────────────────────────────────────────

def _trace_id_from_event(event: dict) -> str:
    headers    = {k.lower(): v for k, v in (event.get("headers", {}) or {}).items()}
    candidates = [
        headers.get("x-amzn-trace-id"),
        headers.get("x-slack-request-id"),
    ]
    body = event.get("body")
    if isinstance(body, str) and body:
        try:
            parsed = json.loads(body)
            evt    = parsed.get("event", {}) if isinstance(parsed, dict) else {}
            if isinstance(evt, dict):
                candidates.append(evt.get("event_ts"))
                candidates.append(evt.get("client_msg_id"))
            candidates.append(parsed.get("event_id") if isinstance(parsed, dict) else None)
        except Exception:
            pass
    for value in candidates:
        if value:
            return str(value).replace(" ", "_")[:120]
    return str(uuid4())


class RouterDecision(BaseModel):
    intent: Literal["slack", "email", "none"] = Field(
        description="'slack' if Slack/Jira task, 'email' if meeting invite, 'none' if neither"
    )
    reason: str = Field(description="One-sentence explanation")


def parse_input(state: OrchestratorState) -> OrchestratorState:
    event    = state.raw_event
    headers  = {k.lower(): v for k, v in event.get("headers", {}).items()}
    trace_id = _trace_id_from_event(event)
    logger.info("[trace=%s] parse_input start", trace_id)

    if int(headers.get("x-slack-retry-num", 0)) > 0:
        logger.info("[trace=%s] parse_input retry ignored", trace_id)
        return state.model_copy(update={"intent": "none", "is_retry": True})

    body_raw = event.get("body", "")
    if event.get("isBase64Encoded"):
        body_raw = base64.b64decode(body_raw).decode("utf-8")

    if body_raw.startswith("payload="):
        payload_str = urllib.parse.unquote_plus(body_raw.split("payload=")[1])
        payload     = json.loads(payload_str)
        action      = payload["actions"][0]
        action_id   = action["action_id"]
        logger.info("[trace=%s] parse_input interactivity action_id=%s", trace_id, action_id)
        raw_value  = urllib.parse.unquote_plus(action.get("value", ""))
        
        session_id = raw_value if raw_value and raw_value not in ("", "cancel", "{}") else None
        if session_id:
            value_dict = load_session(session_id) or {}
            if not value_dict:
                logger.warning("[trace=%s] session %s expired or missing", trace_id, session_id)
        else:
            value_dict = {}

        if action_id in ("confirm_summary", "cancel_summary"):
            logger.info("[trace=%s] -> meeting_transcript (button: %s)", trace_id, action_id)
            return state.model_copy(update={
                "intent":             "meeting_transcript",
                "slack_event_type":   "interactivity",
                "slack_action_id":    action_id,
                "slack_action_value": value_dict,
                "session_id":         session_id,
                "channel_id":         payload["channel"]["id"],
                "preview_ts":         payload["container"]["message_ts"],
            })

        if action_id in ("select_slot_0", "select_slot_1", "select_slot_2"):
            slot_index = int(action_id[-1])
            all_slots  = value_dict.get("all_proposed_slots", [])
            chosen     = all_slots[slot_index] if slot_index < len(all_slots) else {}
            logger.info(
                "[trace=%s] -> email slot %d selected session=%s start=%s",
                trace_id, slot_index, session_id, chosen.get("start"),
            )
            pending_meeting = {
                "email_data": value_dict.get("email_data"),
                "model_output": {
                    "title":           value_dict.get("meeting_title"),
                    "start_time":      chosen.get("start"),
                    "end_time":        chosen.get("end"),
                    "location":        value_dict.get("meeting_location"),
                    "attendees":       value_dict.get("meeting_attendees", []),
                    "time_confidence": value_dict.get("time_confidence"),
                },
                "all_proposed_slots": all_slots,
                "selected_slot_index": slot_index,
            }
            return state.model_copy(update={
                "intent":           "email",
                "slack_event_type": "interactivity",
                "slack_action_id":  "create_meeting",
                "pending_meeting":  pending_meeting,
                "selected_slot":    pending_meeting["model_output"],
                "session_id":       session_id,
                "channel_id":       payload["channel"]["id"],
                "preview_ts":       payload["container"]["message_ts"],
            })

        if action_id == "cancel_meeting":
            return state.model_copy(update={
                "intent":           "email",
                "slack_event_type": "interactivity",
                "slack_action_id":  "cancel_meeting",
                "pending_meeting":  None,
                "session_id":       session_id,
                "channel_id":       payload["channel"]["id"],
                "preview_ts":       payload["container"]["message_ts"],
            })
        return state.model_copy(update={
            "intent":             "slack",
            "slack_event_type":   "interactivity",
            "slack_action_id":    action_id,
            "slack_action_value": value_dict,
            "session_id":         session_id,
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

    if body.get("type") == "url_verification":
        return state.model_copy(update={"intent": "slack", "slack_event_type": "url_verification"})

    if body.get("type") == "new_transcript":
        if body.get("secret") != WEBHOOK_SECRET:
            logger.warning("[trace=%s] Invalid webhook secret — rejecting", trace_id)
            return state.model_copy(update={"intent": "none", "error": "Invalid secret"})
        logger.info("[trace=%s] -> meeting_transcript (file=%s)", trace_id, body.get("file_name"))
        return state.model_copy(update={
            "intent":               "meeting_transcript",
            "transcript_file_id":   body.get("file_id"),
            "transcript_file_name": body.get("file_name", "transcript.txt"),
        })

    if body.get("message", {}).get("data"):
        try:
            decoded = json.loads(base64.b64decode(body["message"]["data"]).decode("utf-8"))
            if "emailAddress" in decoded:
                logger.info("[trace=%s] parse_input pubsub detected", trace_id)
                return state.model_copy(update={
                    "intent":       "email",
                    "email_source": "pubsub",
                    "email_data":   body,
                })
        except Exception:
            pass

    if "event" in body:
        slack_event = body["event"]
        is_bot      = bool(slack_event.get("bot_id") or slack_event.get("subtype") == "bot_message")
        clean_text  = slack_event.get("text", "").replace("<@", "").replace(">", "")
        logger.info("[trace=%s] parse_input slack message is_bot=%s", trace_id, is_bot)
        return state.model_copy(update={
            "intent":           "unknown",
            "slack_event_type": "message",
            "is_bot":           is_bot,
            "channel_id":       slack_event.get("channel"),
            "user_text":        clean_text,
            "message_ts":       slack_event.get("ts"),
        })

    if "from_email" in body or "subject" in body:
        return state.model_copy(update={
            "intent":       "email",
            "email_source": "direct",
            "email_data":   body,
        })

    return state.model_copy(update={"intent": "unknown"})


def router_agent(state: OrchestratorState) -> OrchestratorState:
    trace_id = _trace_id_from_event(state.raw_event)
    if state.intent in ("slack", "email", "none", "meeting_transcript"):
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
    llm = _llm(model_name=MODEL, structured_output=RouterDecision)
    try:
        decision: RouterDecision = llm.invoke(
            [SystemMessage(content=sys_msg), HumanMessage(content="\n".join(context_parts))]
        )
        logger.info("[trace=%s] router intent=%s reason=%s", trace_id, decision.intent, decision.reason)
        return state.model_copy(update={"intent": decision.intent, "intent_reason": decision.reason})
    except Exception as e:
        logger.error("[trace=%s] router LLM error: %s", trace_id, e)
        return state.model_copy(update={"intent": "none", "error": str(e)})


def route_to_agent(state: OrchestratorState) -> Literal["slack_subgraph", "calendar_subgraph", "meeting_subgraph", "__end__"]:
    trace_id = _trace_id_from_event(state.raw_event)
    if state.intent == "slack":
        logger.info("[trace=%s] route -> slack_subgraph", trace_id)
        return "slack_subgraph"
    if state.intent == "email":
        logger.info("[trace=%s] route -> calendar_subgraph", trace_id)
        return "calendar_subgraph"
    if state.intent == "meeting_transcript":
        logger.info("[trace=%s] route -> meeting_subgraph", trace_id)
        return "meeting_subgraph"
    return "__end__"


# ─────────────────────────────────────────────────────────────────────────────
# Full orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_orchestrator(checkpointer=None):
    b = StateGraph(OrchestratorState)
    b.add_node("parse_input",       parse_input)
    b.add_node("router_agent",      router_agent)
    b.add_node("slack_subgraph",    build_slack_subgraph())
    b.add_node("calendar_subgraph", build_calendar_subgraph_cod())
    b.add_node("meeting_subgraph",  build_meeting_subgraph())

    b.add_edge(START, "parse_input")
    b.add_edge("parse_input", "router_agent")
    b.add_conditional_edges("router_agent", route_to_agent, {
        "slack_subgraph":    "slack_subgraph",
        "calendar_subgraph": "calendar_subgraph",
        "meeting_subgraph":  "meeting_subgraph",
        "__end__":           END,
    })
    b.add_edge("slack_subgraph",    END)
    b.add_edge("calendar_subgraph", END)
    b.add_edge("meeting_subgraph",  END)
    return b.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# Lambda entry point
# ─────────────────────────────────────────────────────────────────────────────

_checkpointer = MemorySaver()
_graph        = build_orchestrator(checkpointer=_checkpointer)


def sqs_handler(event: dict, context) -> dict:
    """SQS trigger entry point — unwraps each record and runs the CoD graph."""
    records = event.get("Records", [])
    logger.info("sqs_handler: received %d record(s)", len(records))
    for record in records:
        try:
            http_event = json.loads(record["body"])
            handler(http_event, context)
        except Exception as e:
            logger.error("sqs_handler: failed to process record: %s", e, exc_info=True)
            raise
    return {"statusCode": 200, "body": "ok"}


def handler(event: dict, context) -> dict:
    trace_id = _trace_id_from_event(event)
    logger.info("[trace=%s] orchestrator handler start", trace_id)

    initial = OrchestratorState(raw_event=event)
    config  = {"configurable": {"thread_id": trace_id}}
    try:
        result = _graph.invoke(initial, config=config)
        final  = OrchestratorState(**result) if isinstance(result, dict) else result
    except Exception as e:
        logger.error("[trace=%s] graph error: %s", trace_id, e, exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    if final.slack_event_type == "url_verification":
        body = json.loads(event.get("body", "{}"))
        return {"statusCode": 200, "body": json.dumps({"challenge": body.get("challenge")})}

    return {"statusCode": 200, "body": "ok"}
