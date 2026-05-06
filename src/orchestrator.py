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
from calendar_cod import build_calendar_subgraph_cod
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

        # For confirm_summary: value is direct JSON (not a Redis session ID)
        # For other actions: value is a Redis session ID
        if action_id in ("confirm_summary", "cancel_summary"):
            # Try direct JSON parse first (new approach — no Redis expiry risk)
            value_dict = {}
            if raw_value and raw_value not in ("", "cancel", "{}"):
                try:
                    value_dict = json.loads(raw_value)
                    session_id = None  # no session ID for direct JSON
                    logger.info("[trace=%s] confirm_summary: parsed direct JSON value", trace_id)
                except (json.JSONDecodeError, ValueError):
                    # Fallback: treat as Redis session ID (old cards)
                    value_dict = load_session(raw_value) or {}
                    if not value_dict:
                        logger.warning("[trace=%s] session %s expired or missing", trace_id, raw_value)
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

        if action_id in ("confirm_meeting_jira", "skip_meeting_jira"):
            # value is a Redis session ID for Jira button clicks — load it explicitly
            jira_value_dict = {}
            if session_id:
                jira_value_dict = load_session(session_id) or {}
                if not jira_value_dict:
                    logger.warning("[trace=%s] Jira session %s expired or missing", trace_id, session_id)
                else:
                    logger.info("[trace=%s] Jira session loaded ok: %s", trace_id, session_id)
            logger.info("[trace=%s] -> meeting_transcript (jira button: %s)", trace_id, action_id)
            return state.model_copy(update={
                "intent":             "meeting_transcript",
                "slack_event_type":   "interactivity",
                "slack_action_id":    action_id,
                "slack_action_value": jira_value_dict,
                "session_id":         session_id,
                "channel_id":         payload["channel"]["id"],
                "preview_ts":         payload["container"]["message_ts"],
            })

        if action_id in ("select_slot_0", "select_slot_1", "select_slot_2"):
            slot_index = int(action_id[-1])
            # Load the calendar session directly — value_dict is only set in
            # confirm_summary block which doesn't run for slot selection buttons
            slot_session_id = payload["actions"][0]["value"]
            slot_value_dict = {}
            try:
                slot_value_dict = load_session(slot_session_id) or {}
                if not slot_value_dict:
                    logger.warning("[trace=%s] select_slot: session %s not found or expired", trace_id, slot_session_id)
            except Exception as _se:
                logger.warning("[trace=%s] select_slot: session load failed: %s", trace_id, _se)
            all_slots  = slot_value_dict.get("all_proposed_slots", [])
            chosen     = all_slots[slot_index] if slot_index < len(all_slots) else {}
            logger.info(
                "[trace=%s] -> email slot %d selected session=%s start=%s",
                trace_id, slot_index, session_id, chosen.get("start"),
            )
            pending_meeting = {
                "email_data": slot_value_dict.get("email_data"),
                "model_output": {
                    "title":           slot_value_dict.get("meeting_title"),
                    "start_time":      chosen.get("start"),
                    "end_time":        chosen.get("end"),
                    "location":        slot_value_dict.get("meeting_location"),
                    "attendees":       slot_value_dict.get("meeting_attendees", []),
                    "time_confidence": slot_value_dict.get("time_confidence"),
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
        logger.warning("[trace=%s] empty body received", trace_id)
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

    if body.get("type") == "calendar_done":
        if body.get("secret") != WEBHOOK_SECRET:
            logger.warning("[trace=%s] calendar_done: invalid secret — rejecting", trace_id)
            return state.model_copy(update={"intent": "none", "error": "Invalid secret"})
        s3_key        = body.get("s3_key", "")
        calendar_link = body.get("calendar_link", "")
        logger.info("[trace=%s] calendar_done: s3_key=%s link=%s", trace_id, s3_key, calendar_link)

        # Update meta.json with calendar link
        if s3_key:
            try:
                import boto3 as _boto3, json as _json
                s3 = _boto3.client("s3")
                from state import S3_BUCKET
                raw  = s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/meta.json")["Body"].read()
                meta = _json.loads(raw.decode("utf-8"))
                meta["calendar_link"] = calendar_link
                meta["calendar_done"] = True
                s3.put_object(
                    Bucket=S3_BUCKET, Key=f"{s3_key}/meta.json",
                    Body=_json.dumps(meta, indent=2).encode(), ContentType="application/json",
                )
                logger.info("calendar_done: meta.json updated with calendar_link")
            except Exception as ce:
                logger.warning("calendar_done: could not update meta.json: %s", ce)

        # Release held consolidated email
        hold_key = f"calendar_hold:{s3_key}"
        try:
            from redis_store import _get_client as _get_r
            from meeting_agent import _send_consolidated_email
            import json as _json2
            _r = _get_r()
            hold_raw = _r.get(hold_key)
            if hold_raw:
                hold_data = _json2.loads(hold_raw.decode())
                _r.delete(hold_key)
                logger.info("calendar_done: found held email — sending with calendar_link=%s",
                            calendar_link[:60] if calendar_link else "none")

                # Update Slack with approved slot
                _channel = hold_data.get("channel_id", "") or hold_data.get("channel", "")
                if _channel:
                    try:
                        from slack_sdk import WebClient as _WC
                        from state import SLACK_BOT_TOKEN as _SBT
                        _slack = _WC(token=_SBT)
                        _slot_text = (
                            "\u2705 *Calendar event confirmed!*\n"
                            f"\U0001f4c5 Slot approved: <{calendar_link}|View calendar event>\n"
                            "_You will receive a Google Calendar invite separately._"
                            if calendar_link else
                            "\u2705 *Calendar agent completed scheduling.*\n"
                            "_You will receive a Google Calendar invite separately._"
                        )
                        _slack.chat_postMessage(channel=_channel, text=_slot_text)
                        logger.info("calendar_done: posted approved slot to Slack channel=%s", _channel)
                    except Exception as _se:
                        logger.warning("calendar_done: Slack update failed: %s", _se)

                _send_consolidated_email(
                    tickets_created=hold_data.get("tickets_created", []),
                    tickets_skipped=hold_data.get("tickets_skipped", 0),
                    file_name=hold_data.get("file_name", "meeting"),
                    channel=hold_data.get("channel", ""),
                    s3_key=hold_data.get("s3_key", ""),
                )
            else:
                logger.info(
                    "calendar_done: no held email found for s3_key=%s "
                    "(may have already sent or TTL expired)", s3_key,
                )
        except Exception as _re:
            logger.warning("calendar_done: could not release held email: %s", _re)

        return state.model_copy(update={"intent": "none"})

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
