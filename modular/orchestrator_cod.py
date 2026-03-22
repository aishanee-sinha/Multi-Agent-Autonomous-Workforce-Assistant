"""
orchestrator.py — Top-level graph + Lambda entry point
=======================================================
Architecture:
  Lambda handler → parse_input → router_agent (Qwen LLM)
                                      │
                              chain_of_debate          ← NEW
                         (Proposer ↔ Challenger → Judge)
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

Chain-of-Debate (router only):
  Runs only when intent is still 'unknown' after router_agent.
  Round 1 — Proposer argues for router_agent's initial call.
  Round 2 — Challenger argues against / proposes an alternative.
  Round 3 — Judge reads both arguments and issues the final verdict.
  All three roles call the same external Anthropic claude-sonnet-4-20250514
  model directly via the REST API so no EC2 instance is involved.
"""

import base64, json, logging, os, urllib.parse, urllib.request
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import OrchestratorState, _llm
from slack_agent import build_slack_subgraph
from calendar_agent import build_calendar_subgraph

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def _trace_id_from_event(event: dict) -> str:
    """Build a stable trace id for one invocation so logs can be correlated."""
    headers = {k.lower(): v for k, v in (event.get("headers", {}) or {}).items()}
    candidates = [
        headers.get("x-amzn-trace-id"),
        headers.get("x-slack-request-id"),
    ]
    body = event.get("body")
    if isinstance(body, str) and body:
        try:
            parsed = json.loads(body)
            evt = parsed.get("event", {}) if isinstance(parsed, dict) else {}
            if isinstance(evt, dict):
                candidates.append(evt.get("event_ts"))
                candidates.append(evt.get("client_msg_id"))
            candidates.append(parsed.get("event_id") if isinstance(parsed, dict) else None)
        except Exception:
            pass

    for value in candidates:
        if value:
            safe = str(value).replace(" ", "_")
            return safe[:120]
    return str(uuid4())


# ─────────────────────────────────────────────────────────────────────────────
# Router schema
# ─────────────────────────────────────────────────────────────────────────────
class RouterDecision(BaseModel):
    intent: Literal["slack", "email", "none"] = Field(
        description="'slack' if Slack/Jira task, 'email' if meeting invite, 'none' if neither"
    )
    reason: str = Field(description="One-sentence explanation")


# ─────────────────────────────────────────────────────────────────────────────
# Chain-of-Debate helpers
# ─────────────────────────────────────────────────────────────────────────────
_DEBATE_MODEL   = "claude-sonnet-4-20250514"
_ANTHROPIC_URL  = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VER  = "2023-06-01"
_VALID_INTENTS  = {"slack", "email", "none"}


def _anthropic_chat(system: str, user: str, max_tokens: int = 512) -> str:
    """
    Thin wrapper around the Anthropic /v1/messages REST endpoint.
    Reads ANTHROPIC_API_KEY from the environment (same Lambda env-var set).
    Does NOT touch the EC2 instance — pure HTTPS to api.anthropic.com.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY env var is not set")

    payload = json.dumps({
        "model":      _DEBATE_MODEL,
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   [{"role": "user", "content": user}],
    }).encode("utf-8")

    req = urllib.request.Request(
        _ANTHROPIC_URL,
        data=payload,
        headers={
            "content-type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": _ANTHROPIC_VER,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    # content is a list of blocks; grab the first text block
    for block in body.get("content", []):
        if block.get("type") == "text":
            return block["text"].strip()
    return ""


def _extract_intent(text: str) -> str | None:
    """
    Parse the judge's free-text verdict and return a canonical intent string,
    or None if no valid intent is found.
    """
    lower = text.lower()
    for intent in _VALID_INTENTS:
        if intent in lower:
            return intent
    return None


def chain_of_debate(state: OrchestratorState) -> OrchestratorState:
    """
    Chain-of-Debate node — runs only when intent is still 'unknown'.

    Flow
    ────
    Round 1 (Proposer)   — argues *for* the initial router_agent verdict.
    Round 2 (Challenger) — argues *against* and proposes an alternative.
    Round 3 (Judge)      — reads both arguments and issues the final intent.

    All three LLM calls go directly to the Anthropic API (claude-sonnet-4).
    The EC2 instance is never contacted.
    """
    trace_id = _trace_id_from_event(state.raw_event)

    # Only run when the intent is still unresolved
    if state.intent != "unknown":
        logger.info(
            "[trace=%s] chain_of_debate skipped — intent already resolved: %s",
            trace_id, state.intent,
        )
        return state

    # ── Build shared context ─────────────────────────────────────────────────
    context_parts: list[str] = []
    if state.user_text:
        context_parts.append(f'Slack message: "{state.user_text}"')
    if state.email_data:
        ed = state.email_data
        context_parts.append(
            f"Email from {ed.get('from_email', '')} "
            f"subject: \"{ed.get('subject', '')}\" "
            f"body: \"{ed.get('snippet', ed.get('body', ''))[:300]}\""
        )

    if not context_parts:
        logger.info("[trace=%s] chain_of_debate — no context, defaulting to none", trace_id)
        return state.model_copy(update={"intent": "none"})

    context_block = "\n".join(context_parts)
    initial_intent = getattr(state, "intent_reason", "unknown")

    # ── Round 1: Proposer ────────────────────────────────────────────────────
    proposer_sys = (
        "You are the Proposer in an intent-routing debate for a workplace automation system.\n"
        "Valid intents:\n"
        "  slack — Slack message asking to create a Jira task\n"
        "  email — email containing meeting scheduling intent\n"
        "  none  — not actionable\n\n"
        "Your job: argue clearly and concisely (≤3 sentences) WHY the initial "
        "classification is correct. Focus on evidence from the message."
    )
    proposer_user = (
        f"Input context:\n{context_block}\n\n"
        f"Initial classification: {initial_intent}\n\n"
        "Make the strongest case FOR this classification."
    )

    try:
        proposer_arg = _anthropic_chat(proposer_sys, proposer_user)
        logger.info("[trace=%s] chain_of_debate proposer: %s", trace_id, proposer_arg[:120])
    except Exception as e:
        logger.error("[trace=%s] chain_of_debate proposer error: %s", trace_id, e)
        return state  # fall back to existing (unknown) intent for re-routing

    # ── Round 2: Challenger ──────────────────────────────────────────────────
    challenger_sys = (
        "You are the Challenger in an intent-routing debate for a workplace automation system.\n"
        "Valid intents:\n"
        "  slack — Slack message asking to create a Jira task\n"
        "  email — email containing meeting scheduling intent\n"
        "  none  — not actionable\n\n"
        "Your job: argue against the Proposer's reasoning (≤3 sentences) and "
        "propose a different or refined classification if warranted. "
        "If you genuinely agree with the Proposer, state so briefly."
    )
    challenger_user = (
        f"Input context:\n{context_block}\n\n"
        f"Proposer's argument (for '{initial_intent}'):\n{proposer_arg}\n\n"
        "Challenge this classification or concede if correct."
    )

    try:
        challenger_arg = _anthropic_chat(challenger_sys, challenger_user)
        logger.info("[trace=%s] chain_of_debate challenger: %s", trace_id, challenger_arg[:120])
    except Exception as e:
        logger.error("[trace=%s] chain_of_debate challenger error: %s", trace_id, e)
        return state

    # ── Round 3: Judge ───────────────────────────────────────────────────────
    judge_sys = (
        "You are the Judge in an intent-routing debate for a workplace automation system.\n"
        "Valid intents (pick exactly one): slack | email | none\n\n"
        "  slack — Slack message requesting Jira task creation\n"
        "  email — email containing meeting scheduling intent\n"
        "  none  — not actionable\n\n"
        "Read the Proposer and Challenger arguments, weigh the evidence, and "
        "output ONLY a JSON object — no prose before or after — with two keys:\n"
        '  { "intent": "<slack|email|none>", "reason": "<one sentence>" }'
    )
    judge_user = (
        f"Input context:\n{context_block}\n\n"
        f"Proposer (for '{initial_intent}'):\n{proposer_arg}\n\n"
        f"Challenger:\n{challenger_arg}\n\n"
        "Deliver your verdict as JSON."
    )

    try:
        judge_raw = _anthropic_chat(judge_sys, judge_user, max_tokens=256)
        logger.info("[trace=%s] chain_of_debate judge raw: %s", trace_id, judge_raw[:200])

        # Strip optional markdown fences before parsing
        clean = judge_raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        verdict = json.loads(clean)
        final_intent = verdict.get("intent", "").lower()
        final_reason = verdict.get("reason", "")

        if final_intent not in _VALID_INTENTS:
            logger.warning(
                "[trace=%s] chain_of_debate judge returned invalid intent '%s', using 'none'",
                trace_id, final_intent,
            )
            final_intent = "none"
            final_reason = "Judge returned unrecognised intent; defaulted to none."

    except json.JSONDecodeError:
        # Try regex-style fallback — extract any valid intent keyword from text
        final_intent = _extract_intent(judge_raw) or "none"
        final_reason = f"Judge JSON parse failed; extracted intent from text: {judge_raw[:100]}"
        logger.warning(
            "[trace=%s] chain_of_debate judge JSON parse failed, fallback intent=%s",
            trace_id, final_intent,
        )
    except Exception as e:
        logger.error("[trace=%s] chain_of_debate judge error: %s", trace_id, e)
        return state  # keep intent as unknown; route_to_agent will send to END

    logger.info(
        "[trace=%s] chain_of_debate final: intent=%s reason=%s",
        trace_id, final_intent, final_reason,
    )
    return state.model_copy(update={
        "intent":        final_intent,
        "intent_reason": final_reason,
        "debate_log": {
            "proposer":   proposer_arg,
            "challenger": challenger_arg,
            "judge_raw":  judge_raw,
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# parse_input
# ─────────────────────────────────────────────────────────────────────────────
def parse_input(state: OrchestratorState) -> OrchestratorState:
    event   = state.raw_event
    headers = {k.lower(): v for k, v in event.get("headers", {}).items()}
    trace_id = _trace_id_from_event(event)
    logger.info("[trace=%s] parse_input start", trace_id)

    # ── Slack retry guard ───────────────────────────────────────────────────
    if int(headers.get("x-slack-retry-num", 0)) > 0:
        logger.info("[trace=%s] parse_input retry ignored: retry_num=%s", trace_id, headers.get("x-slack-retry-num"))
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
        logger.info("[trace=%s] parse_input interactivity action_id=%s", trace_id, action_id)
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
        logger.info("[trace=%s] parse_input empty body", trace_id)
        return state.model_copy(update={"intent": "none"})

    try:
        body = json.loads(body_str)
    except json.JSONDecodeError:
        logger.warning("[trace=%s] parse_input body is not valid JSON", trace_id)
        return state.model_copy(update={"intent": "none", "error": "Unparseable body"})

    # ── Slack URL verification ───────────────────────────────────────────────
    if body.get("type") == "url_verification":
        logger.info("[trace=%s] parse_input url_verification", trace_id)
        return state.model_copy(update={"intent": "slack", "slack_event_type": "url_verification"})

    # ── Gmail Pub/Sub push — autonomous email trigger ────────────────────────
    if body.get("message", {}).get("data"):
        try:
            decoded = json.loads(base64.b64decode(body["message"]["data"]).decode("utf-8"))
            if "emailAddress" in decoded:
                logger.info("[trace=%s] parse_input pubsub email trigger detected", trace_id)
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
        logger.info(
            "[trace=%s] parse_input slack message event: is_bot=%s channel=%s",
            trace_id,
            is_bot,
            slack_event.get("channel"),
        )
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
        logger.info("[trace=%s] parse_input direct email payload", trace_id)
        return state.model_copy(update={
            "intent":       "email",
            "email_source": "direct",
            "email_data":   body,
        })

    logger.info("[trace=%s] parse_input fell through to unknown intent", trace_id)
    return state.model_copy(update={"intent": "unknown"})


# ─────────────────────────────────────────────────────────────────────────────
# router_agent
# ─────────────────────────────────────────────────────────────────────────────
def router_agent(state: OrchestratorState) -> OrchestratorState:
    """
    Only called when intent is still 'unknown' (Slack message events).
    All other intents are set definitively by parse_input.
    Produces an initial intent that chain_of_debate will then verify/refine.
    """
    trace_id = _trace_id_from_event(state.raw_event)
    if state.intent in ("slack", "email", "none"):
        logger.info("[trace=%s] router skipped with pre-set intent=%s", trace_id, state.intent)
        return state

    if state.is_bot:
        logger.info("[trace=%s] router ignored bot message", trace_id)
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
        logger.info("[trace=%s] router found no actionable context", trace_id)
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
        logger.info("[trace=%s] router decision: intent=%s reason=%s", trace_id, decision.intent, decision.reason)
        # Keep intent as 'unknown' so chain_of_debate is triggered; store the
        # initial call in intent_reason for the Proposer to argue from.
        return state.model_copy(update={
            "intent":        "unknown",   # chain_of_debate will finalise this
            "intent_reason": f"{decision.intent}: {decision.reason}",
        })
    except Exception as e:
        logger.error("[trace=%s] router LLM error: %s", trace_id, e)
        return state.model_copy(update={"intent": "none", "error": str(e)})


def route_to_agent(state: OrchestratorState) -> Literal["slack_subgraph", "calendar_subgraph", "__end__"]:
    trace_id = _trace_id_from_event(state.raw_event)
    if state.intent == "slack":
        logger.info("[trace=%s] route_to_agent -> slack_subgraph", trace_id)
        return "slack_subgraph"
    if state.intent == "email":
        logger.info("[trace=%s] route_to_agent -> calendar_subgraph", trace_id)
        return "calendar_subgraph"
    logger.info("[trace=%s] route_to_agent -> end", trace_id)
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
    b.add_node("chain_of_debate",   chain_of_debate)   # ← NEW
    b.add_node("slack_subgraph",    slack_sg)
    b.add_node("calendar_subgraph", calendar_sg)

    b.add_edge(START,              "parse_input")
    b.add_edge("parse_input",      "router_agent")
    b.add_edge("router_agent",     "chain_of_debate")  # ← NEW
    b.add_conditional_edges("chain_of_debate", route_to_agent, {  # ← was router_agent
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
    import os
    trace_id = _trace_id_from_event(event)
    logger.info("[trace=%s] handler start", trace_id)
    logger.info("=== ALL ENV VARS ===")
    for key in ["SLACK_NOTIFY_CHANNEL", "SLACK_BOT_TOKEN", "EC2_IP",
                "GOOGLE_TOKEN_JSON", "GROUP_EMAILS_JSON", "JIRA_BASE_URL",
                "ANTHROPIC_API_KEY"]:   # ← added for chain_of_debate
        val = os.environ.get(key, "NOT SET")
        # Don't log full secrets — just first 10 chars
        safe = val[:10] + "..." if len(val) > 10 else val
        logger.info("[trace=%s] env %s=%s", trace_id, key, safe)
    logger.info("[trace=%s] event=%s", trace_id, json.dumps(event))
    initial = OrchestratorState(raw_event=event)
    config  = {"configurable": {"thread_id": trace_id}}
    logger.info("[trace=%s] graph invoke thread_id=%s", trace_id, trace_id)

    try:
        result = _graph.invoke(initial, config=config)
        final  = OrchestratorState(**result) if isinstance(result, dict) else result
    except Exception as e:
        logger.error("[trace=%s] graph error: %s", trace_id, e, exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

    # Slack URL verification
    if final.slack_event_type == "url_verification":
        logger.info("[trace=%s] handler returning Slack URL verification response", trace_id)
        body = json.loads(event.get("body", "{}"))
        return {"statusCode": 200, "body": json.dumps({"challenge": body.get("challenge")})}

    logger.info("[trace=%s] handler complete status=200", trace_id)
    return {"statusCode": 200, "body": "ok"}
