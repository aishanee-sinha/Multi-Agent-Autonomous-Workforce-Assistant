"""
meeting_agent.py — Meeting Summarizer LangGraph subgraph
=========================================================
Integrates with peers' orchestrator as a third subgraph alongside
slack_agent (Jira) and calendar_cod (Google Calendar + CoD slot selection).

Pipeline nodes (new transcript trigger):
  1. meeting_fetch_transcript     Download from Drive + S3 idempotency lock (FIX-1)
  2. meeting_preprocess           Clean/normalise transcript text
  3. meeting_summarize            Chunked vLLM inference + merge-pass abstract (FIX-2)
  4. meeting_triage_cod           Chain-of-Debate: classify each action item by
                                  priority (Critical/High/Medium/Low) + risk
  5. meeting_generate_artifacts   ICS UTC-correct (FIX-3) + CSV
  6. meeting_store_s3             All artifacts → S3, sets meeting_s3_key
  7. meeting_post_slack           Block Kit card with triage, Confirm/Cancel buttons

Button click nodes:
  8a. meeting_send_email          Fetch artifacts from S3, send via SES
  8b. meeting_post_cancel         Update Slack message to dismissed

CoD triage (node 4) — analogous to calendar_cod.py's slot_cod:
  Same 3-role Proposer → Challenger → Judge pattern.
  Input:  list of action items + full transcript context
  Output: each action item annotated with priority + risk + deadline assessment
  Slack card shows colour-coded triage (🔴 Critical / 🟠 High / 🟡 Medium / 🟢 Low)
  so the approver sees what needs immediate attention before clicking Confirm.

Stateless button design (same pattern as calendar_agent):
  s3_key + file_name embedded in Slack button value JSON. No DynamoDB needed.
"""

import io
import json
import logging
import os
import re
import time
import csv as csv_mod
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import boto3
from botocore.exceptions import ClientError
import requests
from slack_sdk import WebClient
from google.oauth2 import service_account
from googleapiclient.discovery import build as gdrive_build
from googleapiclient.http import MediaIoBaseDownload
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import (
    OrchestratorState,
    S3_BUCKET, S3_PREFIX, VLLM_MODEL_NAME, EC2_IP,
    SLACK_BOT_TOKEN, SLACK_NOTIFY_CHANNEL,
    SES_FROM_EMAIL, PARTICIPANT_EMAILS, _llm,
)
from feedback_logger import log_feedback
from rag_retriever import get_similar_approved, format_as_few_shot

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SCOPES           = ["https://www.googleapis.com/auth/drive.readonly"]
CHUNK_SIZE       = 10000
CHUNK_OVERLAP    = 1000
LOCK_TTL_SECONDS = 1800   # 30 min — stale lock TTL

# FIX-3: Timezone abbreviation → UTC offset hours
TZ_OFFSETS = {
    "PDT": -7,   "PST": -8,
    "MDT": -6,   "MST": -7,
    "CDT": -5,   "CST": -6,
    "EDT": -4,   "EST": -5,
    "BST": +1,   "GMT":  0,  "UTC": 0,
    "CET": +1,   "CEST": +2,
    "IST": +5.5,
    "JST": +9,   "KST": +9,
    "AEST": +10, "AEDT": +11,
}
_TZ_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in TZ_OFFSETS) + r")\b")
_ACTIONS_JSON_RE = re.compile(r"ACTIONS_JSON:\s*(\[.*?\])", re.DOTALL | re.IGNORECASE)
_SECTION_RE = re.compile(
    r"(ABSTRACT|DECISIONS|PROBLEMS|ACTIONS):\s*(.*?)(?=\n(?:ABSTRACT|DECISIONS|PROBLEMS|ACTIONS|ACTIONS_JSON):|$)",
    re.DOTALL | re.IGNORECASE,
)

# Priority emoji for Slack display
PRIORITY_EMOJI = {
    "Critical": "🔴",
    "High":     "🟠",
    "Medium":   "🟡",
    "Low":      "🟢",
}

MODEL = "meeting"   # vLLM LoRA adapter name — same as VLLM_MODEL_NAME


# ─────────────────────────────────────────────────────────────────────────────
# CoD Pydantic schemas  (same pattern as calendar_cod.py SlotProposal etc.)
# ─────────────────────────────────────────────────────────────────────────────

class ActionPriorityProposal(BaseModel):
    priority:    str = Field(description="Critical | High | Medium | Low")
    risk:        str = Field(description="1-sentence risk if this action item is missed or delayed")
    deadline_ok: bool = Field(description="True if stated deadline is realistic given meeting context")
    argument:    str = Field(description="1-2 sentence justification for the priority level")


class PriorityChallenge(BaseModel):
    agrees:           bool = Field(description="True if challenger agrees with proposed priority")
    counter_priority: str  = Field(description="Challenger's suggested priority, or same if agreeing")
    argument:         str  = Field(description="1-2 sentence challenge or concession")


class TriageVerdict(BaseModel):
    final_priority: str  = Field(description="Critical | High | Medium | Low — final decision")
    risk_summary:   str  = Field(description="One sentence: risk if item is delayed or missed")
    deadline_note:  str  = Field(description="'On track' or short note if deadline seems unrealistic")
    rationale:      str  = Field(description="One sentence explaining the priority choice")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────
P4_SYSTEM_PROMPT = (
    "You are a professional meeting minutes assistant. "
    "Your output must contain exactly these sections in order: "
    "ABSTRACT, DECISIONS, PROBLEMS, ACTIONS, ACTIONS_JSON. "
    "ACTIONS_JSON must be a valid JSON array."
)
P4_USER_INSTRUCTION = (
    "Given the meeting transcript below, produce:\n"
    "  ABSTRACT:   A concise paragraph summarising the meeting.\n"
    "  DECISIONS:  Bullet list of decisions made.\n"
    "  PROBLEMS:   Bullet list of problems or risks raised.\n"
    "  ACTIONS:    Bullet list formatted as [Owner] - task - Due: deadline.\n"
    "  ACTIONS_JSON: JSON array, each item has keys: "
    "owner, task, deadline, discussed_at_sec.\n"
    "Rules: Do not fabricate facts. Set owner/deadline to TBD when not mentioned. "
    "ACTIONS_JSON must parse as valid JSON. discussed_at_sec is float or 0.0."
)
MERGE_SYSTEM_PROMPT = (
    "You are a professional meeting minutes assistant. "
    "Merge the partial abstracts below into a single coherent paragraph of 80-200 words. "
    "Do not add facts not in the input. Output only the merged paragraph — no headers."
)


def _chunk_prompt(transcript: str) -> list:
    return [
        {"role": "system", "content": P4_SYSTEM_PROMPT},
        {"role": "user",   "content": f"{P4_USER_INSTRUCTION}\n\nTRANSCRIPT:\n{transcript}"},
    ]


def _merge_prompt(abstracts: list[str]) -> list:
    joined = "\n\n---\n\n".join(f"Part {i+1}:\n{a}" for i, a in enumerate(abstracts))
    return [
        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Partial abstracts:\n\n{joined}"},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lock_key(file_id: str) -> str:
    return f"{S3_PREFIX}/locks/{file_id}.lock"


def _acquire_s3_lock(s3, file_id: str) -> bool:
    """FIX-1: Write S3 lock before processing. Returns False if already active."""
    key = _lock_key(file_id)
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body = json.loads(obj["Body"].read())
        age  = time.time() - body.get("ts", 0)
        if age < LOCK_TTL_SECONDS:
            logger.info("Lock active for %s (age=%.0fs)", file_id, age)
            return False
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
    s3.put_object(
        Bucket=S3_BUCKET, Key=key,
        Body=json.dumps({"ts": time.time(), "status": "processing"}).encode(),
        ContentType="application/json",
    )
    logger.info("Lock acquired for %s", file_id)
    return True


def _release_s3_lock(s3, file_id: str, status: str = "done"):
    try:
        s3.put_object(
            Bucket=S3_BUCKET, Key=_lock_key(file_id),
            Body=json.dumps({"ts": time.time(), "status": status}).encode(),
            ContentType="application/json",
        )
    except Exception as e:
        logger.warning("Could not update lock: %s", e)


def _download_from_drive(file_id: str) -> str:
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
    sa_info = json.loads(sa_json)
    creds   = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    svc     = gdrive_build("drive", "v3", credentials=creds)
    req     = svc.files().get_media(fileId=file_id)
    fh      = io.BytesIO()
    dl      = MediaIoBaseDownload(fh, req)
    done    = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue().decode("utf-8")


def _preprocess(transcript: str) -> str:
    FILLERS = {
        "hmm","yeah","okay","ok","uh","um","huh","mm","mm-hmm","yes","no",
        "right","sure","bye","hello","hi","hey","oh","ah","hm","yep","nope",
        "alright","good","great","cool","nice","k",
    }
    speaker_re = re.compile(r"^\[(.+?)\]\s*[\d:]+\s*$")
    lines = transcript.splitlines()
    utterances = []; current_speaker = None; current_text = []

    for line in lines:
        line = line.strip()
        if not line: continue
        m = speaker_re.match(line)
        if m:
            if current_speaker and current_text:
                utterances.append((current_speaker, " ".join(current_text)))
            current_speaker = m.group(1).strip(); current_text = []
        elif current_speaker:
            if len(line) < 5: continue
            clean = re.sub(r"[^a-zA-Z\s]", "", line).strip().lower()
            words = clean.split()
            if words and all(w in FILLERS for w in words): continue
            current_text.append(line)

    if current_speaker and current_text:
        utterances.append((current_speaker, " ".join(current_text)))

    merged = []
    for speaker, text in utterances:
        if merged and merged[-1][0] == speaker:
            merged[-1][1] += " " + text
        else:
            merged.append([speaker, text])

    result_lines = [f"[{s}]: {t.strip()}" for s, t in merged if t.strip()]
    processed    = "\n".join(result_lines)

    if not processed.strip() or (len(processed) < 200 and len(transcript) > 500):
        return transcript
    return processed


def _vllm_call(messages: list, max_new_tokens: int = 768) -> str:
    resp = requests.post(
        f"http://{EC2_IP}:8000/v1/chat/completions",
        json={"model": VLLM_MODEL_NAME, "messages": messages,
              "max_tokens": max_new_tokens, "temperature": 0.0},
        headers={"Content-Type": "application/json"}, timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _clean_summary(raw: str) -> str:
    headers = ["ABSTRACT:", "DECISIONS:", "ACTIONS:", "PROBLEMS:"]
    earliest = len(raw)
    for h in headers:
        pos = raw.upper().find(h)
        if pos != -1 and pos < earliest: earliest = pos
    return raw[earliest:].strip() if earliest < len(raw) else raw.strip()


def _extract_section(text: str, section: str) -> str:
    pattern = rf"{section}:\s*(.*?)(?=\n(?:ABSTRACT|DECISIONS|PROBLEMS|ACTIONS|ACTIONS_JSON):|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _run_chunked_inference(transcript: str) -> str:
    """FIX-2: Chunked inference with merge-pass abstract."""
    chunks = []; start = 0
    while start < len(transcript):
        end = start + CHUNK_SIZE
        chunks.append(transcript[start:end])
        next_start = end - CHUNK_OVERLAP
        if next_start <= start: break
        start = next_start

    logger.info("Inference: %d chars | %d chunk(s)", len(transcript), len(chunks))

    all_abstracts = []; all_decisions = []; all_problems = []
    all_actions   = []; all_action_jsons = []

    for i, chunk in enumerate(chunks):
        logger.info("Chunk %d/%d", i + 1, len(chunks))
        try:
            content  = _vllm_call(_chunk_prompt(chunk))
            cleaned  = _clean_summary(content)
            if abs := _extract_section(cleaned, "ABSTRACT"):  all_abstracts.append(abs)
            if dec := _extract_section(cleaned, "DECISIONS"):  all_decisions.append(dec)
            if prb := _extract_section(cleaned, "PROBLEMS"):   all_problems.append(prb)
            if act := _extract_section(cleaned, "ACTIONS"):    all_actions.append(act)
            m = _ACTIONS_JSON_RE.search(cleaned)
            if m:
                try: all_action_jsons.extend(json.loads(m.group(1)))
                except Exception: pass
        except Exception as e:
            logger.error("Chunk %d failed: %s", i + 1, e)

    if not all_abstracts:
        merged_abstract = "No abstract generated."
    elif len(all_abstracts) == 1:
        merged_abstract = all_abstracts[0]
    else:
        logger.info("Merge-pass for %d abstracts", len(all_abstracts))
        try:
            merged_abstract = _vllm_call(_merge_prompt(all_abstracts), max_new_tokens=256).strip()
        except Exception as e:
            logger.error("Merge-pass failed: %s", e)
            merged_abstract = all_abstracts[0]

    def _dedup(blocks):
        seen, out = set(), []
        for block in blocks:
            for line in block.splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line); out.append(line)
        return "\n".join(out)

    seen_tasks, deduped = set(), []
    for ai in all_action_jsons:
        key = str(ai.get("task", ""))[:80].lower().strip()
        if key and key not in seen_tasks:
            seen_tasks.add(key); deduped.append(ai)

    result  = f"ABSTRACT:\n{merged_abstract}\n\n"
    result += f"DECISIONS:\n{_dedup(all_decisions) or 'None identified.'}\n\n"
    result += f"PROBLEMS:\n{_dedup(all_problems) or 'None identified.'}\n\n"
    result += f"ACTIONS:\n{_dedup(all_actions) or 'None identified.'}\n\n"
    result += f"ACTIONS_JSON:\n{json.dumps(deduped, indent=2)}"
    return result


def _parse_model_output(raw: str) -> dict:
    result = {"abstract": "", "decisions": "", "problems": "", "actions": "", "actions_json": [], "raw_output": raw}
    for match in _SECTION_RE.finditer(raw):
        sec = match.group(1).upper(); content = match.group(2).strip()
        if sec == "ABSTRACT":    result["abstract"]   = content
        elif sec == "DECISIONS": result["decisions"]  = content
        elif sec == "PROBLEMS":  result["problems"]   = content
        elif sec == "ACTIONS":   result["actions"]    = content
    m = _ACTIONS_JSON_RE.search(raw)
    if m:
        try: result["actions_json"] = json.loads(m.group(1))
        except Exception: pass
    return result


def _detect_tz_offset(transcript: str) -> float:
    """FIX-3: Return UTC offset hours from first TZ abbreviation found."""
    m = _TZ_RE.search(transcript)
    if m:
        offset = TZ_OFFSETS[m.group(1)]
        logger.info("TZ detected: %s (UTC%+.1f)", m.group(1), offset)
        return float(offset)
    return 0.0


def _parse_meeting_datetime(transcript: str, action_items: list) -> datetime:
    """FIX-3: Parse local time from transcript/actions, return true UTC datetime."""
    tz_offset = _detect_tz_offset(transcript)
    now_utc   = datetime.now(timezone.utc)

    def _to_utc(hour: int, minute: int) -> datetime:
        naive  = now_utc.replace(tzinfo=None).replace(hour=hour, minute=minute, second=0, microsecond=0)
        utc_dt = (naive - timedelta(hours=tz_offset)).replace(tzinfo=timezone.utc)
        if utc_dt < now_utc: utc_dt += timedelta(days=1)
        return utc_dt

    for ai in action_items:
        deadline = str(ai.get("deadline", "")).lower()
        m = re.search(r"(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)", deadline, re.IGNORECASE)
        if m:
            h = int(m.group(1)); mins = int(m.group(2) or 0); ampm = m.group(3).lower()
            if ampm == "pm" and h != 12: h += 12
            elif ampm == "am" and h == 12: h = 0
            try: return _to_utc(h, mins)
            except ValueError: pass

    for pattern in [r"(?:at|by|@)\s*(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)",
                    r"(\d{1,2})\s*(am|pm)\s*(?:today|tonight|sharp)",
                    r"(\d{1,2}):(\d{2})\s*(am|pm)"]:
        m = re.search(pattern, transcript, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g]
            try:
                if len(groups) >= 2 and groups[-1].lower() in ("am", "pm"):
                    h = int(groups[0]); ampm = groups[-1].lower()
                    mins = int(groups[1]) if len(groups) >= 3 else 0
                    if ampm == "pm" and h != 12: h += 12
                    elif ampm == "am" and h == 12: h = 0
                    return _to_utc(h, mins)
            except (ValueError, IndexError): continue

    return now_utc + timedelta(hours=1)


def _generate_ics(meeting_name: str, action_items: list, meeting_dt: datetime) -> bytes:
    """FIX-3: DTSTART always emitted as UTC with Z suffix."""
    if meeting_dt.tzinfo is None:
        meeting_dt = meeting_dt.replace(tzinfo=timezone.utc)
    else:
        meeting_dt = meeting_dt.astimezone(timezone.utc)
    meeting_end = meeting_dt + timedelta(hours=1)
    fmt = "%Y%m%dT%H%M%SZ"
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0",
        "PRODID:-//Meeting Summarizer Agent//EN",
        "BEGIN:VEVENT",
        f"DTSTART:{meeting_dt.strftime(fmt)}",
        f"DTEND:{meeting_end.strftime(fmt)}",
        f"SUMMARY:Meeting Summary - {meeting_name}",
        "END:VEVENT",
    ]
    for ai in action_items:
        lines += [
            "BEGIN:VTODO",
            f"SUMMARY:[{ai.get('owner','TBD')}] {str(ai.get('task',''))[:200]}",
            f"DESCRIPTION:Deadline: {ai.get('deadline','TBD')}",
            "END:VTODO",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines).encode("utf-8")


def _generate_csv(meeting_name: str, action_items: list, triage: list = None) -> bytes:
    import io as _io
    triage_map = {}
    if triage:
        for t in triage:
            triage_map[t.get("task_key", "")] = t
    buf = _io.StringIO()
    w   = csv_mod.DictWriter(
        buf,
        fieldnames=["meeting","poc_name","task_description","deadline",
                    "priority","risk","discussed_at_sec"],
        lineterminator="\r\n",
    )
    w.writeheader()
    for ai in action_items:
        key    = str(ai.get("task", ""))[:80].lower().strip()
        triage_row = triage_map.get(key, {})
        w.writerow({
            "meeting":          meeting_name,
            "poc_name":         str(ai.get("owner",    "TBD")),
            "task_description": str(ai.get("task",     "")),
            "deadline":         str(ai.get("deadline", "TBD")),
            "priority":         triage_row.get("final_priority", ""),
            "risk":             triage_row.get("risk_summary",   ""),
            "discussed_at_sec": str(ai.get("discussed_at_sec", 0.0)),
        })
    return buf.getvalue().encode("utf-8")


def _build_summary_text(parsed: dict) -> str:
    parts = []
    for key, label in [("abstract","ABSTRACT"),("decisions","DECISIONS"),
                       ("problems","PROBLEMS"),("actions","ACTIONS")]:
        if parsed.get(key):
            parts.append(f"{label}:\n{parsed[key]}")
    return "\n\n".join(parts)


def _build_html_email(file_name: str, summary_text: str, triage: list = None) -> str:
    body_html = summary_text.replace("\n", "<br>")
    triage_html = ""
    if triage:
        rows = ""
        for t in triage:
            emoji = PRIORITY_EMOJI.get(t.get("final_priority", ""), "⚪")
            rows += (
                f"<tr><td>{emoji} {t.get('final_priority','')}</td>"
                f"<td>{t.get('owner','')}</td>"
                f"<td>{t.get('task','')}</td>"
                f"<td>{t.get('risk_summary','')}</td></tr>"
            )
        triage_html = f"""
        <h3 style="color:#2c3e50;">Action Item Triage</h3>
        <table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">
          <tr style="background:#ecf0f1">
            <th>Priority</th><th>Owner</th><th>Task</th><th>Risk if Delayed</th>
          </tr>
          {rows}
        </table><br>"""
    return f"""<html><body style="font-family:Arial,sans-serif;max-width:700px;margin:auto;padding:20px;">
      <h2 style="color:#2c3e50;">Meeting Summary</h2>
      <h3 style="color:#7f8c8d;">{file_name}</h3>
      <hr style="border:1px solid #ecf0f1;">
      {triage_html}
      <div style="line-height:1.8;color:#34495e;">{body_html}</div>
      <hr style="border:1px solid #ecf0f1;">
      <p style="color:#95a5a6;font-size:12px;">
        Generated by Meeting Summarizer Agent (Qwen2.5-14B + CoD Triage)
      </p></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# CoD Triage — run_triage_cod()
# Analogous to calendar_cod.py's _run_cod()
# ─────────────────────────────────────────────────────────────────────────────

def _build_triage_context(action_item: dict, transcript_snippet: str,
                           all_actions: list, problems: str) -> str:
    return (
        f"=== MEETING CONTEXT ===\n"
        f"Problems/risks raised in meeting:\n{problems or 'None identified.'}\n\n"
        f"All action items:\n" +
        "\n".join(f"- [{a.get('owner','?')}] {a.get('task','')} (Due: {a.get('deadline','TBD')})"
                  for a in all_actions) +
        f"\n\n=== ACTION ITEM TO TRIAGE ===\n"
        f"Owner:    {action_item.get('owner', 'TBD')}\n"
        f"Task:     {action_item.get('task', '')}\n"
        f"Deadline: {action_item.get('deadline', 'TBD')}\n\n"
        f"=== RELEVANT TRANSCRIPT EXCERPT ===\n{transcript_snippet[:600]}"
    )


def _run_triage_cod(action_item: dict, transcript: str,
                    all_actions: list, problems: str) -> Optional[dict]:
    """
    3-round CoD (Proposer → Challenger → Judge) to classify one action item.
    Returns a dict with final_priority, risk_summary, deadline_note, rationale.
    Falls back gracefully — never blocks the pipeline.
    """
    # Find the most relevant transcript excerpt (look for owner name or task keywords)
    owner = action_item.get("owner", "")
    task_words = str(action_item.get("task", "")).split()[:3]
    search_terms = [owner] + task_words
    best_excerpt = transcript[:600]  # fallback to transcript start
    for term in search_terms:
        if term and term.lower() in transcript.lower():
            idx = transcript.lower().find(term.lower())
            best_excerpt = transcript[max(0, idx - 100): idx + 500]
            break

    context = _build_triage_context(action_item, best_excerpt, all_actions, problems)

    try:
        # ── Round 1: Proposer ────────────────────────────────────────────────
        proposal: ActionPriorityProposal = _llm(
            structured_output=ActionPriorityProposal, model_name=MODEL
        ).invoke([
            SystemMessage(content=(
                "You are the Proposer in a meeting action item triage debate.\n"
                "Classify the action item's priority based on meeting context.\n\n"
                "Priority levels:\n"
                "  Critical — blocks the team or a release; must happen today/immediately\n"
                "  High     — significant risk if missed this week; clearly time-sensitive\n"
                "  Medium   — important but flexible; can slip a day or two\n"
                "  Low      — nice to have; no immediate risk if delayed\n\n"
                "Rules:\n"
                "  - Base priority on impact to the team, not just deadline text\n"
                "  - If the meeting raised this as a blocker or critical risk, it is Critical\n"
                "  - Consider the deadline: 'EOD today' is more urgent than 'next Tuesday'\n"
                "Return an ActionPriorityProposal."
            )),
            HumanMessage(content=context),
        ])
        logger.info("Triage proposer [%s]: %s | %s",
                    action_item.get("owner"), proposal.priority, proposal.argument[:80])

        # ── Round 2: Challenger ──────────────────────────────────────────────
        challenge: PriorityChallenge = _llm(
            structured_output=PriorityChallenge, model_name=MODEL
        ).invoke([
            SystemMessage(content=(
                "You are the Challenger in a meeting action item triage debate.\n"
                "Review the Proposer's priority classification.\n\n"
                "Challenge if any of these apply:\n"
                "  - Proposer ignored a clear blocker signal from the meeting\n"
                "  - Proposer over-inflated priority for a routine task\n"
                "  - Deadline urgency was not properly weighed (EOD vs next week)\n"
                "  - Other action items in context suggest this should rank differently\n"
                "Agree if the classification is genuinely correct.\n"
                "Return a PriorityChallenge."
            )),
            HumanMessage(content=(
                f"{context}\n\n"
                f"Proposer said: {proposal.priority}\n"
                f"Proposer argument: {proposal.argument}"
            )),
        ])
        logger.info("Triage challenger [%s]: agrees=%s counter=%s",
                    action_item.get("owner"), challenge.agrees, challenge.counter_priority)

        # ── Round 3: Judge ───────────────────────────────────────────────────
        verdict: TriageVerdict = _llm(
            structured_output=TriageVerdict, model_name=MODEL
        ).invoke([
            SystemMessage(content=(
                "You are the Judge in a meeting action item triage debate.\n"
                "Make the final priority decision.\n\n"
                "Rules:\n"
                "  - final_priority must be one of: Critical, High, Medium, Low\n"
                "  - Weigh both the Proposer and Challenger arguments\n"
                "  - risk_summary: one sentence on what happens if this item is missed\n"
                "  - deadline_note: 'On track' if deadline is realistic, else a brief note\n"
                "  - rationale: one sentence explaining your final choice\n"
                "Return a TriageVerdict."
            )),
            HumanMessage(content=(
                f"{context}\n\n"
                f"Proposer: {proposal.priority} — {proposal.argument}\n"
                f"Challenger {'agreed' if challenge.agrees else 'countered with'}: "
                f"{challenge.counter_priority} — {challenge.argument}"
            )),
        ])
        logger.info("Triage verdict [%s]: %s | %s",
                    action_item.get("owner"), verdict.final_priority, verdict.rationale)

        return {
            "owner":          action_item.get("owner", "TBD"),
            "task":           action_item.get("task", ""),
            "task_key":       str(action_item.get("task", ""))[:80].lower().strip(),
            "deadline":       action_item.get("deadline", "TBD"),
            "final_priority": verdict.final_priority,
            "risk_summary":   verdict.risk_summary,
            "deadline_note":  verdict.deadline_note,
            "rationale":      verdict.rationale,
        }

    except Exception as e:
        logger.error("CoD triage failed for [%s]: %s", action_item.get("owner"), e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry router
# ─────────────────────────────────────────────────────────────────────────────
def route_meeting_entry(state: OrchestratorState) -> Literal[
    "meeting_fetch_transcript", "meeting_send_email", "meeting_post_cancel"
]:
    action = state.slack_action_id or ""
    if action == "confirm_summary":  return "meeting_send_email"
    if action == "cancel_summary":   return "meeting_post_cancel"
    return "meeting_fetch_transcript"


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: meeting_fetch_transcript
# ─────────────────────────────────────────────────────────────────────────────
def meeting_fetch_transcript(state: OrchestratorState) -> OrchestratorState:
    """FIX-1: S3 lock + Google Drive download."""
    file_id   = state.transcript_file_id
    file_name = state.transcript_file_name or "transcript.txt"

    if not file_id:
        return state.model_copy(update={"error": "missing file_id"})

    s3 = boto3.client("s3")
    if not _acquire_s3_lock(s3, file_id):
        return state.model_copy(update={"error": "duplicate_trigger"})

    try:
        raw_text = _download_from_drive(file_id)
        logger.info("Downloaded %s: %d chars", file_name, len(raw_text))
        return state.model_copy(update={"transcript_text": raw_text})
    except Exception as e:
        _release_s3_lock(s3, file_id, status="error")
        logger.error("Drive download failed: %s", e)
        return state.model_copy(update={"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: meeting_preprocess
# ─────────────────────────────────────────────────────────────────────────────
def meeting_preprocess(state: OrchestratorState) -> OrchestratorState:
    if state.error: return state
    raw = state.transcript_text or ""
    if not raw: return state.model_copy(update={"error": "empty transcript"})
    return state.model_copy(update={"transcript_processed": _preprocess(raw)})


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: meeting_summarize
# ─────────────────────────────────────────────────────────────────────────────
def meeting_summarize(state: OrchestratorState) -> OrchestratorState:
    if state.error: return state
    transcript = state.transcript_processed or state.transcript_text or ""
    if not transcript: return state.model_copy(update={"error": "no transcript to summarize"})
    try:
        raw_output = _run_chunked_inference(transcript)
        parsed     = _parse_model_output(raw_output)
        logger.info("Summarized: %d action items", len(parsed.get("actions_json", [])))

        # ── Save prompt+response for RLHF feedback logging later ───────
        prompt_log = json.dumps({"transcript_preview": transcript[:500],
                                  "file_name": state.transcript_file_name})
        response_log = json.dumps({
            "abstract": parsed.get("abstract", "")[:500],
            "n_actions": len(parsed.get("actions_json", [])),
            "decisions": parsed.get("decisions", "")[:300],
        })

        return state.model_copy(update={
            "meeting_summary_parsed": parsed,
            "llm_prompt_log": prompt_log,
            "llm_response_log": response_log,
        })
    except Exception as e:
        logger.error("Summarize failed: %s", e)
        return state.model_copy(update={"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: meeting_triage_cod  ← Chain-of-Debate node
# ─────────────────────────────────────────────────────────────────────────────
def meeting_triage_cod(state: OrchestratorState) -> OrchestratorState:
    """
    Run CoD triage on each extracted action item.
    Analogous to slot_cod in calendar_cod.py — structured debate to classify
    priority and risk before the approver sees the Slack card.

    Fails gracefully — if CoD errors, pipeline continues without triage results
    and the Slack card shows the summary without colour-coded priority.
    """
    if state.error: return state
    parsed  = state.meeting_summary_parsed or {}
    actions = parsed.get("actions_json", [])
    problems = parsed.get("problems", "")

    if not actions:
        logger.info("No action items to triage")
        return state.model_copy(update={"meeting_triage": []})

    transcript = state.transcript_processed or state.transcript_text or ""
    triage_results = []

    logger.info("Starting CoD triage for %d action item(s)", len(actions))
    for ai in actions:
        result = _run_triage_cod(ai, transcript, actions, problems)
        if result:
            triage_results.append(result)
        else:
            # Fallback entry — CoD failed for this item, use Medium as safe default
            triage_results.append({
                "owner":          ai.get("owner", "TBD"),
                "task":           ai.get("task", ""),
                "task_key":       str(ai.get("task", ""))[:80].lower().strip(),
                "deadline":       ai.get("deadline", "TBD"),
                "final_priority": "Medium",
                "risk_summary":   "Triage unavailable.",
                "deadline_note":  "On track",
                "rationale":      "CoD triage failed — defaulting to Medium.",
            })

    logger.info("CoD triage complete: %d item(s) classified", len(triage_results))
    return state.model_copy(update={"meeting_triage": triage_results})


# ─────────────────────────────────────────────────────────────────────────────
# Node 5: meeting_generate_artifacts
# ─────────────────────────────────────────────────────────────────────────────
def meeting_generate_artifacts(state: OrchestratorState) -> OrchestratorState:
    if state.error: return state
    parsed    = state.meeting_summary_parsed or {}
    file_name = state.transcript_file_name or "transcript.txt"
    raw_text  = state.transcript_text or ""
    actions   = parsed.get("actions_json", [])
    triage    = state.meeting_triage or []

    ics_bytes = csv_bytes = None
    if actions:
        meeting_dt = _parse_meeting_datetime(raw_text, actions)
        logger.info("Meeting UTC: %s", meeting_dt.isoformat())
        ics_bytes  = _generate_ics(file_name, actions, meeting_dt)
    csv_bytes = _generate_csv(file_name, actions, triage)
    return state.model_copy(update={"meeting_ics_bytes": ics_bytes, "meeting_csv_bytes": csv_bytes})


# ─────────────────────────────────────────────────────────────────────────────
# Node 6: meeting_store_s3
# ─────────────────────────────────────────────────────────────────────────────
def meeting_store_s3(state: OrchestratorState) -> OrchestratorState:
    if state.error: return state
    parsed    = state.meeting_summary_parsed or {}
    file_name = state.transcript_file_name or "transcript.txt"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = file_name.replace(".txt", "").replace(" ", "_")
    s3_key    = f"{S3_PREFIX}/meetings/{timestamp}_{base_name}"
    s3        = boto3.client("s3")

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/transcript.txt",
                      Body=(state.transcript_text or "").encode(), ContentType="text/plain")
        s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/raw_output.txt",
                      Body=(parsed.get("raw_output", "")).encode(), ContentType="text/plain")
        summary_text = _build_summary_text(parsed)
        meta = {
            "file_name":     file_name, "s3_key": s3_key, "timestamp": timestamp,
            "summary_text":  summary_text, "abstract": parsed.get("abstract", ""),
            "decisions":     parsed.get("decisions", ""), "problems": parsed.get("problems", ""),
            "actions":       parsed.get("actions", ""), "triage": state.meeting_triage or [],
            "actions_count": len(parsed.get("actions_json", [])),
        }
        s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/meta.json",
                      Body=json.dumps(meta, indent=2).encode(), ContentType="application/json")
        if state.meeting_ics_bytes:
            s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/invite.ics",
                          Body=state.meeting_ics_bytes, ContentType="text/calendar")
        if state.meeting_csv_bytes:
            s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/actions.csv",
                          Body=state.meeting_csv_bytes, ContentType="text/csv")
        logger.info("Stored s3://%s/%s/", S3_BUCKET, s3_key)
        if state.transcript_file_id:
            _release_s3_lock(s3, state.transcript_file_id, status="done")
        return state.model_copy(update={"meeting_s3_key": s3_key})
    except Exception as e:
        logger.error("S3 store failed: %s", e)
        return state.model_copy(update={"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Node 7: meeting_post_slack
# ─────────────────────────────────────────────────────────────────────────────
def meeting_post_slack(state: OrchestratorState) -> OrchestratorState:
    if state.error and state.error != "duplicate_trigger":
        logger.error("Skipping Slack post due to error: %s", state.error)
        return state
    if state.error == "duplicate_trigger":
        return state

    parsed    = state.meeting_summary_parsed or {}
    file_name = state.transcript_file_name or "transcript.txt"
    s3_key    = state.meeting_s3_key or ""
    triage    = state.meeting_triage or []

    abstract  = parsed.get("abstract",  "No abstract generated.")[:300]
    decisions = parsed.get("decisions", "None.") or "None."
    problems  = parsed.get("problems",  "None.") or "None."
    n_actions = len(parsed.get("actions_json", []))
    has_ics   = state.meeting_ics_bytes is not None

    # Build triage section for Slack
    triage_text = ""
    if triage:
        lines = []
        for t in sorted(triage, key=lambda x: ["Critical","High","Medium","Low"].index(
                x.get("final_priority","Low") if x.get("final_priority","Low") in
                ["Critical","High","Medium","Low"] else "Low")):
            emoji = PRIORITY_EMOJI.get(t.get("final_priority", "Medium"), "⚪")
            lines.append(
                f"{emoji} *{t.get('final_priority','')}* — "
                f"[{t.get('owner','')}] {t.get('task','')[:60]}\n"
                f"   _{t.get('risk_summary','')}_ | {t.get('deadline_note','')}"
            )
        triage_text = "\n".join(lines)

    blocks = [
        {"type": "header",  "text": {"type": "plain_text", "text": f"Meeting Summary: {file_name}"}},
        {"type": "section", "text": {"type": "mrkdwn",     "text": f"*Abstract*\n{abstract}"}},
        {"type": "divider"},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Decisions*\n{decisions[:400]}"},
            {"type": "mrkdwn", "text": f"*Problems / Risks*\n{problems[:400]}"},
        ]},
        {"type": "divider"},
    ]

    # Triage block (only if CoD produced results)
    if triage_text:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Action Item Triage ({n_actions})*\n{triage_text[:800]}"}
        })
        blocks.append({"type": "divider"})
    else:
        actions_text = parsed.get("actions", "None.") or "None."
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": (
                f"*Action Items ({n_actions})*\n{actions_text[:500]}"
                + ("\n\n_ICS calendar invite included._" if has_ics else "")
            )}
        })
        blocks.append({"type": "divider"})

    blocks += [
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            f"*Ready to send to:* {', '.join(PARTICIPANT_EMAILS)}\n"
            "Click *Confirm* to email summary + triage + ICS + CSV.\n"
            "Click *Cancel* to dismiss."
        )}},
        {"type": "actions", "elements": [
            {
                "type": "button", "style": "primary",
                "text":      {"type": "plain_text", "text": "Confirm - Send Email"},
                "value":     json.dumps({"file_name": file_name, "s3_key": s3_key}),
                "action_id": "confirm_summary",
            },
            {
                "type": "button", "style": "danger",
                "text":      {"type": "plain_text", "text": "Cancel"},
                "action_id": "cancel_summary",
            },
        ]},
    ]

    try:
        slack   = WebClient(token=SLACK_BOT_TOKEN)
        channel = SLACK_NOTIFY_CHANNEL or os.environ.get("SLACK_CHANNEL_ID", "")
        resp    = slack.chat_postMessage(
            channel=channel, blocks=blocks,
            text=f"New meeting summary ready: {file_name}",
        )
        logger.info("Posted to Slack channel=%s ts=%s", channel, resp["ts"])
    except Exception as e:
        logger.error("Slack post failed: %s", e)
        return state.model_copy(update={"error": str(e)})
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 8a: meeting_send_email
# ─────────────────────────────────────────────────────────────────────────────
def meeting_send_email(state: OrchestratorState) -> OrchestratorState:
    import email.mime.multipart as mp
    import email.mime.text      as mt
    import email.mime.base      as mb
    import email.encoders       as encoders

    value_dict = state.slack_action_value or {}
    file_name  = value_dict.get("file_name", "meeting")
    s3_key     = value_dict.get("s3_key", "")
    channel_id = state.channel_id
    preview_ts = state.preview_ts
    slack      = WebClient(token=SLACK_BOT_TOKEN)

    try:
        s3   = boto3.client("s3")
        meta = json.loads(
            s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/meta.json")["Body"].read()
        )
        ics_bytes = csv_bytes = None
        try: ics_bytes = s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/invite.ics")["Body"].read()
        except Exception: pass
        try: csv_bytes = s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/actions.csv")["Body"].read()
        except Exception: pass

        triage        = meta.get("triage", [])
        summary_text  = meta.get("summary_text", "")
        ses = boto3.client("ses", region_name="us-east-1")
        msg = mp.MIMEMultipart("mixed")
        msg["Subject"] = f"Meeting Summary: {file_name}"
        msg["From"]    = SES_FROM_EMAIL
        msg["To"]      = ", ".join(PARTICIPANT_EMAILS)
        msg.attach(mt.MIMEText(_build_html_email(file_name, summary_text, triage), "html"))

        if ics_bytes:
            part = mb.MIMEBase("text", "calendar", method="REQUEST", name="invite.ics")
            part.set_payload(ics_bytes); encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename="invite.ics")
            msg.attach(part)
        if csv_bytes:
            part = mb.MIMEBase("text", "csv", name="action_items.csv")
            part.set_payload(csv_bytes); encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename="action_items.csv")
            msg.attach(part)

        ses.send_raw_email(
            Source=SES_FROM_EMAIL, Destinations=PARTICIPANT_EMAILS,
            RawMessage={"Data": msg.as_string()},
        )
        logger.info("Email sent to %s", PARTICIPANT_EMAILS)

        # ── RLHF: log confirmed feedback ──────────────────────────────────
        if state.llm_prompt_log:
            try:
                log_feedback(
                    flow="meeting",
                    prompt=state.llm_prompt_log,
                    response=state.llm_response_log or "",
                    human_decision="approved",
                    metadata={"file_name": file_name, "s3_key": s3_key},
                )
            except Exception as fb_err:
                logger.warning("RLHF feedback log failed (approved): %s", fb_err)

        if channel_id and preview_ts:
            slack.chat_update(
                channel=channel_id, ts=preview_ts, blocks=None,
                text=f"Summary confirmed. Email sent to: {', '.join(PARTICIPANT_EMAILS)}",
            )
    except Exception as e:
        logger.error("Send email failed: %s", e)
        if channel_id and preview_ts:
            try: slack.chat_update(channel=channel_id, ts=preview_ts,
                                   text=f"Error sending email: {e}", blocks=None)
            except Exception: pass
        return state.model_copy(update={"error": str(e)})
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Node 8b: meeting_post_cancel
# ─────────────────────────────────────────────────────────────────────────────
def meeting_post_cancel(state: OrchestratorState) -> OrchestratorState:
    # ── RLHF: log cancelled feedback ────────────────────────────────────
    if state.llm_prompt_log:
        try:
            log_feedback(
                flow="meeting",
                prompt=state.llm_prompt_log,
                response=state.llm_response_log or "",
                human_decision="cancelled",
                metadata={"channel": state.channel_id},
            )
        except Exception as e:
            logger.warning("RLHF feedback log failed (cancelled): %s", e)

    if state.channel_id and state.preview_ts:
        try:
            WebClient(token=SLACK_BOT_TOKEN).chat_update(
                channel=state.channel_id, ts=state.preview_ts, blocks=None,
                text="Meeting summary dismissed. No email sent.",
            )
        except Exception as e:
            return state.model_copy(update={"error": str(e)})
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Subgraph builder
# ─────────────────────────────────────────────────────────────────────────────
def build_meeting_subgraph() -> StateGraph:
    """
    New transcript flow:
      fetch → preprocess → summarize → triage_cod → artifacts → s3 → slack → END

    Button click flow:
      send_email → END   (confirm_summary)
      post_cancel → END  (cancel_summary)
    """
    b = StateGraph(OrchestratorState)

    b.add_node("meeting_fetch_transcript",   meeting_fetch_transcript)
    b.add_node("meeting_preprocess",         meeting_preprocess)
    b.add_node("meeting_summarize",          meeting_summarize)
    b.add_node("meeting_triage_cod",         meeting_triage_cod)
    b.add_node("meeting_generate_artifacts", meeting_generate_artifacts)
    b.add_node("meeting_store_s3",           meeting_store_s3)
    b.add_node("meeting_post_slack",         meeting_post_slack)
    b.add_node("meeting_send_email",         meeting_send_email)
    b.add_node("meeting_post_cancel",        meeting_post_cancel)

    b.add_conditional_edges(START, route_meeting_entry, {
        "meeting_fetch_transcript": "meeting_fetch_transcript",
        "meeting_send_email":       "meeting_send_email",
        "meeting_post_cancel":      "meeting_post_cancel",
    })

    # New transcript pipeline
    b.add_edge("meeting_fetch_transcript",   "meeting_preprocess")
    b.add_edge("meeting_preprocess",         "meeting_summarize")
    b.add_edge("meeting_summarize",          "meeting_triage_cod")
    b.add_edge("meeting_triage_cod",         "meeting_generate_artifacts")
    b.add_edge("meeting_generate_artifacts", "meeting_store_s3")
    b.add_edge("meeting_store_s3",           "meeting_post_slack")
    b.add_edge("meeting_post_slack",         END)

    # Button flows
    b.add_edge("meeting_send_email",  END)
    b.add_edge("meeting_post_cancel", END)

    return b.compile()