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
                                  Fallback: parses actions from plain text if ACTIONS_JSON empty
  5. meeting_jira_cod             Chain-of-Debate: detect which action items should
                                  become Jira tickets (type, summary, assignee)
  6. meeting_generate_artifacts   ICS UTC-correct (FIX-3) + CSV
  7. meeting_store_s3             All artifacts to S3, sets meeting_s3_key
  8. meeting_post_slack           Block Kit card with triage + Jira proposals,
                                  Confirm/Cancel buttons

Button click nodes:
  9a. meeting_send_email          Fetch artifacts from S3, send via SES, then kick
                                  off the Jira proposal queue
  9b. meeting_post_cancel         Update Slack message to dismissed

Jira queue nodes (one proposal at a time):
  10. meeting_post_next_jira      Post next pending Jira proposal card from Redis queue
  11. meeting_create_jira         Create the Jira ticket, advance queue
      (skip_meeting_jira routes directly back to meeting_post_next_jira via the router)

CoD patterns:
  - triage_cod  : Proposer -> Challenger -> Judge per action item (priority/risk)
  - jira_cod    : Proposer -> Challenger -> Judge per action item (should it be a Jira ticket?)

Stateless button design (same pattern as calendar_agent):
  Session data stored in Redis via save_session(); only session_id in button value.
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
import requests
from botocore.exceptions import ClientError
from requests.auth import HTTPBasicAuth
from slack_sdk import WebClient
from google.oauth2 import service_account
from googleapiclient.discovery import build as gdrive_build
from googleapiclient.http import MediaIoBaseDownload
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from redis_store import save_session, load_session, record_feedback

from state import (
    OrchestratorState,
    S3_BUCKET, S3_PREFIX, VLLM_MODEL_NAME, EC2_IP,
    SLACK_BOT_TOKEN, SLACK_NOTIFY_CHANNEL,
    SES_FROM_EMAIL, PARTICIPANT_EMAILS, EMAIL_AGENT_INBOX, _llm,
    JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN,
    JIRA_PROJECT_KEY, JIRA_ISSUE_TYPE, TEAM_MAP,
)

logger = logging.getLogger(__name__)

# Constants
SCOPES           = ["https://www.googleapis.com/auth/drive.readonly"]
CHUNK_SIZE       = 10000
CHUNK_OVERLAP    = 1000
LOCK_TTL_SECONDS = 1800

TZ_OFFSETS = {
    "PDT": -7, "PST": -8, "MDT": -6, "MST": -7, "CDT": -5, "CST": -6,
    "EDT": -4, "EST": -5, "BST": +1, "GMT":  0, "UTC": 0,
    "CET": +1, "CEST": +2, "IST": +5.5, "JST": +9, "KST": +9,
    "AEST": +10, "AEDT": +11,
}
_TZ_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in TZ_OFFSETS) + r")\b")
_ACTIONS_JSON_RE = re.compile(r"ACTIONS_JSON:\s*(\[.*?\])", re.DOTALL | re.IGNORECASE)
_SECTION_RE = re.compile(
    r"(ABSTRACT|DECISIONS|PROBLEMS|ACTIONS):\s*(.*?)(?=\n(?:ABSTRACT|DECISIONS|PROBLEMS|ACTIONS|ACTIONS_JSON):|$)",
    re.DOTALL | re.IGNORECASE,
)

PRIORITY_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
TICKET_TYPE_EMOJI = {"Bug": "🐛", "Story": "📖", "Task": "✅"}
MODEL = "meeting"


# =============================================================================
# Pydantic schemas — Triage CoD
# =============================================================================

class ActionPriorityProposal(BaseModel):
    priority:    str  = Field(description="Critical | High | Medium | Low")
    risk:        str  = Field(description="1-sentence risk if this action item is missed or delayed")
    deadline_ok: bool = Field(description="True if stated deadline is realistic given meeting context")
    argument:    str  = Field(description="1-2 sentence justification for the priority level")

class PriorityChallenge(BaseModel):
    agrees:           bool = Field(description="True if challenger agrees with proposed priority")
    counter_priority: str  = Field(description="Challenger's suggested priority, or same if agreeing")
    argument:         str  = Field(description="1-2 sentence challenge or concession")

class TriageVerdict(BaseModel):
    final_priority: str  = Field(description="Critical | High | Medium | Low")
    risk_summary:   str  = Field(description="One sentence: risk if item is delayed or missed")
    deadline_note:  str  = Field(description="'On track' or short note if deadline seems unrealistic")
    rationale:      str  = Field(description="One sentence explaining the priority choice")


# =============================================================================
# Pydantic schemas — Jira CoD
# =============================================================================

class JiraProposalProposal(BaseModel):
    should_create: bool = Field(description=(
        "True if this action item warrants a Jira ticket. "
        "Warrant: concrete task with identifiable owner, trackable deliverable. "
        "Do not warrant: vague notes, internal decisions, or items with no follow-up work."
    ))
    summary:     str = Field(description="Concise Jira ticket summary (max 120 chars). Empty if should_create=False.")
    ticket_type: str = Field(description="Task | Bug | Story. Empty if should_create=False.")
    argument:    str = Field(description="1-2 sentence justification for the decision.")

class JiraProposalChallenge(BaseModel):
    agrees:                bool = Field(description="True if challenger agrees.")
    counter_should_create: bool = Field(description="Challenger's suggested decision if disagreeing.")
    argument:              str  = Field(description="1-2 sentence challenge or concession.")

class JiraProposalVerdict(BaseModel):
    create_ticket: bool = Field(description="Final decision: True to propose as a Jira ticket.")
    summary:       str  = Field(description="Final Jira summary (max 120 chars). Empty if create_ticket=False.")
    ticket_type:   str  = Field(description="Task | Bug | Story. Empty if create_ticket=False.")
    rationale:     str  = Field(description="One sentence: why this should/shouldn't be tracked in Jira.")


# =============================================================================
# Summarization prompts
# =============================================================================

P4_SYSTEM_PROMPT = (
    "You are a professional meeting minutes assistant. "
    "Your output MUST contain exactly these five sections in this exact order: "
    "ABSTRACT, DECISIONS, PROBLEMS, ACTIONS, ACTIONS_JSON. "
    "The ACTIONS_JSON section is MANDATORY and MUST be a valid JSON array — never omit it. "
    "Never output plain text where JSON is required. "
    "Never stop before outputting ACTIONS_JSON."
)

P4_USER_INSTRUCTION = (
    "Given the meeting transcript below, produce all five sections exactly as shown.\n\n"
    "ABSTRACT: A concise paragraph (2-4 sentences) summarising the meeting.\n\n"
    "DECISIONS: Bullet list of decisions made. Use '- ' prefix per item.\n\n"
    "PROBLEMS: Bullet list of problems or risks raised. Use '- ' prefix per item.\n\n"
    "ACTIONS: Bullet list of action items. Format each as: - [Owner Name] Task description - Due: deadline\n\n"
    "ACTIONS_JSON: A JSON array where EVERY action item from ACTIONS appears as an object.\n"
    "Each object MUST have these exact keys: owner, task, deadline, discussed_at_sec\n"
    "Rules:\n"
    "  - owner: full name string, use TBD if not mentioned\n"
    "  - task: clear action description string\n"
    "  - deadline: date or relative string, use TBD if not mentioned\n"
    "  - discussed_at_sec: seconds elapsed from the START of the meeting when this item was discussed.\n"
    "    If the transcript has timestamps like [14:02] or 14:02, convert to seconds from the first timestamp.\n"
    "    Example: if meeting starts at 14:00 and item discussed at 14:07, discussed_at_sec = 420.0\n"
    "    If no timestamps in transcript, use 0.0\n"
    "  - discussed_at_sec MUST be a number (float), NEVER an epoch timestamp like 1684728000.0\n\n"
    "EXAMPLE of correct ACTIONS_JSON output:\n"
    "ACTIONS_JSON:\n"
    "[\n"
    "  {\"owner\": \"Alice Smith\", \"task\": \"Fix login bug\", \"deadline\": \"Friday\", \"discussed_at_sec\": 120.0},\n"
    "  {\"owner\": \"Bob Jones\", \"task\": \"Update documentation\", \"deadline\": \"TBD\", \"discussed_at_sec\": 0.0}\n"
    "]\n\n"
    "CRITICAL RULES:\n"
    "  1. ACTIONS_JSON is NOT optional — always output it even if ACTIONS is empty (use [])\n"
    "  2. The JSON array must be valid — use double quotes, no trailing commas\n"
    "  3. Number of items in ACTIONS_JSON must match number of items in ACTIONS\n"
    "  4. Do not fabricate facts not in the transcript\n"
    "  5. Output ACTIONS_JSON as the LAST section — never stop before it\n"
)
MERGE_SYSTEM_PROMPT = (
    "You are a professional meeting minutes assistant. "
    "Merge the partial abstracts below into a single coherent paragraph of 80-200 words. "
    "Do not add facts not in the input. Output only the merged paragraph, no headers."
)

def _chunk_prompt(transcript):
    return [
        {"role": "system", "content": P4_SYSTEM_PROMPT},
        {"role": "user",   "content": f"{P4_USER_INSTRUCTION}\n\nTRANSCRIPT:\n{transcript}"},
    ]

def _merge_prompt(abstracts):
    joined = "\n\n---\n\n".join(f"Part {i+1}:\n{a}" for i, a in enumerate(abstracts))
    return [
        {"role": "system", "content": MERGE_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Partial abstracts:\n\n{joined}"},
    ]


# =============================================================================
# S3 lock helpers (FIX-1)
# =============================================================================

def _lock_key(file_id):
    return f"{S3_PREFIX}/locks/{file_id}.lock"

def _acquire_s3_lock(s3, file_id):
    key = _lock_key(file_id)
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body = json.loads(obj["Body"].read())
        if time.time() - body.get("ts", 0) < LOCK_TTL_SECONDS:
            return False
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("NoSuchKey", "404"):
            raise
    s3.put_object(Bucket=S3_BUCKET, Key=key,
                  Body=json.dumps({"ts": time.time(), "status": "processing"}).encode(),
                  ContentType="application/json")
    return True

def _release_s3_lock(s3, file_id, status="done"):
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=_lock_key(file_id),
                      Body=json.dumps({"ts": time.time(), "status": status}).encode(),
                      ContentType="application/json")
    except Exception as e:
        logger.warning("Could not update lock: %s", e)


# =============================================================================
# Google Drive download
# =============================================================================

def _get_service_account_info() -> dict:
    """Load Google service account JSON from S3 (avoids 4KB Lambda env var limit).
    Falls back to GOOGLE_SERVICE_ACCOUNT_JSON env var if S3 read fails.
    """
    try:
        s3  = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key="config/google-sa.json")
        sa_info = json.loads(obj["Body"].read().decode("utf-8"))
        logger.info("_get_service_account_info: loaded from S3")
        return sa_info
    except Exception as e:
        logger.warning("_get_service_account_info: S3 load failed (%s), falling back to env var", e)
        sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
        return json.loads(sa_json)


def _download_from_drive(file_id):
    sa_info = _get_service_account_info()
    creds   = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    svc     = gdrive_build("drive", "v3", credentials=creds)
    req     = svc.files().get_media(fileId=file_id)
    fh      = io.BytesIO()
    dl      = MediaIoBaseDownload(fh, req)
    done    = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue().decode("utf-8")


# =============================================================================
# Preprocessor
# =============================================================================

def _preprocess(transcript):
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


# =============================================================================
# vLLM call + inference helpers
# =============================================================================

def _vllm_call(messages, max_new_tokens=1536):
    resp = requests.post(
        f"http://{EC2_IP}:8000/v1/chat/completions",
        json={"model": VLLM_MODEL_NAME, "messages": messages,
              "max_tokens": max_new_tokens, "temperature": 0.0},
        headers={"Content-Type": "application/json"}, timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _clean_summary(raw):
    headers = ["ABSTRACT:", "DECISIONS:", "ACTIONS:", "PROBLEMS:"]
    earliest = len(raw)
    for h in headers:
        pos = raw.upper().find(h)
        if pos != -1 and pos < earliest: earliest = pos
    return raw[earliest:].strip() if earliest < len(raw) else raw.strip()

def _extract_section(text, section):
    pattern = rf"{section}:\s*(.*?)(?=\n(?:ABSTRACT|DECISIONS|PROBLEMS|ACTIONS|ACTIONS_JSON):|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""

def _run_chunked_inference(transcript):
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
            content = _vllm_call(_chunk_prompt(chunk))
            cleaned = _clean_summary(content)
            if abs := _extract_section(cleaned, "ABSTRACT"):  all_abstracts.append(abs)
            if dec := _extract_section(cleaned, "DECISIONS"): all_decisions.append(dec)
            if prb := _extract_section(cleaned, "PROBLEMS"):  all_problems.append(prb)
            if act := _extract_section(cleaned, "ACTIONS"):   all_actions.append(act)
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

    # ── JSON recovery pass ────────────────────────────────────────────────────
    # If model generated ACTIONS text but no ACTIONS_JSON, do a targeted
    # conversion call to extract structured JSON from the plain-text actions.
    actions_text_merged = _dedup(all_actions)
    if not deduped and actions_text_merged and actions_text_merged != "None identified.":
        logger.info("ACTIONS_JSON empty — attempting JSON recovery pass")
        recovery_messages = [
            {"role": "system", "content":
                "You are a JSON extractor. Convert the action items list into a JSON array. "
                "Output ONLY the JSON array, nothing else. No markdown, no explanation."},
            {"role": "user", "content":
                f"Convert these action items into a JSON array where each item has keys: "
                f"owner (string), task (string), deadline (string), discussed_at_sec (float).\n\n"
                f"Action items:\n{actions_text_merged}\n\n"
                f"Rules:\n"
                f"- Use TBD for missing owner or deadline\n"
                f"- discussed_at_sec must be 0.0 if unknown\n"
                f"- Output ONLY the JSON array starting with [ and ending with ]\n"
                f"- No explanation, no markdown code blocks, just the raw JSON array\n\n"
                f"Example output format:\n"
                f'[{{"owner": "Alice", "task": "Fix bug", "deadline": "Friday", "discussed_at_sec": 0.0}}]'},
        ]
        try:
            recovery_raw = _vllm_call(recovery_messages, max_new_tokens=512)
            # Strip markdown fences if present
            recovery_clean = re.sub(r"```(?:json)?", "", recovery_raw).strip().strip("`").strip()
            # Find the JSON array
            arr_match = re.search(r"\[.*\]", recovery_clean, re.DOTALL)
            if arr_match:
                recovered = json.loads(arr_match.group(0))
                if isinstance(recovered, list) and recovered:
                    deduped = recovered
                    logger.info("JSON recovery pass succeeded: %d item(s) recovered", len(deduped))
                else:
                    logger.warning("JSON recovery pass returned empty array")
            else:
                logger.warning("JSON recovery pass: no array found in output")
        except Exception as re_e:
            logger.error("JSON recovery pass failed: %s", re_e)

    result  = f"ABSTRACT:\n{merged_abstract}\n\n"
    result += f"DECISIONS:\n{_dedup(all_decisions) or 'None identified.'}\n\n"
    result += f"PROBLEMS:\n{_dedup(all_problems) or 'None identified.'}\n\n"
    result += f"ACTIONS:\n{actions_text_merged or 'None identified.'}\n\n"
    result += f"ACTIONS_JSON:\n{json.dumps(deduped, indent=2)}"
    return result

def _parse_model_output(raw):
    result = {"abstract": "", "decisions": "", "problems": "", "actions": "", "actions_json": [], "raw_output": raw}
    for match in _SECTION_RE.finditer(raw):
        sec = match.group(1).upper(); content = match.group(2).strip()
        if sec == "ABSTRACT":    result["abstract"]  = content
        elif sec == "DECISIONS": result["decisions"] = content
        elif sec == "PROBLEMS":  result["problems"]  = content
        elif sec == "ACTIONS":   result["actions"]   = content
    m = _ACTIONS_JSON_RE.search(raw)
    if m:
        try: result["actions_json"] = json.loads(m.group(1))
        except Exception: pass
    return result


# =============================================================================
# Fallback: parse action items from plain ACTIONS text (restored from v1)
# Needed when model generates text-only ACTIONS but not valid ACTIONS_JSON.
# =============================================================================

def _parse_actions_from_text(actions_text):
    items = []
    if not actions_text:
        return items
    for line in actions_text.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        m = re.match(r"\[([^\]]+)\]\s*[-:]\s*(.+?)(?:\s*[-\u2013]\s*Due:\s*(.+))?$", line)
        if m:
            items.append({
                "owner":            m.group(1).strip(),
                "task":             m.group(2).strip(),
                "deadline":         m.group(3).strip() if m.group(3) else "TBD",
                "discussed_at_sec": 0.0,
            })
        elif line:
            items.append({"owner": "TBD", "task": line[:200], "deadline": "TBD", "discussed_at_sec": 0.0})
    return items


def _fix_discussed_at_sec(action_items: list, transcript: str) -> list:
    """
    Fix discussed_at_sec values that the model generated incorrectly.
    - Detects epoch timestamps (>86400) and resets them to 0.0
    - Extracts the meeting start time from transcript timestamps like [14:00] or HH:MM
    - Converts HH:MM timestamps found near each action item's owner/task to seconds from start
    """
    if not action_items:
        return action_items

    # Extract all HH:MM timestamps from transcript with their positions
    ts_pattern = re.compile(r"(?:\[\w[^\]]*\]\s*)?(\d{1,2}):(\d{2})(?:\s|$)")
    timestamps = []
    for m in ts_pattern.finditer(transcript):
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            pos_seconds = h * 3600 + mi * 60
            timestamps.append((m.start(), pos_seconds))

    if not timestamps:
        # No timestamps in transcript — just fix bad epoch values
        fixed = []
        for ai in action_items:
            val = ai.get("discussed_at_sec", 0.0)
            if isinstance(val, (int, float)) and val > 86400:
                ai = {**ai, "discussed_at_sec": 0.0}
            fixed.append(ai)
        return fixed

    start_seconds = timestamps[0][1]  # meeting start time in seconds of day

    fixed = []
    for ai in action_items:
        val = ai.get("discussed_at_sec", 0.0)
        # If model generated a valid relative timestamp (0-86400), keep it
        if isinstance(val, (int, float)) and 0.0 <= val <= 86400:
            fixed.append(ai)
            continue

        # Bad epoch or string — try to find the timestamp near this item in transcript
        task_text = str(ai.get("task", ""))[:60].lower()
        owner_text = str(ai.get("owner", ""))[:30].lower()
        best_pos = None
        best_score = -1

        # Find the position in transcript where this task/owner is discussed
        for word in task_text.split()[:5]:
            if len(word) < 4:
                continue
            idx = transcript.lower().find(word)
            if idx > 0:
                # Find nearest timestamp before this position
                for ts_pos, ts_sec in reversed(timestamps):
                    if ts_pos <= idx:
                        score = len(word)
                        if score > best_score:
                            best_score = score
                            best_pos = ts_sec
                        break

        if best_pos is not None:
            relative_sec = float(best_pos - start_seconds)
            if relative_sec < 0:
                relative_sec = 0.0
        else:
            relative_sec = 0.0

        fixed.append({**ai, "discussed_at_sec": relative_sec})

    return fixed


# =============================================================================
# ICS / CSV / artifact helpers (FIX-3)
# =============================================================================

def _detect_tz_offset(transcript):
    m = _TZ_RE.search(transcript)
    if m:
        offset = TZ_OFFSETS[m.group(1)]
        logger.info("TZ detected: %s (UTC%+.1f)", m.group(1), offset)
        return float(offset)
    return 0.0

def _parse_meeting_datetime(transcript, action_items):
    tz_offset = _detect_tz_offset(transcript)
    now_utc   = datetime.now(timezone.utc)
    def _to_utc(hour, minute):
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

def _generate_ics(meeting_name, action_items, meeting_dt):
    if meeting_dt.tzinfo is None:
        meeting_dt = meeting_dt.replace(tzinfo=timezone.utc)
    else:
        meeting_dt = meeting_dt.astimezone(timezone.utc)
    meeting_end = meeting_dt + timedelta(hours=1)
    fmt = "%Y%m%dT%H%M%SZ"
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Meeting Summarizer Agent//EN",
        "BEGIN:VEVENT",
        f"DTSTART:{meeting_dt.strftime(fmt)}", f"DTEND:{meeting_end.strftime(fmt)}",
        f"SUMMARY:Meeting Summary - {meeting_name}", "END:VEVENT",
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

def _generate_csv(meeting_name, action_items, triage=None, jira_map=None):
    """
    Generate action-items CSV.
    jira_map: optional dict {task_description_lower -> jira_key} built from
              tickets_created list after Jira queue completes.  When supplied,
              the 'jira_key' column is populated for matched rows.
    """
    import io as _io
    triage_map = {t.get("task_key", ""): t for t in (triage or [])}
    jira_map   = jira_map or {}
    buf = _io.StringIO()
    w   = csv_mod.DictWriter(
        buf,
        fieldnames=["meeting","poc_name","task_description","deadline","priority","risk","jira_key","discussed_at_sec"],
        lineterminator="\r\n",
    )
    w.writeheader()
    for ai in action_items:
        key = str(ai.get("task", ""))[:80].lower().strip()
        tr  = triage_map.get(key, {})
        w.writerow({
            "meeting":          meeting_name,
            "poc_name":         str(ai.get("owner", "TBD")),
            "task_description": str(ai.get("task", "")),
            "deadline":         str(ai.get("deadline", "TBD")),
            "priority":         tr.get("final_priority", ""),
            "risk":             tr.get("risk_summary", ""),
            "jira_key":         jira_map.get(key, ""),
            "discussed_at_sec": str(ai.get("discussed_at_sec", 0.0)),
        })
    return buf.getvalue().encode("utf-8")

def _build_summary_text(parsed):
    parts = []
    for key, label in [("abstract","ABSTRACT"),("decisions","DECISIONS"),("problems","PROBLEMS"),("actions","ACTIONS")]:
        if parsed.get(key):
            parts.append(f"{label}:\n{parsed[key]}")
    return "\n\n".join(parts)

def _build_html_email(file_name, summary_text, triage=None, jira_proposals=None):
    body_html   = summary_text.replace("\n", "<br>")
    triage_html = ""
    if triage:
        rows = "".join(
            f"<tr><td>{PRIORITY_EMOJI.get(t.get('final_priority',''),'⚪')} {t.get('final_priority','')}</td>"
            f"<td>{t.get('owner','')}</td><td>{t.get('task','')}</td><td>{t.get('risk_summary','')}</td></tr>"
            for t in triage
        )
        triage_html = (
            '<h3 style="color:#2c3e50;">Action Item Triage</h3>'
            '<table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">'
            '<tr style="background:#ecf0f1"><th>Priority</th><th>Owner</th><th>Task</th><th>Risk if Delayed</th></tr>'
            f"{rows}</table><br>"
        )
    jira_html = ""
    if jira_proposals:
        rows = "".join(
            f"<tr><td>{TICKET_TYPE_EMOJI.get(p.get('ticket_type','Task'),'🎫')} {p.get('ticket_type','Task')}</td>"
            f"<td>{p.get('summary','')}</td><td>{p.get('assignee','')}</td></tr>"
            for p in jira_proposals
        )
        jira_html = (
            '<h3 style="color:#2c3e50;">Proposed Jira Tickets</h3>'
            '<table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">'
            '<tr style="background:#ecf0f1"><th>Type</th><th>Summary</th><th>Assignee</th></tr>'
            f"{rows}</table><br>"
        )
    return (
        '<html><body style="font-family:Arial,sans-serif;max-width:700px;margin:auto;padding:20px;">'
        f'<h2 style="color:#2c3e50;">Meeting Summary</h2>'
        f'<h3 style="color:#7f8c8d;">{file_name}</h3>'
        '<hr style="border:1px solid #ecf0f1;">'
        f"{triage_html}{jira_html}"
        f'<div style="line-height:1.8;color:#34495e;">{body_html}</div>'
        '<hr style="border:1px solid #ecf0f1;">'
        '<p style="color:#95a5a6;font-size:12px;">Generated by Meeting Summarizer Agent (Qwen2.5-14B + CoD Triage + Jira CoD)</p>'
        "</body></html>"
    )


# =============================================================================
# CoD helpers
# =============================================================================

def _find_transcript_excerpt(action_item, transcript):
    owner      = action_item.get("owner", "")
    task_words = str(action_item.get("task", "")).split()[:3]
    excerpt    = transcript[:600]
    for term in [owner] + task_words:
        if term and term.lower() in transcript.lower():
            idx     = transcript.lower().find(term.lower())
            excerpt = transcript[max(0, idx - 100): idx + 500]
            break
    return excerpt

def _build_triage_context(action_item, transcript_snippet, all_actions, problems):
    return (
        f"=== MEETING CONTEXT ===\n"
        f"Problems/risks raised:\n{problems or 'None identified.'}\n\n"
        f"All action items:\n" +
        "\n".join(f"- [{a.get('owner','?')}] {a.get('task','')} (Due: {a.get('deadline','TBD')})"
                  for a in all_actions) +
        f"\n\n=== ACTION ITEM TO TRIAGE ===\n"
        f"Owner: {action_item.get('owner','TBD')}\n"
        f"Task:  {action_item.get('task','')}\n"
        f"Due:   {action_item.get('deadline','TBD')}\n\n"
        f"=== TRANSCRIPT EXCERPT ===\n{transcript_snippet[:600]}"
    )

def _run_triage_cod(action_item, transcript, all_actions, problems):
    context = _build_triage_context(
        action_item, _find_transcript_excerpt(action_item, transcript), all_actions, problems
    )
    try:
        proposal = _llm(structured_output=ActionPriorityProposal, model_name=MODEL).invoke([
            SystemMessage(content=(
                "You are the Proposer in a meeting action item triage debate.\n"
                "Classify the action item priority: Critical | High | Medium | Low.\n"
                "Critical: blocks team or release today. High: significant risk this week.\n"
                "Medium: important but flexible. Low: nice to have, no immediate risk.\n"
                "Base priority on impact, not just deadline text. Return ActionPriorityProposal."
            )),
            HumanMessage(content=context),
        ])
        challenge = _llm(structured_output=PriorityChallenge, model_name=MODEL).invoke([
            SystemMessage(content=(
                "You are the Challenger in a triage debate. Review the Proposer's classification.\n"
                "Challenge if: Proposer ignored a blocker, over-inflated priority for a routine task,\n"
                "or didn't weigh deadline urgency (EOD vs next week). Agree if correct.\n"
                "Return PriorityChallenge."
            )),
            HumanMessage(content=(
                f"{context}\n\nProposer: {proposal.priority} — {proposal.argument}"
            )),
        ])
        verdict = _llm(structured_output=TriageVerdict, model_name=MODEL).invoke([
            SystemMessage(content=(
                "You are the Judge in a triage debate. Make the final priority decision.\n"
                "Weigh both arguments. final_priority: Critical|High|Medium|Low.\n"
                "risk_summary: one sentence. deadline_note: 'On track' or brief note.\n"
                "Return TriageVerdict."
            )),
            HumanMessage(content=(
                f"{context}\n\nProposer: {proposal.priority} — {proposal.argument}\n"
                f"Challenger {'agreed' if challenge.agrees else 'countered with'}: "
                f"{challenge.counter_priority} — {challenge.argument}"
            )),
        ])
        return {
            "owner": action_item.get("owner", "TBD"), "task": action_item.get("task", ""),
            "task_key": str(action_item.get("task", ""))[:80].lower().strip(),
            "deadline": action_item.get("deadline", "TBD"),
            "final_priority": verdict.final_priority, "risk_summary": verdict.risk_summary,
            "deadline_note": verdict.deadline_note, "rationale": verdict.rationale,
        }
    except Exception as e:
        logger.error("Triage CoD failed for [%s]: %s", action_item.get("owner"), e)
        return None

def _run_jira_cod(action_item, transcript_excerpt, triage_result, all_actions):
    """3-round CoD to decide if this action item warrants a Jira ticket."""
    owner    = action_item.get("owner", "TBD")
    task     = action_item.get("task", "")
    deadline = action_item.get("deadline", "TBD")
    priority = (triage_result or {}).get("final_priority", "Medium")
    context = (
        f"=== ACTION ITEM ===\nOwner: {owner}\nTask: {task}\nDeadline: {deadline}\n"
        f"CoD Priority: {priority}\n\n"
        f"=== ALL ACTION ITEMS ===\n"
        + "\n".join(f"- [{a.get('owner','?')}] {a.get('task','')} (Due: {a.get('deadline','TBD')})"
                    for a in all_actions)
        + f"\n\n=== TRANSCRIPT EXCERPT ===\n{transcript_excerpt[:500]}"
    )
    proposer_sys = (
        "You are the Proposer in a Jira ticket detection debate.\n"
        "Decide if this action item should be a Jira ticket.\n"
        "CREATE if: concrete deliverable, clear owner, trackable work item (bug fix, feature, task).\n"
        "SKIP if: vague note, internal meeting decision, no real follow-up work.\n"
        "If creating, write a Jira summary (<=120 chars) and type: Task | Bug | Story.\n"
        "Return JiraProposalProposal."
    )
    challenger_sys = (
        "You are the Challenger. Review the Proposer's Jira ticket decision.\n"
        "Challenge if: creating a ticket for something too vague, missing a genuine deliverable,\n"
        "wrong type, or summary doesn't reflect the task. Agree if correct.\n"
        "Return JiraProposalChallenge."
    )
    judge_sys = (
        "You are the Judge. Make the final Jira ticket decision.\n"
        "create_ticket=True only for genuine trackable deliverables.\n"
        "summary: final <=120 chars. ticket_type: Task|Bug|Story. rationale: one sentence.\n"
        "Return JiraProposalVerdict."
    )
    try:
        proposal = _llm(structured_output=JiraProposalProposal, model_name=MODEL).invoke([
            SystemMessage(content=proposer_sys), HumanMessage(content=context)
        ])
        logger.info("Jira proposer [%s]: create=%s type=%s", owner, proposal.should_create, proposal.ticket_type)
        challenge = _llm(structured_output=JiraProposalChallenge, model_name=MODEL).invoke([
            SystemMessage(content=challenger_sys),
            HumanMessage(content=(
                f"{context}\n\nProposer: {'CREATE' if proposal.should_create else 'SKIP'}\n"
                f"Summary: {proposal.summary}\nType: {proposal.ticket_type}\nArg: {proposal.argument}"
            )),
        ])
        logger.info("Jira challenger [%s]: agrees=%s counter=%s", owner, challenge.agrees, challenge.counter_should_create)
        verdict = _llm(structured_output=JiraProposalVerdict, model_name=MODEL).invoke([
            SystemMessage(content=judge_sys),
            HumanMessage(content=(
                f"{context}\n\nProposer: {'CREATE' if proposal.should_create else 'SKIP'} — {proposal.argument}\n"
                f"Challenger {'agreed' if challenge.agrees else 'countered'}: "
                f"{'CREATE' if challenge.counter_should_create else 'SKIP'} — {challenge.argument}"
            )),
        ])
        logger.info("Jira verdict [%s]: create=%s summary=%s", owner, verdict.create_ticket, (verdict.summary or "")[:60])
        if not verdict.create_ticket:
            return None
        return {
            "owner": owner, "assignee": owner,
            "summary": verdict.summary[:120] if verdict.summary else f"[{owner}] {task[:80]}",
            "ticket_type": verdict.ticket_type or "Task",
            "rationale": verdict.rationale,
        }
    except Exception as e:
        logger.error("Jira CoD failed for [%s]: %s", owner, e)
        return None


# =============================================================================
# Jira REST API helper
# =============================================================================

def _create_jira_ticket(summary, assignee_name, ticket_type, description=""):
    """Create a Jira issue. Returns issue key (e.g. 'KAN-42') or None on failure."""
    jira_account_id = TEAM_MAP.get(assignee_name)
    if not jira_account_id:
        for name, acc_id in TEAM_MAP.items():
            if name.lower() in assignee_name.lower():
                jira_account_id = acc_id
                break
    fields = {
        "project":     {"key": JIRA_PROJECT_KEY},
        "summary":     summary[:120],
        "description": {
            "type": "doc", "version": 1,
            "content": [{"type": "paragraph", "content": [
                {"type": "text", "text": description or "Created via Meeting Summarizer Agent."}
            ]}],
        },
        "issuetype": {"name": ticket_type or JIRA_ISSUE_TYPE},
    }
    if jira_account_id:
        fields["assignee"] = {"id": jira_account_id}
    try:
        resp = requests.post(
            f"{JIRA_BASE_URL}/rest/api/3/issue",
            json={"fields": fields},
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN),
            timeout=30,
        )
        logger.info("Jira create status=%s", resp.status_code)
        if resp.ok:
            key = resp.json().get("key")
            logger.info("Jira ticket created: %s", key)
            return key
        logger.error("Jira create failed: %s %s", resp.status_code, resp.text[:300])
        return None
    except Exception as e:
        logger.error("Jira create exception: %s", e)
        return None


# =============================================================================
# Entry router
# =============================================================================

def route_meeting_entry(state: OrchestratorState) -> Literal[
    "meeting_fetch_transcript", "meeting_send_email", "meeting_post_cancel",
    "meeting_create_jira", "meeting_post_next_jira"
]:
    action = state.slack_action_id or ""
    if action == "confirm_summary":       return "meeting_send_email"
    if action == "cancel_summary":        return "meeting_post_cancel"
    if action == "confirm_meeting_jira":  return "meeting_create_jira"
    if action == "skip_meeting_jira":     return "meeting_post_next_jira"
    return "meeting_fetch_transcript"


# =============================================================================
# Node 1: meeting_fetch_transcript
# =============================================================================

def meeting_fetch_transcript(state: OrchestratorState) -> OrchestratorState:
    """Download transcript from Google Drive.
    Uses Redis SET NX for idempotency instead of S3 lock —
    auto-expires after LOCK_TTL_SECONDS, no manual cleanup needed.
    If Redis is unavailable, proceeds anyway (fail-open for resilience).
    """
    file_id   = state.transcript_file_id
    file_name = state.transcript_file_name or "transcript.txt"

    if not file_id:
        return state.model_copy(update={"error": "missing file_id"})

    # Redis-based dedup: SET NX with TTL — atomic, auto-expiring
    lock_key = f"meeting_lock:{file_id}"
    try:
        from redis_store import _get_client
        r = _get_client()
        acquired = r.set(lock_key, "processing", nx=True, ex=LOCK_TTL_SECONDS)
        if not acquired:
            logger.info("Duplicate trigger ignored for %s (Redis lock active)", file_id)
            return state.model_copy(update={"error": "duplicate_trigger"})
        logger.info("Redis lock acquired for %s", file_id)
    except Exception as e:
        # Redis unavailable — proceed anyway, Apps Script retry logic handles dedup
        logger.warning("Redis lock unavailable, proceeding without dedup: %s", e)

    # Global vLLM serialization lock — only one transcript inference runs at a time.
    # vLLM is a single GPU instance; concurrent inference from two Lambda invocations
    # causes severe queue pressure and timeout cascades.
    # We poll with a short sleep rather than blocking Redis BLPOP so Lambda stays alive.
    VLLM_LOCK_KEY = "vllm_global_lock"
    VLLM_LOCK_TTL = 900  # 15 min — same as Lambda max timeout
    VLLM_WAIT_MAX = 840  # wait up to 14 min for the lock to free
    VLLM_POLL_SEC = 15   # check every 15 seconds
    _vllm_lock_acquired = False
    try:
        from redis_store import _get_client as _get_r
        _r = _get_r()
        _waited = 0
        while _waited < VLLM_WAIT_MAX:
            _vllm_lock_acquired = _r.set(
                VLLM_LOCK_KEY, file_id, nx=True, ex=VLLM_LOCK_TTL
            )
            if _vllm_lock_acquired:
                logger.info("vLLM global lock acquired for %s", file_id)
                break
            _owner = (_r.get(VLLM_LOCK_KEY) or b"").decode()
            logger.info(
                "vLLM busy (held by %s) — waiting %ds (elapsed=%ds)",
                _owner, VLLM_POLL_SEC, _waited,
            )
            import time as _time; _time.sleep(VLLM_POLL_SEC)
            _waited += VLLM_POLL_SEC
        if not _vllm_lock_acquired:
            logger.error("vLLM global lock wait timed out after %ds — proceeding anyway", _waited)
    except Exception as _ve:
        logger.warning("vLLM global lock unavailable: %s — proceeding without serialization", _ve)

    try:
        raw_text = _download_from_drive(file_id)
        logger.info("Downloaded %s: %d chars", file_name, len(raw_text))
        return state.model_copy(update={"transcript_text": raw_text})
    except Exception as e:
        logger.error("Drive download failed: %s", e)
        # Release lock on failure so Apps Script retry can reprocess
        try:
            from redis_store import _get_client
            _get_client().delete(lock_key)
        except Exception:
            pass
        return state.model_copy(update={"error": str(e)})


# =============================================================================
# Node 2: meeting_preprocess
# =============================================================================

def meeting_preprocess(state: OrchestratorState) -> OrchestratorState:
    if state.error: return state
    raw = state.transcript_text or ""
    if not raw: return state.model_copy(update={"error": "empty transcript"})
    return state.model_copy(update={"transcript_processed": _preprocess(raw)})


# =============================================================================
# Node 3: meeting_summarize
# =============================================================================

def meeting_summarize(state: OrchestratorState) -> OrchestratorState:
    if state.error: return state
    transcript = state.transcript_processed or state.transcript_text or ""
    if not transcript: return state.model_copy(update={"error": "no transcript to summarize"})
    try:
        raw_output = _run_chunked_inference(transcript)
        parsed     = _parse_model_output(raw_output)
        logger.info("Summarized: %d action items", len(parsed.get("actions_json", [])))
        return state.model_copy(update={"meeting_summary_parsed": parsed})
    except Exception as e:
        logger.error("Summarize failed: %s", e)
        return state.model_copy(update={"error": str(e)})


# =============================================================================
# Node 4: meeting_triage_cod
# =============================================================================

def meeting_triage_cod(state: OrchestratorState) -> OrchestratorState:
    """
    Run CoD triage on each extracted action item.

    Fallback (restored from v1): if ACTIONS_JSON is empty, parses action items
    from the plain text ACTIONS section so CoD always has something to work with.
    Fails gracefully — pipeline continues even if CoD errors.
    """
    if state.error: return state
    parsed   = state.meeting_summary_parsed or {}
    actions  = parsed.get("actions_json", [])
    problems = parsed.get("problems", "")

    # Fallback: parse from plain text ACTIONS when ACTIONS_JSON is missing
    if not actions:
        actions_text = parsed.get("actions", "")
        if actions_text:
            actions = _parse_actions_from_text(actions_text)
            logger.info("ACTIONS_JSON empty — parsed %d item(s) from text ACTIONS", len(actions))
            state = state.model_copy(update={"meeting_summary_parsed": {**parsed, "actions_json": actions}})

    if not actions:
        logger.info("No action items to triage")
        return state.model_copy(update={"meeting_triage": []})

    # ── Pre-filter: skip CoD for vague items (owner=TBD or deadline=TBD) ─────
    # Items with no clear owner or deadline are unlikely to become Jira tickets.
    # Cap at 8 items max to stay safely within Lambda 15-min timeout.
    COD_MAX_ITEMS = 8
    concrete, vague = [], []
    for ai in actions:
        owner    = str(ai.get("owner", "TBD")).strip().upper()
        deadline = str(ai.get("deadline", "TBD")).strip().upper()
        if owner == "TBD" or deadline == "TBD":
            vague.append(ai)
        else:
            concrete.append(ai)

    cod_items     = concrete[:COD_MAX_ITEMS]
    skipped_items = vague + concrete[COD_MAX_ITEMS:]

    if skipped_items:
        logger.info(
            "CoD pre-filter: %d item(s) for CoD, %d item(s) skipped (TBD or over cap=%d)",
            len(cod_items), len(skipped_items), COD_MAX_ITEMS,
        )

    default_triage = [
        {
            "owner":          ai.get("owner", "TBD"),
            "task":           ai.get("task", ""),
            "task_key":       str(ai.get("task", ""))[:80].lower().strip(),
            "deadline":       ai.get("deadline", "TBD"),
            "final_priority": "Medium",
            "risk_summary":   "Skipped CoD pre-filter — TBD owner/deadline or over item cap.",
            "deadline_note":  "On track",
            "rationale":      "Pre-filter: defaulting to Medium.",
        }
        for ai in skipped_items
    ]

    actions = cod_items

    # Fix any bad discussed_at_sec values (epoch timestamps or invalid floats)
    transcript = state.transcript_processed or state.transcript_text or ""
    actions = _fix_discussed_at_sec(actions, transcript)
    state = state.model_copy(update={"meeting_summary_parsed": {**parsed, "actions_json": actions}})

    transcript     = state.transcript_processed or state.transcript_text or ""
    logger.info("Starting CoD triage for %d action item(s) [parallel]", len(actions))

    def _triage_one(ai):
        result = _run_triage_cod(ai, transcript, actions, problems)
        return result or {
            "owner": ai.get("owner", "TBD"), "task": ai.get("task", ""),
            "task_key": str(ai.get("task", ""))[:80].lower().strip(),
            "deadline": ai.get("deadline", "TBD"),
            "final_priority": "Medium", "risk_summary": "Triage unavailable.",
            "deadline_note": "On track", "rationale": "CoD triage failed — defaulting to Medium.",
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed
    # Cap workers at 4 to avoid overwhelming vLLM with too many concurrent requests
    max_workers = min(4, len(actions))
    triage_results = [None] * len(actions)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_triage_one, ai): i for i, ai in enumerate(actions)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                triage_results[idx] = future.result()
            except Exception as e:
                logger.error("Triage parallel worker failed for idx=%d: %s", idx, e)
                ai = actions[idx]
                triage_results[idx] = {
                    "owner": ai.get("owner", "TBD"), "task": ai.get("task", ""),
                    "task_key": str(ai.get("task", ""))[:80].lower().strip(),
                    "deadline": ai.get("deadline", "TBD"),
                    "final_priority": "Medium", "risk_summary": "Triage unavailable.",
                    "deadline_note": "On track", "rationale": "CoD triage failed — defaulting to Medium.",
                }

    # Merge CoD results with default triage for pre-filtered items
    triage_results = triage_results + default_triage
    logger.info(
        "CoD triage complete: %d CoD classified + %d pre-filter defaults = %d total",
        len(cod_items), len(default_triage), len(triage_results),
    )
    return state.model_copy(update={"meeting_triage": triage_results})


# =============================================================================
# Node 5: meeting_jira_cod
# =============================================================================

def meeting_jira_cod(state: OrchestratorState) -> OrchestratorState:
    """
    Run CoD to decide which action items should become Jira tickets.
    Each action item: 3-round debate (Proposer -> Challenger -> Judge).
    Results stored in meeting_jira_proposals.
    Fails gracefully — pipeline continues with empty proposals on error.
    """
    if state.error: return state

    parsed  = state.meeting_summary_parsed or {}
    actions = parsed.get("actions_json", [])
    triage  = state.meeting_triage or []

    if not actions:
        logger.info("No action items for Jira CoD")
        return state.model_copy(update={"meeting_jira_proposals": []})

    triage_map = {t.get("task_key", ""): t for t in triage}
    transcript = state.transcript_processed or state.transcript_text or ""
    proposals  = []

    logger.info("Starting Jira CoD for %d action item(s) [parallel]", len(actions))

    def _jira_one(ai):
        task_key      = str(ai.get("task", ""))[:80].lower().strip()
        triage_result = triage_map.get(task_key)
        excerpt       = _find_transcript_excerpt(ai, transcript)
        return ai, _run_jira_cod(ai, excerpt, triage_result, actions)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = min(3, len(actions))
    results_ordered = [None] * len(actions)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_jira_one, ai): i for i, ai in enumerate(actions)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results_ordered[idx] = future.result()
            except Exception as e:
                logger.error("Jira CoD parallel worker failed for idx=%d: %s", idx, e)
                results_ordered[idx] = (actions[idx], None)

    for ai, result in results_ordered:
        if result:
            proposals.append(result)
            logger.info("Jira CoD PROPOSE: [%s] %s", ai.get("owner"), result["summary"][:60])
        else:
            logger.info("Jira CoD SKIP: [%s] %s", ai.get("owner"), str(ai.get("task", ""))[:40])

    logger.info("Jira CoD complete: %d/%d items proposed as tickets [parallel]", len(proposals), len(actions))
    return state.model_copy(update={"meeting_jira_proposals": proposals})


# =============================================================================
# Node 6: meeting_generate_artifacts
# =============================================================================

def meeting_generate_artifacts(state: OrchestratorState) -> OrchestratorState:
    """Generate CSV of action items with priority/risk. No ICS — calendar
    scheduling is handled autonomously by the email agent (msadi)."""
    if state.error: return state
    parsed    = state.meeting_summary_parsed or {}
    file_name = state.transcript_file_name or "transcript.txt"
    actions   = parsed.get("actions_json", [])
    triage    = state.meeting_triage or []
    csv_bytes = _generate_csv(file_name, actions, triage)
    logger.info("Artifacts: CSV generated (%d action items). No ICS — handled by calendar agent.", len(actions))

    # Release the global vLLM lock — inference + CoD are complete.
    # The next queued transcript can now start processing.
    try:
        from redis_store import _get_client as _get_r
        _r = _get_r()
        _r.delete("vllm_global_lock")
        logger.info("vLLM global lock released after artifact generation")
    except Exception as _ve:
        logger.warning("Could not release vLLM global lock: %s", _ve)

    return state.model_copy(update={"meeting_ics_bytes": None, "meeting_csv_bytes": csv_bytes})


# =============================================================================
# Node 7: meeting_store_s3
# =============================================================================

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
            "file_name": file_name, "s3_key": s3_key, "timestamp": timestamp,
            "summary_text": summary_text, "abstract": parsed.get("abstract", ""),
            "decisions": parsed.get("decisions", ""), "problems": parsed.get("problems", ""),
            "actions": parsed.get("actions", ""), "triage": state.meeting_triage or [],
            "jira_proposals": state.meeting_jira_proposals or [],
            "actions_json": parsed.get("actions_json", []),
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


# =============================================================================
# Node 8: meeting_post_slack
# =============================================================================

def meeting_post_slack(state: OrchestratorState) -> OrchestratorState:
    if state.error and state.error != "duplicate_trigger":
        logger.error("Skipping Slack post due to error: %s", state.error)
        return state
    if state.error == "duplicate_trigger":
        return state

    parsed         = state.meeting_summary_parsed or {}
    file_name      = state.transcript_file_name or "transcript.txt"
    s3_key         = state.meeting_s3_key or ""
    triage         = state.meeting_triage or []
    jira_proposals = state.meeting_jira_proposals or []
    channel        = SLACK_NOTIFY_CHANNEL or os.environ.get("SLACK_CHANNEL_ID", "")

    abstract  = parsed.get("abstract",  "No abstract generated.")[:300]
    decisions = parsed.get("decisions", "None.") or "None."
    problems  = parsed.get("problems",  "None.") or "None."
    n_actions = len(parsed.get("actions_json", []))
    has_ics   = state.meeting_ics_bytes is not None

    # Build triage section
    triage_text = ""
    if triage:
        priority_order = ["Critical", "High", "Medium", "Low"]
        lines = []
        for t in sorted(triage, key=lambda x: priority_order.index(
                x.get("final_priority", "Low") if x.get("final_priority", "Low") in priority_order else "Low")):
            emoji = PRIORITY_EMOJI.get(t.get("final_priority", "Medium"), "⚪")
            lines.append(
                f"{emoji} *{t.get('final_priority','')}* — "
                f"[{t.get('owner','')}] {t.get('task','')[:60]}\n"
                f"   _{t.get('risk_summary','')}_ | {t.get('deadline_note','')}"
            )
        triage_text = "\n".join(lines)

    # Build Jira proposals preview section
    jira_text = ""
    if jira_proposals:
        lines = []
        for p in jira_proposals:
            emoji = TICKET_TYPE_EMOJI.get(p.get("ticket_type", "Task"), "🎫")
            lines.append(f"{emoji} *{p.get('ticket_type','Task')}* — [{p.get('owner','')}] {p.get('summary','')[:70]}")
        jira_text = "\n".join(lines)

    # Save Jira queue to Redis so the confirm button can reference it
    jira_queue_session_id = None
    if jira_proposals:
        try:
            jira_queue_session_id = save_session({
                "flow": "meeting_jira_queue", "items": jira_proposals,
                "channel_id": channel, "file_name": file_name,
                "s3_key": s3_key,
                "tickets_created": [],
                "tickets_skipped": 0,
            })
            logger.info("Jira queue saved: %s (%d item(s))", jira_queue_session_id, len(jira_proposals))
        except Exception as e:
            logger.error("Failed to save Jira queue: %s", e)

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

    if triage_text:
        blocks += [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Action Item Triage ({n_actions})*\n{triage_text[:2500]}"}},
            {"type": "divider"},
        ]
    else:
        actions_text = parsed.get("actions", "None.") or "None."
        blocks += [
            {"type": "section", "text": {"type": "mrkdwn", "text": (
                f"*Action Items ({n_actions})*\n{actions_text[:500]}"
                + ("\n\n_ICS calendar invite included._" if has_ics else "")
            )}},
            {"type": "divider"},
        ]

    if jira_text:
        blocks += [
            {"type": "section", "text": {"type": "mrkdwn", "text": (
                f"*Proposed Jira Tickets ({len(jira_proposals)})*\n{jira_text[:600]}\n\n"
                "_After confirming, each ticket will appear for individual approval._"
            )}},
            {"type": "divider"},
        ]

    confirm_note = (f" + review *{len(jira_proposals)} Jira ticket(s)*" if jira_proposals else "")
    blocks += [
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            f"*Ready to send to:* {', '.join(PARTICIPANT_EMAILS)}\n"
            f"Click *Confirm* to email summary + triage + ICS + CSV{confirm_note}.\n"
            "Click *Cancel* to dismiss."
        )}},
        {"type": "actions", "elements": [
            {
                "type": "button", "style": "primary",
                "text":      {"type": "plain_text", "text": "Confirm - Send Email"},
                "value":     json.dumps({
                    "flow": "meeting", "file_name": file_name, "s3_key": s3_key,
                    "jira_queue_session_id": jira_queue_session_id,
                }),
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
        resp = WebClient(token=SLACK_BOT_TOKEN).chat_postMessage(
            channel=channel, blocks=blocks,
            text=f"New meeting summary ready: {file_name}",
        )
        logger.info("Posted to Slack channel=%s ts=%s", channel, resp["ts"])
    except Exception as e:
        logger.error("Slack post failed: %s", e)
        return state.model_copy(update={"error": str(e)})
    return state



# =============================================================================
# Follow-up meeting suggestion extractor (for email agent)
# =============================================================================

def _extract_followup_suggestions(transcript: str, action_items: list) -> tuple[list, list]:
    """
    Extract potential follow-up meeting references from transcript + action deadlines.
    Returns (suggestions, deadline_lines) for the email agent payload.
    """
    # Pattern: explicit sync/meeting mentions in transcript
    sync_re = re.compile(
        r".{0,30}(?:sync|regroup|reconvene|meet|standup|check.in).{0,60}"
        r"(?:tomorrow|monday|tuesday|wednesday|thursday|friday|"
        r"\d{1,2}\s*(?:am|pm)|\d{1,2}:\d{2})",
        re.IGNORECASE,
    )
    suggestions = []
    seen = set()
    for m in sync_re.finditer(transcript):
        snippet = m.group().strip()[:120]
        if snippet.lower() not in seen:
            seen.add(snippet.lower())
            suggestions.append(snippet)

    # Action item deadlines as scheduling anchors
    deadline_lines = []
    for ai in action_items:
        d = str(ai.get("deadline", "")).strip()
        if d and d.upper() != "TBD":
            deadline_lines.append(
                f"  - [{ai.get('owner','?')}] {str(ai.get('task',''))[:80]} — Due: {d}"
            )

    return suggestions[:3], deadline_lines

# =============================================================================
# Node 9a: meeting_send_email
# =============================================================================

def meeting_send_email(state: OrchestratorState) -> OrchestratorState:
    """
    On Confirm click:
      1. Sends plain-text trigger to EMAIL_AGENT_INBOX (msadi) with ONLY:
           - Follow-up meeting proposals extracted from transcript discussion
           - Participant list
         No attachments. Calendar agent handles scheduling autonomously.
      2. Updates Slack card and kicks off Jira proposal queue.
      Consolidated participant email is sent later by meeting_post_next_jira
      once the entire Jira queue is processed.
    """
    import email.mime.multipart as mp
    import email.mime.text      as mt

    value_dict            = state.slack_action_value or {}
    file_name             = value_dict.get("file_name", "meeting")
    s3_key                = value_dict.get("s3_key", "").strip()
    jira_queue_session_id = value_dict.get("jira_queue_session_id")
    channel_id            = state.channel_id
    preview_ts            = state.preview_ts
    slack                 = WebClient(token=SLACK_BOT_TOKEN)

    if not s3_key:
        logger.error("meeting_send_email: s3_key missing. value_dict=%s", value_dict)
        if channel_id and preview_ts:
            try:
                slack.chat_update(
                    channel=channel_id, ts=preview_ts, blocks=None,
                    text="Session expired — please re-upload the transcript to regenerate the summary.",
                )
            except Exception:
                pass
        return state.model_copy(update={"error": "session_expired_or_missing_s3_key"})

    try:
        s3 = boto3.client("s3")

        # ── Read meta.json ─────────────────────────────────────────────────────
        meta_key = f"{s3_key}/meta.json"
        logger.info("meeting_send_email: reading s3://%s/%s", S3_BUCKET, meta_key)
        raw_meta_bytes = s3.get_object(Bucket=S3_BUCKET, Key=meta_key)["Body"].read()
        logger.info("meeting_send_email: meta.json raw bytes len=%d", len(raw_meta_bytes))
        if raw_meta_bytes.startswith(b"\xef\xbb\xbf"):
            raw_meta_bytes = raw_meta_bytes[3:]
        raw_meta = raw_meta_bytes.decode("utf-8").strip()
        try:
            meta = json.loads(raw_meta)
        except json.JSONDecodeError as je:
            logger.error(
                "meeting_send_email: meta.json parse failed — %s  s3_key=%s  content=%r",
                je, meta_key, raw_meta[:300],
            )
            if channel_id and preview_ts:
                try:
                    slack.chat_update(
                        channel=channel_id, ts=preview_ts, blocks=None,
                        text=f"⚠️ Failed to parse meeting metadata (key: `{meta_key}`). Error: {je}",
                    )
                except Exception:
                    pass
            return state.model_copy(update={"error": f"meta_json_parse_error: {je}"})

        action_items = meta.get("actions_json", [])

        # ── Load transcript for follow-up extraction ───────────────────────────
        transcript_raw = ""
        try:
            transcript_raw = s3.get_object(
                Bucket=S3_BUCKET, Key=f"{s3_key}/transcript.txt"
            )["Body"].read().decode("utf-8")
        except Exception as te:
            logger.warning("meeting_send_email: could not load transcript from S3: %s", te)

        # ── Extract follow-up meeting proposals from transcript ─────────────────
        followup_suggestions, _ = _extract_followup_suggestions(transcript_raw, action_items)

        # ── Send plain-text trigger to calendar agent (no attachments) ─────────
        proposals_text = (
            "\n".join(f"  - {s}" for s in followup_suggestions)
            if followup_suggestions
            else "  (No explicit follow-up meeting proposals detected in discussion)"
        )
        participants_text = "\n".join(f"  - {e}" for e in PARTICIPANT_EMAILS)

        agent_body = (
            f"MEETING_CALENDAR_TRIGGER\n"
            f"FILE: {file_name}\n"
            f"\n"
            f"MEETING_PROPOSALS_FROM_DISCUSSION:\n"
            f"{proposals_text}\n"
            f"\n"
            f"PARTICIPANTS:\n"
            f"{participants_text}\n"
            f"\n"
            f"ACTION:\n"
            f"  Please detect scheduling intent from the proposals above,\n"
            f"  check participant calendars for conflicts, schedule the meeting,\n"
            f"  and send calendar invites directly to all participants listed.\n"
        )

        # ── Build MIME message ────────────────────────────────────────────────
        agent_msg = mp.MIMEMultipart("alternative")
        agent_msg["Subject"] = f"[MEETING-AGENT] {file_name}"
        agent_msg["From"]    = SES_FROM_EMAIL
        agent_msg["To"]      = EMAIL_AGENT_INBOX
        # Cc all participants so calendar agent reads them from email headers
        if PARTICIPANT_EMAILS:
            agent_msg["Cc"] = ", ".join(PARTICIPANT_EMAILS)
        agent_msg.attach(mt.MIMEText(agent_body, "plain"))

        # ── Send via Gmail API (avoids SES DMARC spam issue) ──────────────────
        _google_token_raw = os.environ.get("GOOGLE_TOKEN_JSON", "")
        _sent_via_gmail = False
        if _google_token_raw:
            try:
                import base64 as _b64
                from googleapiclient.discovery import build as _gbuild
                from google.oauth2.credentials import Credentials as _GCreds
                from google.auth.transport.requests import Request as _GReq
                _td = json.loads(_google_token_raw)
                _creds = _GCreds(
                    token=_td.get("token"),
                    refresh_token=_td.get("refresh_token"),
                    token_uri=_td.get("token_uri", "https://oauth2.googleapis.com/token"),
                    client_id=_td.get("client_id"),
                    client_secret=_td.get("client_secret"),
                    scopes=_td.get("scopes", []),
                )
                if _creds.expired and _creds.refresh_token:
                    _creds.refresh(_GReq())
                _gmail = _gbuild("gmail", "v1", credentials=_creds, cache_discovery=False)
                _raw = _b64.urlsafe_b64encode(agent_msg.as_bytes()).decode()
                _gmail.users().messages().send(userId="me", body={"raw": _raw}).execute()
                _sent_via_gmail = True
                logger.info(
                    "meeting_send_email: calendar trigger sent via Gmail API to %s | cc=%s | proposals=%d",
                    EMAIL_AGENT_INBOX, PARTICIPANT_EMAILS, len(followup_suggestions),
                )
            except Exception as _ge:
                logger.warning("meeting_send_email: Gmail API send failed (%s) — falling back to SES", _ge)

        if not _sent_via_gmail:
            # Fallback to SES if Gmail API unavailable or failed
            ses = boto3.client("ses", region_name="us-east-1")
            all_trigger_destinations = [EMAIL_AGENT_INBOX] + PARTICIPANT_EMAILS
            ses.send_raw_email(
                Source=SES_FROM_EMAIL,
                Destinations=all_trigger_destinations,
                RawMessage={"Data": agent_msg.as_string()},
            )
            logger.info(
                "meeting_send_email: calendar trigger sent via SES to %s | cc=%s | proposals=%d",
                EMAIL_AGENT_INBOX, PARTICIPANT_EMAILS, len(followup_suggestions),
            )

        # ── Update Slack card ──────────────────────────────────────────────────
        if channel_id and preview_ts:
            jira_note = (f" Reviewing *{len(meta.get('jira_proposals', []))} Jira ticket(s)* next."
                         if meta.get("jira_proposals") else "")
            slack.chat_update(
                channel=channel_id, ts=preview_ts, blocks=None,
                text=(
                    f"✅ Confirmed. 📅 Calendar agent notified for scheduling.{jira_note}\n"
                    f"_Consolidated summary email will be sent to participants after Jira review._"
                ),
            )

    except Exception as e:
        logger.error("meeting_send_email failed: %s", e)
        if channel_id and preview_ts:
            try:
                slack.chat_update(
                    channel=channel_id, ts=preview_ts,
                    text=f"Error: {e}", blocks=None,
                )
            except Exception:
                pass
        return state.model_copy(update={"error": str(e)})

    # Hand the Jira queue session forward to meeting_post_next_jira
    return state.model_copy(update={"meeting_jira_queue_session": jira_queue_session_id})


# =============================================================================
# Node 9b: meeting_post_cancel
# =============================================================================

def meeting_post_cancel(state: OrchestratorState) -> OrchestratorState:
    if state.channel_id and state.preview_ts:
        try:
            WebClient(token=SLACK_BOT_TOKEN).chat_update(
                channel=state.channel_id, ts=state.preview_ts, blocks=None,
                text="Meeting summary dismissed. No email sent.",
            )
        except Exception as e:
            return state.model_copy(update={"error": str(e)})
    return state



# =============================================================================
# Consolidated email sender (called when Jira queue is complete)
# =============================================================================

def _send_consolidated_email(
    tickets_created: list,
    tickets_skipped: int,
    file_name: str,
    channel: str,
    s3_key: str = "",
):
    """
    Single consolidated email to PARTICIPANT_EMAILS after Jira queue done.
    Contains: full meeting summary + Jira tickets table + CSV attachment
    (all action items with priority/risk, with and without Jira tickets).
    Calendar invite note — scheduling agent handles that autonomously.
    """
    import email.mime.multipart as mp
    import email.mime.text      as mt
    import email.mime.base      as mb
    import email.encoders       as encoders

    if not PARTICIPANT_EMAILS:
        logger.warning("_send_consolidated_email: no PARTICIPANT_EMAILS configured")
        return

    total     = len(tickets_created) + tickets_skipped
    created_n = len(tickets_created)

    # ── Load summary + action items from S3; rebuild CSV with Jira keys ──────────
    summary_html_section = ""
    csv_bytes = None
    if s3_key:
        try:
            s3 = boto3.client("s3")
            raw  = s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/meta.json")["Body"].read()
            meta = json.loads(raw.decode("utf-8").strip())
            summary_text = meta.get("summary_text", "")
            if summary_text:
                summary_html_section = (
                    '<h3 style="color:#2c3e50;">📋 Meeting Summary</h3>'
                    f'<div style="line-height:1.8;color:#34495e;">'
                    f'{summary_text.replace(chr(10), "<br>")}</div>'
                    '<br><hr style="border:1px solid #ecf0f1;">'
                )
            # Build jira_map: match by OWNER (reliable) rather than task text (unreliable
            # because Jira summaries are reworded by CoD and truncated at 120 chars).
            # owner_jira_map: owner_lower -> jira_key (last ticket wins if same owner has multiple)
            owner_jira_map = {}
            for t in tickets_created:
                if t.get("jira_key"):
                    owner_key = t.get("assignee", t.get("owner", "")).lower().strip()
                    if owner_key:
                        owner_jira_map[owner_key] = t.get("jira_key", "")

            def _find_jira_key_by_owner(owner):
                owner_lower = owner.lower().strip()
                # Exact match first
                if owner_lower in owner_jira_map:
                    return owner_jira_map[owner_lower]
                # Partial match (e.g. "Ketki" matches "Ketki Maddiwar")
                for k, v in owner_jira_map.items():
                    if owner_lower in k or k in owner_lower:
                        return v
                return ""

            jira_map = {
                ai.get("task", "")[:80].lower().strip(): _find_jira_key_by_owner(ai.get("owner", ""))
                for ai in meta.get("actions_json", [])
            }
            # Regenerate CSV fresh so jira_key column reflects confirmed tickets
            action_items_meta = meta.get("actions_json", [])
            triage_meta       = meta.get("triage", [])
            if action_items_meta:
                csv_bytes = _generate_csv(
                    file_name, action_items_meta, triage=triage_meta, jira_map=jira_map
                )
                # Overwrite the S3 copy so it stays up to date
                try:
                    s3.put_object(
                        Bucket=S3_BUCKET,
                        Key=f"{s3_key}/actions.csv",
                        Body=csv_bytes,
                        ContentType="text/csv",
                    )
                    logger.info(
                        "_send_consolidated_email: CSV regenerated with %d jira_key(s), "
                        "%d bytes — S3 updated",
                        len(jira_map), len(csv_bytes),
                    )
                except Exception as ue:
                    logger.warning("_send_consolidated_email: S3 CSV overwrite failed: %s", ue)
            else:
                logger.warning("_send_consolidated_email: no action_items in meta — CSV skipped")
        except Exception as me:
            logger.warning("_send_consolidated_email: meta/CSV rebuild failed: %s", me)

    # ── Jira tickets table ─────────────────────────────────────────────────────
    ticket_rows = ""
    for t in tickets_created:
        jira_key    = t.get("jira_key", "")
        summary     = t.get("summary", "")
        assignee    = t.get("assignee", "TBD")
        ticket_type = t.get("ticket_type", "Task")
        link = (
            f'<a href="{JIRA_BASE_URL}/browse/{jira_key}">{jira_key}</a>'
            if jira_key and JIRA_BASE_URL else (jira_key or "—")
        )
        ticket_rows += (
            f"<tr><td>{link}</td><td>{ticket_type}</td>"
            f"<td>{summary[:100]}</td><td>{assignee}</td></tr>"
        )
    no_tickets_row = (
        "<tr><td colspan='4' style='text-align:center;color:#95a5a6;'>"
        "No tickets created</td></tr>"
    )
    ticket_table = (
        f'<p><b>{created_n} of {total} proposed Jira ticket(s) created:</b></p>'
        '<table border="1" cellpadding="6" '
        'style="border-collapse:collapse;width:100%;margin-bottom:10px;">'
        '<tr style="background:#ecf0f1">'
        '<th>Ticket</th><th>Type</th><th>Summary</th><th>Assignee</th></tr>'
        f'{ticket_rows or no_tickets_row}'
        '</table>'
    )
    skipped_note = (
        f'<p style="color:#95a5a6;font-size:13px;">'
        f'ℹ️ {tickets_skipped} proposal(s) were skipped.</p>'
        if tickets_skipped else ""
    )

    html = (
        '<html><body style="font-family:Arial,sans-serif;'
        'max-width:700px;margin:auto;padding:20px;">'
        '<h2 style="color:#2c3e50;">Meeting Summary &amp; Action Report</h2>'
        f'<h3 style="color:#7f8c8d;">{file_name}</h3>'
        '<hr style="border:1px solid #ecf0f1;">'
        f'{summary_html_section}'
        '<h3 style="color:#2c3e50;">🎫 Jira Tickets</h3>'
        f'{ticket_table}'
        f'{skipped_note}'
        '<br><hr style="border:1px solid #ecf0f1;">'
    )

    # Include calendar link if calendar agent already confirmed, else show pending note
    calendar_link = meta.get("calendar_link", "") if 'meta' in locals() else ""
    if calendar_link:
        html += (
            f'<p style="color:#27ae60;font-style:italic;">'
            f'📅 <b>Calendar Invite:</b> Event created — '
            f'<a href="{calendar_link}">View calendar event</a></p>'
        )
    else:
        html += (
            '<p style="color:#7f8c8d;font-style:italic;">'
            '📅 <b>Calendar Invite:</b> The scheduling agent is resolving conflicts '
            'and will send you a finalized calendar invite separately.</p>'
        )

    html += (
        '<p style="color:#7f8c8d;font-size:13px;">'
        '📎 <b>Attached:</b> action_items.csv — all action items with priority '
        'and risk (includes items both tracked and not tracked in Jira).</p>'
        '<p style="color:#95a5a6;font-size:12px;">'
        'Generated by Meeting Summarizer Agent</p>'
        '</body></html>'
    )

    try:
        ses = boto3.client("ses", region_name="us-east-1")
        msg = mp.MIMEMultipart("mixed")
        msg["Subject"] = f"Meeting Report: {file_name}"
        msg["From"]    = SES_FROM_EMAIL
        msg["To"]      = ", ".join(PARTICIPANT_EMAILS)
        msg.attach(mt.MIMEText(html, "html"))
        if csv_bytes:
            part = mb.MIMEBase("text", "csv", name="action_items.csv")
            part.set_payload(csv_bytes)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename="action_items.csv")
            msg.attach(part)
        ses.send_raw_email(
            Source=SES_FROM_EMAIL,
            Destinations=PARTICIPANT_EMAILS,
            RawMessage={"Data": msg.as_string()},
        )
        logger.info(
            "_send_consolidated_email: sent to %s | created=%d skipped=%d has_csv=%s",
            PARTICIPANT_EMAILS, created_n, tickets_skipped, csv_bytes is not None,
        )
        WebClient(token=SLACK_BOT_TOKEN).chat_postMessage(
            channel=channel,
            text=(
                f"📧 Consolidated meeting report emailed to participants — "
                f"*{created_n} Jira ticket(s) created*, {tickets_skipped} skipped. "
                f"CSV attached."
            ),
        )
    except Exception as e:
        logger.error("_send_consolidated_email failed: %s", e)


def meeting_post_next_jira(state: OrchestratorState) -> OrchestratorState:
    """
    Pop the next item from the Jira proposal queue and post a Slack approval card.
    The queue lives in Redis. One proposal card at a time — matches Jira agent's
    single-request capacity. Each card has Create and Skip buttons.
    """
    # Resolve the queue session from either (a) state field set by send_email/create_jira
    # or (b) slack_action_value set by the skip button click
    queue_session_id = (
        state.meeting_jira_queue_session
        or (state.slack_action_value or {}).get("remaining_session_id")
    )
    if not queue_session_id:
        # No queue session — check if this is a last-card skip that carries
        # consolidated email data directly in slack_action_value
        av = state.slack_action_value or {}
        if (
            state.slack_action_id == "skip_meeting_jira"
            and av.get("is_last_card")
        ):
            logger.info("meeting_post_next_jira: last-card skip — sending consolidated email directly")
            # Mark the card as skipped in Slack
            if state.channel_id and state.preview_ts:
                try:
                    WebClient(token=SLACK_BOT_TOKEN).chat_update(
                        channel=state.channel_id, ts=state.preview_ts, blocks=None,
                        text="⏭️ Skipped.",
                    )
                except Exception as ue:
                    logger.warning("Could not update last skipped card: %s", ue)
            final_tickets = list(av.get("tickets_created", []))
            skipped_count = int(av.get("tickets_skipped", 0)) + 1  # count this skip
            _s3_key_hold   = av.get("s3_key", "")
            _channel_hold  = state.channel_id or SLACK_NOTIFY_CHANNEL or ""
            _fname_hold    = av.get("file_name", "meeting")
            CALENDAR_HOLD_TTL = 7200
            hold_key = f"calendar_hold:{_s3_key_hold}"
            _held = False
            try:
                from redis_store import _get_client as _get_r
                _r = _get_r()
                hold_payload = json.dumps({
                    "tickets_created":  final_tickets,
                    "tickets_skipped":  skipped_count,
                    "file_name":        _fname_hold,
                    "channel":          _channel_hold,
                    "s3_key":           _s3_key_hold,
                    "channel_id":       state.channel_id or "",
                    "preview_ts":       state.preview_ts or "",
                })
                _r.set(hold_key, hold_payload, ex=CALENDAR_HOLD_TTL)
                _held = True
                logger.info(
                    "meeting_post_next_jira: consolidated email held for calendar_done (key=%s)",
                    hold_key,
                )
            except Exception as _he:
                logger.warning("could not store calendar hold (%s) — sending immediately", _he)

            if not _held:
                _send_consolidated_email(
                    tickets_created=final_tickets,
                    tickets_skipped=skipped_count,
                    file_name=_fname_hold,
                    channel=_channel_hold,
                    s3_key=_s3_key_hold,
                )
            else:
                # Post calendar pending status to Slack
                try:
                    _slack = WebClient(token=SLACK_BOT_TOKEN)
                    _slack.chat_postMessage(
                        channel=_channel_hold,
                        text=(
                            "\U0001f4c5 *Calendar agent detecting conflicts...*\n"
                            "_Checking participant calendars and selecting the best available slot. "
                            "Consolidated email will be sent once the calendar event is confirmed._"
                        ),
                    )
                except Exception as _se:
                    logger.warning("could not post calendar pending message: %s", _se)
            return state
        logger.info("meeting_post_next_jira: no queue session, nothing to do")
        return state

    queue_data = load_session(queue_session_id)
    if not queue_data:
        logger.error(
            "meeting_post_next_jira: queue session %s missing — "
            "session expired or Redis error (check redis_store logs above)", queue_session_id
        )
        # Alert user in Slack so they know something went wrong
        notify_channel = state.channel_id or SLACK_NOTIFY_CHANNEL
        if notify_channel:
            try:
                WebClient(token=SLACK_BOT_TOKEN).chat_postMessage(
                    channel=notify_channel,
                    text=(
                        f"⚠️ Jira proposal queue session `{queue_session_id[:8]}...` "
                        "could not be loaded from Redis (expired or connection error). "
                        "Re-upload the transcript to regenerate proposals."
                    ),
                )
            except Exception as slack_err:
                logger.error("meeting_post_next_jira: could not post missing-session alert: %s", slack_err)
        return state

    items     = queue_data.get("items", [])
    channel   = queue_data.get("channel_id") or state.channel_id
    file_name = queue_data.get("file_name", "")

    logger.info(
        "meeting_post_next_jira: session loaded ok — %d item(s) remaining, channel=%s",
        len(items), channel,
    )

    if not items:
        logger.info("meeting_post_next_jira: queue empty — all proposals processed")
        # ── Update the last skipped/actioned card ──────────────────────────────
        if state.slack_action_id == "skip_meeting_jira" and state.channel_id and state.preview_ts:
            try:
                WebClient(token=SLACK_BOT_TOKEN).chat_update(
                    channel=state.channel_id, ts=state.preview_ts, blocks=None,
                    text="⏭️ Skipped.",
                )
            except Exception as ue:
                logger.warning("Could not update skipped card: %s", ue)

        # ── Load ticket tracking from queue session ────────────────────────────
        tickets_created = queue_data.get("tickets_created", [])
        tickets_skipped = queue_data.get("tickets_skipped", 0)

        # ── Send consolidated email to participants ─────────────────────────────
        s3_key_for_email = queue_data.get("s3_key", "")
        _send_consolidated_email(
            tickets_created=tickets_created,
            tickets_skipped=tickets_skipped,
            file_name=file_name,
            channel=channel,
            s3_key=s3_key_for_email,
        )
        return state

    # Pop first item, save remaining as new session
    current   = items[0]
    remaining = items[1:]
    remaining_session_id = None
    # Carry ticket tracking forward to the new remaining session
    tickets_created = list(queue_data.get("tickets_created", []))
    tickets_skipped = int(queue_data.get("tickets_skipped", 0))
    # If this is a skip action, increment skipped count
    if state.slack_action_id == "skip_meeting_jira":
        tickets_skipped += 1
        logger.info("meeting_post_next_jira: skip counted — total skipped=%d", tickets_skipped)

    if remaining:
        try:
            remaining_session_id = save_session({
                "flow": "meeting_jira_queue", "items": remaining,
                "channel_id": channel, "file_name": file_name,
                "s3_key": queue_data.get("s3_key", ""),
                "tickets_created": tickets_created,
                "tickets_skipped": tickets_skipped,
            })
            logger.info("meeting_post_next_jira: saved remaining queue session %s (%d items)", remaining_session_id, len(remaining))
        except Exception as e:
            logger.error("Could not save remaining Jira queue: %s", e)

    summary     = current.get("summary", "No summary")
    assignee    = current.get("assignee", current.get("owner", "TBD"))
    ticket_type = current.get("ticket_type", "Task")
    rationale   = current.get("rationale", "")
    emoji       = TICKET_TYPE_EMOJI.get(ticket_type, "🎫")
    remaining_n = len(remaining)
    queue_label = f"{remaining_n} more after this" if remaining_n else "last one"

    try:
        # Include s3_key, file_name, tickets tracking so last-ticket handler
        # has everything it needs for consolidated email
        _s3_key   = queue_data.get("s3_key", "")
        _fname    = queue_data.get("file_name", "")
        _tkts     = tickets_created  # accumulated so far
        _skipped  = tickets_skipped

        confirm_session = save_session({
            "flow": "meeting_jira", "current": current,
            "remaining_session_id": remaining_session_id, "channel_id": channel,
            "s3_key": _s3_key, "file_name": _fname,
            "tickets_created": _tkts, "tickets_skipped": _skipped,
        })
        skip_session = save_session({
            "flow": "meeting_jira", "current": None,
            "remaining_session_id": remaining_session_id, "channel_id": channel,
            "s3_key": _s3_key, "file_name": _fname,
            "tickets_created": _tkts, "tickets_skipped": _skipped,
            # Flag so skip handler knows this is the last card and must send consolidated email
            "is_last_card": remaining_session_id is None,
        })
        logger.info("meeting_post_next_jira: proposal sessions saved — confirm=%s skip=%s", confirm_session[:8], skip_session[:8])
    except Exception as e:
        logger.error("Could not save Jira proposal sessions: %s", e)
        try:
            WebClient(token=SLACK_BOT_TOKEN).chat_postMessage(
                channel=channel,
                text=f"⚠️ Could not post Jira ticket proposal (Redis save failed): {e}",
            )
        except Exception:
            pass
        return state

    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{emoji} *Proposed Jira Ticket* ({queue_label})\n"
                    f"*Summary:* {summary}\n"
                    f"*Assignee:* {assignee}  \u2022  *Type:* {ticket_type}\n"
                    f"_{rationale}_"
                ),
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button", "style": "primary",
                    "text":      {"type": "plain_text", "text": "\u2705 Create Ticket"},
                    "value":     confirm_session,
                    "action_id": "confirm_meeting_jira",
                },
                {
                    "type": "button",
                    "text":      {"type": "plain_text", "text": "\u23ed\ufe0f Skip"},
                    "value":     skip_session,
                    "action_id": "skip_meeting_jira",
                },
            ],
        },
    ]

    try:
        resp = WebClient(token=SLACK_BOT_TOKEN).chat_postMessage(
            channel=channel, blocks=blocks,
            text=f"Proposed Jira ticket: {summary}",
        )
        logger.info(
            "meeting_post_next_jira: posted proposal ts=%s channel=%s | %d remaining",
            resp.get("ts"), channel, remaining_n,
        )
    except Exception as e:
        logger.error(
            "meeting_post_next_jira: Slack post FAILED — channel=%s error=%s\n"
            "  summary: %s\n  confirm_session: %s",
            channel, e, summary[:80], confirm_session[:8],
        )

    # ── Update the card that was just skipped (remove buttons, mark skipped) ──
    if state.slack_action_id == "skip_meeting_jira" and state.channel_id and state.preview_ts:
        try:
            WebClient(token=SLACK_BOT_TOKEN).chat_update(
                channel=state.channel_id, ts=state.preview_ts, blocks=None,
                text="⏭️ Skipped.",
            )
            logger.info("meeting_post_next_jira: marked skipped card ts=%s", state.preview_ts)
        except Exception as ue:
            logger.warning("Could not update skipped card: %s", ue)

    return state.model_copy(update={"meeting_jira_queue_session": None})


# =============================================================================
# Node 11: meeting_create_jira
# Creates the ticket, updates Slack card, advances the queue.
# =============================================================================

def meeting_create_jira(state: OrchestratorState) -> OrchestratorState:
    """Create a Jira ticket from a proposal button click, then advance the queue."""
    value_dict           = state.slack_action_value or {}
    current              = value_dict.get("current") or {}
    remaining_session_id = value_dict.get("remaining_session_id")
    channel_id           = state.channel_id
    preview_ts           = state.preview_ts
    slack                = WebClient(token=SLACK_BOT_TOKEN)

    summary     = current.get("summary", "")
    assignee    = current.get("assignee", current.get("owner", ""))
    ticket_type = current.get("ticket_type", "Task")

    if not summary:
        logger.warning("meeting_create_jira: empty summary, skipping creation")
        return state.model_copy(update={"meeting_jira_queue_session": remaining_session_id})

    jira_key = _create_jira_ticket(
        summary=summary, assignee_name=assignee, ticket_type=ticket_type,
        description=f"Created from meeting summary via Meeting Summarizer Agent.\nAssignee: {assignee}",
    )

    if channel_id and preview_ts:
        try:
            if jira_key:
                slack.chat_update(
                    channel=channel_id, ts=preview_ts, blocks=None,
                    text=f"\u2705 Jira ticket created: <{JIRA_BASE_URL}/browse/{jira_key}|{jira_key}> \u2014 {summary[:60]}",
                )
            else:
                slack.chat_update(
                    channel=channel_id, ts=preview_ts, blocks=None,
                    text=f"\u26a0\ufe0f Failed to create Jira ticket for: {summary[:60]}",
                )
        except Exception as e:
            logger.warning("Could not update Jira creation card: %s", e)

    if state.session_id:
        try:
            record_feedback(state.session_id, "accepted" if jira_key else "failed", {
                "jira_key": jira_key, "summary": summary,
                "assignee": assignee, "ticket_type": ticket_type,
            })
        except Exception: pass

    # ── Track ticket in remaining queue session so consolidated email has data ─
    if remaining_session_id:
        try:
            remaining_data = load_session(remaining_session_id) or {}
            tickets_created = remaining_data.get("tickets_created", [])
            tickets_skipped = remaining_data.get("tickets_skipped", 0)
            if jira_key:
                tickets_created.append({
                    "jira_key":    jira_key,
                    "summary":     summary,
                    "assignee":    assignee,
                    "ticket_type": ticket_type,
                })
            # Save back
            from redis_store import _get_client as _redis_client
            import json as _json
            r = _redis_client()
            raw = r.get(remaining_session_id)
            if raw:
                rdata = _json.loads(raw)
                rdata["tickets_created"] = tickets_created
                rdata["tickets_skipped"] = tickets_skipped
                from state import SESSION_TTL_SECONDS
                r.setex(remaining_session_id, SESSION_TTL_SECONDS, _json.dumps(rdata))
                logger.info(
                    "meeting_create_jira: updated ticket tracking in session %s "
                    "(created=%d)", remaining_session_id[:8], len(tickets_created),
                )
        except Exception as te:
            logger.warning("Could not update ticket tracking in queue session: %s", te)
    elif jira_key or not remaining_session_id:
        # Last ticket in queue — no remaining session to update.
        # meeting_post_next_jira will get queue_session_id=None and exit early,
        # so we must trigger the consolidated email directly here.
        logger.info("meeting_create_jira: last ticket — triggering consolidated email directly")
        # Build final tickets_created list from value_dict context
        # Reconstruct full ticket list from accumulated session data + this ticket
        try:
            final_tickets = list(value_dict.get("tickets_created", []))
            if jira_key:
                final_tickets.append({
                    "jira_key":    jira_key,
                    "summary":     summary,
                    "assignee":    assignee,
                    "ticket_type": ticket_type,
                })
            skipped_count = int(value_dict.get("tickets_skipped", 0))
        except Exception:
            final_tickets = ([{"jira_key": jira_key, "summary": summary,
                                "assignee": assignee, "ticket_type": ticket_type}]
                             if jira_key else [])
            skipped_count = 0

        # Get s3_key and file_name from the channel/session context
        s3_key_for_email  = value_dict.get("s3_key", "")
        file_name_for_email = value_dict.get("file_name", "meeting")
        channel_for_email = channel_id or SLACK_NOTIFY_CHANNEL or ""

        # ── Hold consolidated email until calendar agent confirms ─────────────
        # Store the pending email payload in Redis so calendar_done webhook
        # can retrieve it and trigger the email with the calendar link included.
        CALENDAR_HOLD_TTL = 7200  # 2 hours — matches SESSION_TTL_SECONDS
        hold_key = f"calendar_hold:{s3_key_for_email}"
        try:
            from redis_store import _get_client as _get_r
            _r = _get_r()
            hold_payload = json.dumps({
                "tickets_created":  final_tickets,
                "tickets_skipped":  skipped_count,
                "file_name":        file_name_for_email,
                "channel":          channel_for_email,
                "s3_key":           s3_key_for_email,
                "channel_id":       channel_id or "",
                "preview_ts":       state.preview_ts or "",
            })
            _r.set(hold_key, hold_payload, ex=CALENDAR_HOLD_TTL)
            logger.info(
                "meeting_create_jira: consolidated email held — waiting for calendar_done (key=%s)",
                hold_key,
            )
        except Exception as _he:
            logger.warning(
                "meeting_create_jira: could not store calendar hold (%s) — sending email immediately",
                _he,
            )
            _send_consolidated_email(
                tickets_created=final_tickets,
                tickets_skipped=skipped_count,
                file_name=file_name_for_email,
                channel=channel_for_email,
                s3_key=s3_key_for_email,
            )

        # Update Slack to show calendar agent is working
        if channel_id:
            try:
                _slack = WebClient(token=SLACK_BOT_TOKEN)
                # Find the most recent message ts to update
                # (the last Jira card was already marked — post a new status message)
                _slack.chat_postMessage(
                    channel=channel_id,
                    text=(
                        "\U0001f4c5 *Calendar agent detecting conflicts...*\n"
                        "_Checking participant calendars and selecting the best available slot. "
                        "Consolidated email will be sent once the calendar event is confirmed._"
                    ),
                )
                logger.info("meeting_create_jira: posted calendar pending Slack message")
            except Exception as _se:
                logger.warning("meeting_create_jira: could not post calendar pending message: %s", _se)

    return state.model_copy(update={"meeting_jira_queue_session": remaining_session_id})


# =============================================================================
# Subgraph builder
# =============================================================================

def build_meeting_subgraph() -> StateGraph:
    """
    New transcript pipeline:
      fetch -> preprocess -> summarize -> triage_cod -> jira_cod
            -> artifacts -> s3 -> slack -> END

    Button click flows:
      confirm_summary      -> send_email -> post_next_jira -> END
      cancel_summary       -> post_cancel -> END
      confirm_meeting_jira -> create_jira -> post_next_jira -> END
      skip_meeting_jira    -> post_next_jira -> END
    """
    b = StateGraph(OrchestratorState)

    b.add_node("meeting_fetch_transcript",   meeting_fetch_transcript)
    b.add_node("meeting_preprocess",         meeting_preprocess)
    b.add_node("meeting_summarize",          meeting_summarize)
    b.add_node("meeting_triage_cod",         meeting_triage_cod)
    b.add_node("meeting_jira_cod",           meeting_jira_cod)
    b.add_node("meeting_generate_artifacts", meeting_generate_artifacts)
    b.add_node("meeting_store_s3",           meeting_store_s3)
    b.add_node("meeting_post_slack",         meeting_post_slack)
    b.add_node("meeting_send_email",         meeting_send_email)
    b.add_node("meeting_post_cancel",        meeting_post_cancel)
    b.add_node("meeting_post_next_jira",     meeting_post_next_jira)
    b.add_node("meeting_create_jira",        meeting_create_jira)

    b.add_conditional_edges(START, route_meeting_entry, {
        "meeting_fetch_transcript": "meeting_fetch_transcript",
        "meeting_send_email":       "meeting_send_email",
        "meeting_post_cancel":      "meeting_post_cancel",
        "meeting_create_jira":      "meeting_create_jira",
        "meeting_post_next_jira":   "meeting_post_next_jira",
    })

    # New transcript pipeline
    b.add_edge("meeting_fetch_transcript",   "meeting_preprocess")
    b.add_edge("meeting_preprocess",         "meeting_summarize")
    b.add_edge("meeting_summarize",          "meeting_triage_cod")
    b.add_edge("meeting_triage_cod",         "meeting_jira_cod")
    b.add_edge("meeting_jira_cod",           "meeting_generate_artifacts")
    b.add_edge("meeting_generate_artifacts", "meeting_store_s3")
    b.add_edge("meeting_store_s3",           "meeting_post_slack")
    b.add_edge("meeting_post_slack",         END)

    # Email confirm -> Jira queue
    b.add_edge("meeting_send_email",     "meeting_post_next_jira")
    b.add_edge("meeting_post_cancel",    END)

    # Jira queue flows
    b.add_edge("meeting_create_jira",    "meeting_post_next_jira")
    b.add_edge("meeting_post_next_jira", END)

    return b.compile()
