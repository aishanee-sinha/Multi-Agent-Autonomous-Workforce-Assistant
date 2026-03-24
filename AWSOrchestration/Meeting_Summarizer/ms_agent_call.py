"""
lambda_function.py  —  Meeting Summarizer Agent (v4)
Fixes vs previous version:
  - Duplicate Slack messages: Apps Script marks file processed only on
    HTTP 200; Lambda now returns 200 immediately and logs any internal
    error without returning 500 to Apps Script.
  - Action Items (0) display bug: n_actions now uses parsed actions_json length.
  - ICS time: extracts deadline/time from action items; falls back to
    next occurrence of common meeting-time phrases in the transcript.
  - Preprocessor handles both real (fragmented) and structured transcripts.
  - Chunked inference handles meetings of any length.
"""

import os
import re
import json
import time
import logging
import base64
import urllib.parse
import io
import boto3
import requests
from datetime import datetime, timedelta
from slack_sdk import WebClient
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────────
EC2_IP             = os.environ.get("EC2_IP", "")
SLACK_BOT_TOKEN    = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID   = os.environ.get("SLACK_CHANNEL_ID", "")
SES_FROM_EMAIL     = os.environ.get("SES_FROM_EMAIL", "")
PARTICIPANT_EMAILS = [e.strip() for e in os.environ.get("PARTICIPANT_EMAILS", "").split(",") if e.strip()]
S3_BUCKET          = os.environ.get("S3_BUCKET", "qwen-lora-weights")
S3_PREFIX          = os.environ.get("S3_TRANSCRIPT_PREFIX", "transcript_summarizer")
VLLM_MODEL_NAME    = os.environ.get("VLLM_MODEL_NAME", "meeting")

SCOPES        = ["https://www.googleapis.com/auth/drive.readonly"]
CHUNK_SIZE    = 10000
CHUNK_OVERLAP = 1000

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
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
    "Rules:\n"
    "  - Do not fabricate facts not present in the transcript.\n"
    "  - Set owner/deadline to TBD when not mentioned.\n"
    "  - ACTIONS_JSON must parse as valid JSON.\n"
    "  - discussed_at_sec should be a float or 0.0 if unknown."
)


def build_prompt(transcript: str) -> list:
    return [
        {"role": "system", "content": P4_SYSTEM_PROMPT},
        {"role": "user",   "content": f"{P4_USER_INSTRUCTION}\n\nTRANSCRIPT:\n{transcript}"},
    ]


# ─────────────────────────────────────────────
# Preprocessor — handles real recorded transcripts
# ─────────────────────────────────────────────
def preprocess_transcript(transcript: str) -> str:
    """
    Converts fragmented real meeting transcripts into clean model-friendly format.
    Removes filler lines, merges consecutive same-speaker utterances.
    Falls back to original if transcript is already well-structured.
    """
    FILLERS = {
        "hmm", "yeah", "okay", "ok", "uh", "um", "huh", "mm",
        "mm-hmm", "yes", "no", "right", "sure", "bye", "hello",
        "hi", "hey", "oh", "ah", "hm", "yep", "nope", "alright",
        "good", "great", "cool", "nice", "k",
    }

    speaker_re = re.compile(r"^\[(.+?)\]\s*[\d:]+\s*$")
    lines = transcript.splitlines()

    utterances      = []
    current_speaker = None
    current_text    = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = speaker_re.match(line)
        if m:
            if current_speaker and current_text:
                utterances.append((current_speaker, " ".join(current_text)))
            current_speaker = m.group(1).strip()
            current_text    = []
        elif current_speaker:
            if len(line) < 5:
                continue
            clean = re.sub(r"[^a-zA-Z\s]", "", line).strip().lower()
            words = clean.split()
            if words and all(w in FILLERS for w in words):
                continue
            current_text.append(line)

    if current_speaker and current_text:
        utterances.append((current_speaker, " ".join(current_text)))

    merged = []
    for speaker, text in utterances:
        if merged and merged[-1][0] == speaker:
            merged[-1][1] = merged[-1][1] + " " + text
        else:
            merged.append([speaker, text])

    result_lines = [f"[{spk}]: {txt.strip()}" for spk, txt in merged if txt.strip()]
    processed    = "\n".join(result_lines)

    if len(processed) < 200 and len(transcript) > 500:
        logger.info("Preprocessor minimal output — using original transcript")
        return transcript

    logger.info(f"Preprocessed: {len(transcript)} → {len(processed)} chars ({len(result_lines)} turns)")
    return processed


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
_SECTION_RE = re.compile(
    r"(ABSTRACT|DECISIONS|PROBLEMS|ACTIONS):\s*(.*?)(?=\n(?:ABSTRACT|DECISIONS|PROBLEMS|ACTIONS|ACTIONS_JSON):|$)",
    re.DOTALL | re.IGNORECASE,
)
_ACTIONS_JSON_RE = re.compile(r"ACTIONS_JSON:\s*(\[.*?\])", re.DOTALL | re.IGNORECASE)


def clean_structured_summary(raw: str) -> str:
    headers  = ["ABSTRACT:", "DECISIONS:", "ACTIONS:", "PROBLEMS:"]
    earliest = len(raw)
    for h in headers:
        pos = raw.upper().find(h)
        if pos != -1 and pos < earliest:
            earliest = pos
    return raw[earliest:].strip() if earliest < len(raw) else raw.strip()


def extract_section(text: str, section: str) -> str:
    pattern = rf"{section}:\s*(.*?)(?=\n(?:ABSTRACT|DECISIONS|PROBLEMS|ACTIONS|ACTIONS_JSON):|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_actions_json_block(text: str):
    m = _ACTIONS_JSON_RE.search(text)
    if not m:
        return [], False, "not found"
    candidate = m.group(1).strip()
    try:
        return json.loads(candidate), True, ""
    except json.JSONDecodeError as e:
        try:
            fixed = candidate[:candidate.rfind("]") + 1]
            return json.loads(fixed), True, f"repaired: {e}"
        except Exception:
            return [], False, str(e)


# ─────────────────────────────────────────────
# ICS time parser
# ─────────────────────────────────────────────
def parse_meeting_datetime(transcript: str, action_items: list) -> datetime:
    """
    Extract a specific meeting time from transcript or action items.
    Looks for patterns like '9pm', '9 PM', '21:00', 'tonight at 9'.
    Falls back to next occurrence of 9pm today if mentioned,
    otherwise uses now + 1 hour.
    """
    now = datetime.utcnow()

    # Check action item deadlines first
    for ai in action_items:
        deadline = str(ai.get("deadline", "")).strip().lower()
        time_match = re.search(
            r"(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)", deadline, re.IGNORECASE
        )
        if time_match:
            hour = int(time_match.group(1))
            mins = int(time_match.group(2) or 0)
            ampm = time_match.group(3).lower()
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            try:
                dt = now.replace(hour=hour, minute=mins, second=0, microsecond=0)
                if dt < now:
                    dt += timedelta(days=1)
                return dt
            except ValueError:
                pass

    # Search transcript for time references
    time_patterns = [
        r"(?:at|by|@)\s*(\d{1,2})\s*(?::(\d{2}))?\s*(am|pm)",
        r"(\d{1,2})\s*(am|pm)\s*(?:today|tonight|sharp)",
        r"(\d{1,2}):(\d{2})\s*(am|pm)",
    ]
    for pattern in time_patterns:
        m = re.search(pattern, transcript, re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g]
            try:
                if len(groups) >= 2 and groups[-1].lower() in ("am", "pm"):
                    hour = int(groups[0])
                    ampm = groups[-1].lower()
                    mins = int(groups[1]) if len(groups) >= 3 else 0
                    if ampm == "pm" and hour != 12:
                        hour += 12
                    elif ampm == "am" and hour == 12:
                        hour = 0
                    dt = now.replace(hour=hour, minute=mins, second=0, microsecond=0)
                    if dt < now:
                        dt += timedelta(days=1)
                    return dt
            except (ValueError, IndexError):
                continue

    # Fallback: now + 1 hour
    return now + timedelta(hours=1)


# ─────────────────────────────────────────────
# Main Lambda entry point
# ─────────────────────────────────────────────
def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")

    headers = {k.lower(): v for k, v in event.get("headers", {}).items()}

    if int(headers.get("x-slack-retry-num", 0)) > 0:
        return {"statusCode": 200, "body": "Ignoring retry"}

    body_raw = event.get("body", "")
    if event.get("isBase64Encoded"):
        body_raw = base64.b64decode(body_raw).decode("utf-8")

    if body_raw.startswith("payload="):
        payload_str = urllib.parse.unquote_plus(body_raw.split("payload=")[1])
        payload     = json.loads(payload_str)
        return handle_interactivity(payload)

    body = json.loads(body_raw)

    if body.get("type") == "url_verification":
        return {"statusCode": 200, "body": json.dumps({"challenge": body.get("challenge")})}

    if body.get("type") == "new_transcript":
        # Always return 200 to Apps Script immediately so it does NOT retry.
        # Process the transcript and catch all errors internally.
        try:
            handle_trigger(body)
        except Exception as e:
            logger.error(f"handle_trigger error (returning 200 anyway): {e}")
        return {"statusCode": 200, "body": "ok"}

    return {"statusCode": 400, "body": "Unknown event type"}


# ─────────────────────────────────────────────
# SECTION A: Trigger handler
# ─────────────────────────────────────────────
def handle_trigger(body: dict):
    file_id   = body.get("file_id")
    file_name = body.get("file_name", "transcript.txt")

    logger.info(f"Processing: {file_name} (id={file_id})")

    # Download
    transcript_raw = download_from_drive(file_id)
    logger.info(f"Downloaded: {len(transcript_raw)} chars")

    # Preprocess
    transcript = preprocess_transcript(transcript_raw)

    # Chunked inference
    raw_output = run_inference_chunked(transcript)
    logger.info("Inference complete")

    # Parse
    parsed    = parse_model_output(raw_output)
    n_actions = len(parsed.get("actions_json", []))
    logger.info(f"Parsed: {n_actions} action items")

    # Generate ICS with correct time
    ics_bytes = None
    if parsed.get("actions_json"):
        meeting_dt = parse_meeting_datetime(transcript_raw, parsed["actions_json"])
        logger.info(f"ICS meeting time: {meeting_dt.isoformat()}")
        ics_bytes  = generate_ics(file_name, parsed["actions_json"], meeting_dt)

    # Generate CSV
    csv_bytes = generate_csv(file_name, parsed.get("actions_json", []))

    # Store in S3 and get back the exact s3_key
    s3_key = store_in_s3(file_name, transcript_raw, raw_output, parsed, ics_bytes, csv_bytes)

    # Post to Slack using the same s3_key
    post_summary_to_slack(
        file_name=file_name,
        s3_key=s3_key,
        parsed=parsed,
        ics_bytes=ics_bytes,
    )


# ─────────────────────────────────────────────
# SECTION B: Interactivity handler
# ─────────────────────────────────────────────
def handle_interactivity(payload: dict) -> dict:
    slack_client = WebClient(token=SLACK_BOT_TOKEN)
    action       = payload["actions"][0]
    channel_id   = payload["channel"]["id"]
    message_ts   = payload["container"]["message_ts"]
    user_name    = payload.get("user", {}).get("name", "Someone")

    if action["action_id"] == "confirm_summary":
        try:
            data      = json.loads(urllib.parse.unquote_plus(action["value"]))
            file_name = data.get("file_name", "meeting")
            s3_key    = data.get("s3_key", "")

            s3   = boto3.client("s3")
            meta = json.loads(
                s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/meta.json")["Body"].read()
            )
            ics_bytes = None
            csv_bytes = None
            try:
                ics_bytes = s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/invite.ics")["Body"].read()
            except Exception:
                pass
            try:
                csv_bytes = s3.get_object(Bucket=S3_BUCKET, Key=f"{s3_key}/actions.csv")["Body"].read()
            except Exception:
                pass

            send_email_via_ses(
                file_name=file_name,
                summary_text=meta.get("summary_text", ""),
                ics_bytes=ics_bytes,
                csv_bytes=csv_bytes,
            )
            slack_client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text=f"Summary confirmed by {user_name}. Email sent to: {', '.join(PARTICIPANT_EMAILS)}",
                blocks=None,
            )
            logger.info(f"Email sent. Confirmed by {user_name}.")
        except Exception as e:
            logger.error(f"Confirm error: {e}")
            slack_client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text=f"Error sending email: {str(e)}",
                blocks=None,
            )
    else:
        slack_client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text=f"Summary dismissed by {user_name}. No email sent.",
            blocks=None,
        )

    return {"statusCode": 200, "body": ""}


# ─────────────────────────────────────────────
# SECTION C: Google Drive
# ─────────────────────────────────────────────
def get_drive_service():
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "{}")
    sa_info = json.loads(sa_json)
    creds   = service_account.Credentials.from_service_account_info(sa_info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def download_from_drive(file_id: str) -> str:
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    fh      = io.BytesIO()
    dl      = MediaIoBaseDownload(fh, request)
    done    = False
    while not done:
        _, done = dl.next_chunk()
    return fh.getvalue().decode("utf-8")


# ─────────────────────────────────────────────
# SECTION D: Chunked inference
# ─────────────────────────────────────────────
def run_inference_chunked(transcript: str, max_new_tokens: int = 768) -> str:
    endpoint = f"http://{EC2_IP}:8000/v1/chat/completions"
    headers  = {"Content-Type": "application/json"}

    chunks = []
    start  = 0
    while start < len(transcript):
        end = start + CHUNK_SIZE
        chunks.append(transcript[start:end])
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            break
        start = next_start

    logger.info(f"Transcript: {len(transcript)} chars | {len(chunks)} chunks")

    all_abstracts    = []
    all_decisions    = []
    all_problems     = []
    all_actions      = []
    all_action_jsons = []

    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)")
        payload = {
            "model":       VLLM_MODEL_NAME,
            "messages":    build_prompt(chunk),
            "max_tokens":  max_new_tokens,
            "temperature": 0.0,
            "top_p":       1.0,
        }
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            cleaned = clean_structured_summary(content)

            abstract  = extract_section(cleaned, "ABSTRACT")
            decisions = extract_section(cleaned, "DECISIONS")
            problems  = extract_section(cleaned, "PROBLEMS")
            actions   = extract_section(cleaned, "ACTIONS")

            action_list, valid, _ = parse_actions_json_block(cleaned)
            if valid:
                all_action_jsons.extend(action_list)

            if abstract:  all_abstracts.append(abstract)
            if decisions: all_decisions.append(decisions)
            if problems:  all_problems.append(problems)
            if actions:   all_actions.append(actions)
        except Exception as e:
            logger.error(f"Chunk {i + 1} failed: {e}")

    def _dedup_lines(blocks):
        seen, out = set(), []
        for block in blocks:
            for line in block.splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    out.append(line)
        return "\n".join(out)

    merged  = f"ABSTRACT:\n{all_abstracts[0] if all_abstracts else 'No abstract generated.'}\n\n"
    merged += f"DECISIONS:\n{_dedup_lines(all_decisions) or 'None identified.'}\n\n"
    merged += f"PROBLEMS:\n{_dedup_lines(all_problems) or 'None identified.'}\n\n"
    merged += f"ACTIONS:\n{_dedup_lines(all_actions) or 'None identified.'}\n\n"

    seen_tasks, deduped = set(), []
    for ai in all_action_jsons:
        key = str(ai.get("task", ""))[:80].lower().strip()
        if key and key not in seen_tasks:
            seen_tasks.add(key)
            deduped.append(ai)

    merged += f"ACTIONS_JSON:\n{json.dumps(deduped, indent=2)}"
    logger.info(f"Merged: {len(deduped)} unique action items from {len(chunks)} chunks")
    return merged


# ─────────────────────────────────────────────
# SECTION E: Output parsing
# ─────────────────────────────────────────────
def parse_model_output(raw: str) -> dict:
    result = {"abstract": "", "decisions": "", "problems": "", "actions": "", "actions_json": [], "raw_output": raw}

    for match in _SECTION_RE.finditer(raw):
        section = match.group(1).upper()
        content = match.group(2).strip()
        if section == "ABSTRACT":   result["abstract"]  = content
        elif section == "DECISIONS": result["decisions"] = content
        elif section == "PROBLEMS":  result["problems"]  = content
        elif section == "ACTIONS":   result["actions"]   = content

    action_list, valid, _ = parse_actions_json_block(raw)
    if valid:
        result["actions_json"] = action_list

    return result


# ─────────────────────────────────────────────
# SECTION F: ICS generation with correct time
# ─────────────────────────────────────────────
def generate_ics(meeting_name: str, action_items: list, meeting_dt: datetime = None) -> bytes:
    if meeting_dt is None:
        meeting_dt = datetime.utcnow() + timedelta(hours=1)

    meeting_end = meeting_dt + timedelta(hours=1)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Meeting Summarizer 14B//EN",
        "BEGIN:VEVENT",
        f"DTSTART:{meeting_dt.strftime('%Y%m%dT%H%M%SZ')}",
        f"DTEND:{meeting_end.strftime('%Y%m%dT%H%M%SZ')}",
        f"SUMMARY:Meeting Summary - {meeting_name}",
        "END:VEVENT",
    ]
    for ai in action_items:
        owner    = str(ai.get("owner",    "TBD"))
        task     = str(ai.get("task",     ""))[:200]
        deadline = str(ai.get("deadline", "TBD"))
        lines += [
            "BEGIN:VTODO",
            f"SUMMARY:[{owner}] {task}",
            f"DESCRIPTION:Owner: {owner}\\nTask: {task}\\nDeadline: {deadline}",
            "END:VTODO",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines).encode("utf-8")


# ─────────────────────────────────────────────
# SECTION G: CSV generation
# ─────────────────────────────────────────────
def generate_csv(meeting_name: str, action_items: list) -> bytes:
    import csv, io as _io
    buf    = _io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["meeting", "poc_name", "task_description", "deadline", "discussed_at_sec"],
        lineterminator="\r\n",
    )
    writer.writeheader()
    for ai in action_items:
        writer.writerow({
            "meeting":          meeting_name,
            "poc_name":         str(ai.get("owner",    "TBD")),
            "task_description": str(ai.get("task",     "")),
            "deadline":         str(ai.get("deadline", "TBD")),
            "discussed_at_sec": str(ai.get("discussed_at_sec", 0.0)),
        })
    return buf.getvalue().encode("utf-8")


# ─────────────────────────────────────────────
# SECTION H: S3 storage
# ─────────────────────────────────────────────
def store_in_s3(file_name, transcript, raw_output, parsed, ics_bytes, csv_bytes) -> str:
    s3        = boto3.client("s3")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = file_name.replace(".txt", "").replace(" ", "_")
    s3_key    = f"{S3_PREFIX}/meetings/{timestamp}_{base_name}"

    s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/transcript.txt",
                  Body=transcript.encode("utf-8"), ContentType="text/plain")
    s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/raw_output.txt",
                  Body=raw_output.encode("utf-8"), ContentType="text/plain")

    summary_text = build_summary_text(parsed)
    meta = {
        "file_name":     file_name,
        "s3_key":        s3_key,
        "timestamp":     timestamp,
        "summary_text":  summary_text,
        "abstract":      parsed.get("abstract", ""),
        "decisions":     parsed.get("decisions", ""),
        "problems":      parsed.get("problems", ""),
        "actions":       parsed.get("actions", ""),
        "actions_count": len(parsed.get("actions_json", [])),
    }
    s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/meta.json",
                  Body=json.dumps(meta, indent=2).encode("utf-8"), ContentType="application/json")
    if ics_bytes:
        s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/invite.ics",
                      Body=ics_bytes, ContentType="text/calendar")
    if csv_bytes:
        s3.put_object(Bucket=S3_BUCKET, Key=f"{s3_key}/actions.csv",
                      Body=csv_bytes, ContentType="text/csv")

    logger.info(f"Stored in S3: s3://{S3_BUCKET}/{s3_key}/")
    return s3_key


def build_summary_text(parsed: dict) -> str:
    parts = []
    for key, label in [("abstract","ABSTRACT"),("decisions","DECISIONS"),
                       ("problems","PROBLEMS"),("actions","ACTIONS")]:
        if parsed.get(key):
            parts.append(f"{label}:\n{parsed[key]}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# SECTION I: Slack posting
# ─────────────────────────────────────────────
def post_summary_to_slack(file_name, s3_key, parsed, ics_bytes):
    slack_client = WebClient(token=SLACK_BOT_TOKEN)

    abstract  = parsed.get("abstract",  "No abstract generated.")[:300]
    decisions = parsed.get("decisions", "None.") or "None."
    problems  = parsed.get("problems",  "None.") or "None."
    actions   = parsed.get("actions",   "None.") or "None."
    # Use actual parsed action count — fixes Action Items (0) display bug
    n_actions = len(parsed.get("actions_json", []))
    has_ics   = ics_bytes is not None

    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"Meeting Summary: {file_name}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Abstract*\n{abstract}"}},
        {"type": "divider"},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*Decisions*\n{decisions[:400]}"},
            {"type": "mrkdwn", "text": f"*Problems / Risks*\n{problems[:400]}"},
        ]},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            f"*Action Items ({n_actions})*\n{actions[:500]}"
            + ("\n\n_ICS calendar invite included._" if has_ics else "")
        )}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": (
            f"*Ready to send to:* {', '.join(PARTICIPANT_EMAILS)}\n"
            "Click *Confirm* to email summary + ICS + action items CSV.\n"
            "Click *Cancel* to dismiss."
        )}},
        {"type": "actions", "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Confirm - Send Email"},
             "style": "primary",
             "value": json.dumps({"file_name": file_name, "s3_key": s3_key}),
             "action_id": "confirm_summary"},
            {"type": "button", "text": {"type": "plain_text", "text": "Cancel"},
             "style": "danger", "action_id": "cancel_summary"},
        ]},
    ]

    response = slack_client.chat_postMessage(
        channel=SLACK_CHANNEL_ID,
        blocks=blocks,
        text=f"New meeting summary ready: {file_name}",
    )
    logger.info(f"Posted to Slack {SLACK_CHANNEL_ID}, ts={response['ts']}")


# ─────────────────────────────────────────────
# SECTION J: SES email
# ─────────────────────────────────────────────
def send_email_via_ses(file_name, summary_text, ics_bytes, csv_bytes):
    import email.mime.multipart as mp
    import email.mime.text      as mt
    import email.mime.base      as mb
    import email.encoders       as encoders

    if not PARTICIPANT_EMAILS:
        raise ValueError("PARTICIPANT_EMAILS is empty")
    if not SES_FROM_EMAIL:
        raise ValueError("SES_FROM_EMAIL is empty")

    ses     = boto3.client("ses", region_name="us-east-1")
    msg     = mp.MIMEMultipart("mixed")
    msg["Subject"] = f"Meeting Summary: {file_name}"
    msg["From"]    = SES_FROM_EMAIL
    msg["To"]      = ", ".join(PARTICIPANT_EMAILS)

    msg.attach(mt.MIMEText(build_html_email(file_name, summary_text), "html"))

    if ics_bytes:
        part = mb.MIMEBase("text", "calendar", method="REQUEST", name="invite.ics")
        part.set_payload(ics_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename="invite.ics")
        msg.attach(part)

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
    logger.info(f"Email sent to: {PARTICIPANT_EMAILS}")


def build_html_email(file_name: str, summary_text: str) -> str:
    body_html = summary_text.replace("\n", "<br>")
    return f"""
    <html>
    <body style="font-family:Arial,sans-serif;max-width:700px;margin:auto;padding:20px;">
      <h2 style="color:#2c3e50;">Meeting Summary</h2>
      <h3 style="color:#7f8c8d;">{file_name}</h3>
      <hr style="border:1px solid #ecf0f1;">
      <div style="line-height:1.8;color:#34495e;">{body_html}</div>
      <hr style="border:1px solid #ecf0f1;">
      <p style="color:#95a5a6;font-size:12px;">
        Generated by Meeting Summarizer Agent (Qwen2.5-14B Fine-tuned)<br>
        Attachments: ICS calendar invite, Action items CSV
      </p>
    </body>
    </html>
    """