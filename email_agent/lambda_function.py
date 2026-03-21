import os
import json
import base64
import logging
import re
import email.mime.multipart
import email.mime.text
from datetime import datetime, timezone, timedelta
import urllib.request

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Config ────────────────────────────────────────────────────────────────────
EC2_IP       = os.environ.get("EC2_IP", "YOUR_EC2_IP")
CONFIRM_URL  = os.environ.get("CONFIRM_URL", "YOUR_LAMBDA_FUNCTION_URL")
GROUP_EMAILS = json.loads(os.environ.get("GROUP_EMAILS_JSON", '["user1@gmail.com", "user2@gmail.com"]'))
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "your-sender@gmail.com")

# Token loaded exclusively from Lambda environment variable — never hardcode
GOOGLE_TOKEN = os.environ.get("GOOGLE_TOKEN_JSON", "")
# ── EXACT system prompt from your Colab notebook ──────────────────────────────
SYSTEM_PROMPT = """You are a specialized meeting information extraction system.

TASK: Analyze emails and extract structured meeting details with high precision.

OUTPUT FORMAT (JSON only, no explanations):
{
  "is_meeting": boolean,
  "title": string or null,
  "attendees": array of email addresses,
  "start_time": ISO 8601 string or null,
  "end_time": ISO 8601 string or null,
  "location": string or null,
  "time_confidence": "high" | "medium" | "low" | "none"
}

EXTRACTION RULES:

1. MEETING DETECTION (is_meeting):
   - TRUE if email discusses scheduling/planning a meeting, call, discussion, sync, or appointment
   - TRUE if contains meeting invitations, calendar entries, or time/location details
   - FALSE for: announcements, reports, questions, general updates, FYIs
   - Keywords: meeting, schedule, calendar, invite, call, discussion, sync, appointment

2. TITLE EXTRACTION:
   - Extract meeting subject/purpose (max 100 characters)
   - Use email subject line if it describes the meeting
   - Keep concise: "Team Sync", "Q1 Planning Meeting", "Product Review"
   - Set to null for non-meetings

3. ATTENDEES EXTRACTION:
   - Extract ALL email addresses mentioned in the email
   - Include: To, Cc, mentioned in body
   - Format: lowercase, exact email addresses only
   - Empty array [] for non-meetings or if no emails found

4. START_TIME EXTRACTION:
   - Convert ALL times to ISO 8601 UTC format: "YYYY-MM-DDTHH:MM:SSZ"
   - Parse formats: "March 15 at 2pm", "tomorrow 3:00 PM", "Monday at 9am"
   - Set to null if no time mentioned or non-meeting

5. END_TIME EXTRACTION:
   - Same format as start_time: "YYYY-MM-DDTHH:MM:SSZ"
   - Extract if explicitly mentioned (e.g., "2pm-3pm", "from 9am to 10am")
   - Set to null if not specified or non-meeting

6. LOCATION EXTRACTION:
   - Extract specific locations: room numbers, addresses, virtual platforms
   - Examples: "Room 301", "Zoom", "Google Meet", "Conference Room A"
   - Set to null if not mentioned or non-meeting

7. TIME CONFIDENCE:
   - "high"   = exact date AND time provided
   - "medium" = date provided but time is approximate or missing
   - "low"    = only vague references like "this week", "tomorrow"
   - "none"   = no time information or not a meeting

CRITICAL: ALWAYS return valid JSON. NEVER add explanations."""


# ── TOKEN HELPERS ─────────────────────────────────────────────────────────────

def encode_pending(email_data: dict, model_output: dict) -> str:
    """Encode meeting data directly into the confirmation token."""
    data = json.dumps({"e": email_data, "m": model_output})
    return base64.urlsafe_b64encode(data.encode()).decode()


def decode_pending(token: str) -> dict:
    """Decode meeting data from confirmation token."""
    try:
        data   = base64.urlsafe_b64decode(token.encode() + b"==").decode()
        parsed = json.loads(data)
        return {
            "email_data":   json.dumps(parsed["e"]),
            "model_output": json.dumps(parsed["m"])
        }
    except Exception as e:
        logger.error(f"Decode error: {e}")
        return None


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"Event received: {json.dumps(event)}")

    # TEMP DEBUG — test email sending
    if event.get("test_email"):
        try:
            from googleapiclient.discovery import build
            creds   = _get_google_creds()
            service = build("gmail", "v1", credentials=creds)
            msg     = email.mime.multipart.MIMEMultipart("alternative")
            msg["Subject"] = "Test from Lambda"
            msg["From"]    = SENDER_EMAIL
            msg["To"]      = SENDER_EMAIL
            msg.attach(email.mime.text.MIMEText("<h1>Lambda email works!</h1>", "html"))
            raw    = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            result = service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            return {"statusCode": 200, "body": f"Email sent: {result['id']}"}
        except Exception as e:
            return {"statusCode": 500, "body": f"FAILED: {str(e)}"}

    # Route 1: confirm/cancel button clicked from email
    if event.get("queryStringParameters"):
        return handle_confirmation(event)

    # Route 2: real Gmail Pub/Sub push
    if "body" in event and "subject" not in event:
        return handle_gmail_push(event)

    # Route 3: direct test invocation
    return handle_email_event(event)


# ── ROUTE 1: Gmail Pub/Sub push ───────────────────────────────────────────────

def handle_gmail_push(event):
    """Called when Gmail sends a Pub/Sub push notification."""
    try:
        body = event.get("body", "")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")

        data           = json.loads(body)
        pubsub_message = data.get("message", {})

        if not pubsub_message:
            logger.info("No Pub/Sub message found")
            return {"statusCode": 200, "body": "no message"}

        decoded    = json.loads(
            base64.b64decode(pubsub_message["data"]).decode("utf-8")
        )
        user_email = decoded.get("emailAddress", "")
        history_id = str(decoded.get("historyId", ""))

        logger.info(f"Gmail push for {user_email}, historyId={history_id}")

        emails = fetch_new_emails(history_id)
        logger.info(f"Found {len(emails)} new emails to process")

        import threading
        def process_emails():
            for email_data in emails:
                logger.info(f"Processing: from={email_data.get('from_email')} subject={email_data.get('subject')}")
                result = handle_email_event(email_data)
                logger.info(f"Result: {result}")

        threading.Thread(target=process_emails).start()
        return {"statusCode": 200, "body": "ok"}

    except Exception as e:
        logger.error(f"Push handler error: {e}")
        return {"statusCode": 200, "body": "ok"}


# ── ROUTE 2: Process a single email ──────────────────────────────────────────

def handle_email_event(email_data):
    """Calls EC2 model, sends confirmation if meeting detected."""
    if isinstance(email_data, dict) and email_data.get("Records"):
        email_data = json.loads(email_data["Records"][0]["body"])

    sender    = email_data.get("from_email", "").lower()
    group_set = {e.lower() for e in GROUP_EMAILS}

    if GROUP_EMAILS and sender not in group_set:
        logger.info(f"Sender {sender} not in group — skipping")
        return {"statusCode": 200, "body": "skipped"}

    model_output = call_ec2_model(email_data)
    logger.info(f"Model output: {json.dumps(model_output)}")

    if not model_output.get("is_meeting"):
        logger.info("Not a meeting — skipping")
        return {"statusCode": 200, "body": "not a meeting"}

    token = encode_pending(email_data, model_output)

    send_confirmation_email(
        email_data.get("from_email", ""),
        model_output,
        token
    )

    return {"statusCode": 200, "body": "confirmation sent"}


# ── ROUTE 3: Confirm/Cancel button clicked ────────────────────────────────────

def handle_confirmation(event):
    """User clicked Yes or No in the confirmation email."""
    params = event.get("queryStringParameters", {})
    token  = params.get("token", "")
    action = params.get("action", "")

    pending = decode_pending(token)
    if not pending:
        return _html(404, "Link expired",
                     "This confirmation link has expired or already been used.")

    if action == "confirm":
        model_output   = json.loads(pending["model_output"])
        original_email = json.loads(pending["email_data"])
        link           = create_calendar_event(model_output, original_email)
        title          = model_output.get("title", "Meeting")
        return _html(
            200,
            "Event Created!",
            f'<b>{title}</b> has been added to Google Calendar!<br><br>'
            f'<a href="{link}" style="color:#1a73e8">Open in Google Calendar</a>'
        )
    else:
        return _html(200, "Cancelled", "No calendar event was created.")


# ── EC2 MODEL CALL ────────────────────────────────────────────────────────────

def call_ec2_model(email_data: dict) -> dict:
    """Calls vLLM on EC2 using the email LoRA adapter."""
    email_text = (
        f"Subject: {email_data.get('subject', '')}\n"
        f"From: {email_data.get('from_email', '')}\n"
        f"To: {', '.join(email_data.get('to_emails', []))}\n"
        f"Cc: {', '.join(email_data.get('cc_emails', []))}\n"
        f"Date: {email_data.get('date', '')}\n\n"
        f"Body:\n{email_data.get('body', '')[:2000]}"
    )

    payload = json.dumps({
        "model":       "email",
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": email_text}
        ],
        "max_tokens":  1024,
        "temperature": 0.1,
        "top_p":       1.0,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"http://{EC2_IP}:8000/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            result   = json.loads(resp.read())
            raw_text = result["choices"][0]["message"]["content"]
            logger.info(f"Raw model output: {raw_text}")

            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0]

            match  = re.search(r"\{.*\}", raw_text, re.DOTALL)
            parsed = json.loads(match.group(0) if match else raw_text.strip())

            if parsed.get("time_confidence") is None:
                parsed["time_confidence"] = "none"

            return parsed

    except Exception as e:
        logger.error(f"EC2 model call failed: {e}")
        return {
            "is_meeting":      False,
            "title":           None,
            "attendees":       [],
            "start_time":      None,
            "end_time":        None,
            "location":        None,
            "time_confidence": "none"
        }


# ── FETCH EMAILS FROM GMAIL ───────────────────────────────────────────────────

def fetch_new_emails(history_id: str) -> list:
    """Fetch new emails since historyId using Gmail API."""
    try:
        from googleapiclient.discovery import build
        creds   = _get_google_creds()
        service = build("gmail", "v1", credentials=creds)

        response = service.users().history().list(
            userId="me",
            startHistoryId=history_id,
            historyTypes=["messageAdded"],
            labelId="INBOX"
        ).execute()

        emails = []
        for record in response.get("history", []):
            for msg_added in record.get("messagesAdded", []):
                msg_id = msg_added["message"]["id"]
                full   = service.users().messages().get(
                    userId="me", id=msg_id, format="full"
                ).execute()
                parsed = parse_gmail_message(full)
                if parsed:
                    emails.append(parsed)
        return emails

    except Exception as e:
        logger.error(f"Gmail fetch error: {e}")
        return []


def parse_gmail_message(msg: dict) -> dict:
    """Parse raw Gmail message into clean dict."""
    headers = {h["name"].lower(): h["value"]
               for h in msg["payload"]["headers"]}

    def extract_addr(raw):
        if "<" in raw and ">" in raw:
            return raw.split("<")[1].split(">")[0].strip().lower()
        return raw.strip().lower()

    def extract_all(raw):
        return [extract_addr(p) for p in raw.split(",") if p.strip()]

    return {
        "message_id": msg["id"],
        "thread_id":  msg["threadId"],
        "subject":    headers.get("subject", ""),
        "from_email": extract_addr(headers.get("from", "")),
        "to_emails":  extract_all(headers.get("to", "")),
        "cc_emails":  extract_all(headers.get("cc", "")),
        "date":       headers.get("date", ""),
        "body":       _extract_body(msg["payload"])[:3000],
        "snippet":    msg.get("snippet", ""),
    }


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text body."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(
                data + "=="
            ).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        result = _extract_body(part)
        if result:
            return result
    return ""


# ── CREATE GOOGLE CALENDAR EVENT ─────────────────────────────────────────────

def create_calendar_event(model_output: dict, original_email: dict) -> str:
    """Create Google Calendar event from model output."""
    try:
        from googleapiclient.discovery import build
        from dateutil.parser import parse as dp

        creds   = _get_google_creds()
        service = build("calendar", "v3", credentials=creds)

        start_iso = model_output.get("start_time")
        end_iso   = model_output.get("end_time")

        if not start_iso:
            start_dt  = datetime.now(timezone.utc) + timedelta(days=1)
            start_iso = start_dt.isoformat()
            end_iso   = (start_dt + timedelta(hours=1)).isoformat()
        elif not end_iso:
            start_dt = dp(start_iso)
            end_iso  = (start_dt + timedelta(hours=1)).isoformat()

        attendees = (
            [{"email": e} for e in GROUP_EMAILS]
            if GROUP_EMAILS
            else [{"email": a} for a in model_output.get("attendees", [])]
        )

        event = {
            "summary":     model_output.get("title") or "Meeting",
            "location":    model_output.get("location") or "",
            "description": (
                f"Auto-created from email.\n"
                f"Subject: {original_email.get('subject', '')}\n"
                f"From: {original_email.get('from_email', '')}\n"
                f"Time confidence: {model_output.get('time_confidence', 'none')}"
            ),
            "start":       {"dateTime": start_iso, "timeZone": "UTC"},
            "end":         {"dateTime": end_iso,   "timeZone": "UTC"},
            "attendees":   attendees,
            "sendUpdates": "all",
        }

        created = service.events().insert(
            calendarId="primary",
            body=event,
            sendNotifications=True,
        ).execute()

        logger.info(f"Calendar event created: {created.get('id')}")
        return created.get("htmlLink", "")

    except Exception as e:
        logger.error(f"Calendar error: {e}")
        return ""


# ── SEND CONFIRMATION EMAIL ───────────────────────────────────────────────────

def send_confirmation_email(to_email: str, model_output: dict, token: str):
    """Send confirm/cancel email using Gmail API."""
    try:
        from googleapiclient.discovery import build

        confirm_url = f"{CONFIRM_URL}?token={token}&action=confirm"
        cancel_url  = f"{CONFIRM_URL}?token={token}&action=cancel"
        title       = model_output.get("title") or "Meeting"
        start       = model_output.get("start_time") or "TBD"
        end         = model_output.get("end_time") or "TBD"
        location    = model_output.get("location") or "Not specified"
        attendees   = ", ".join(model_output.get("attendees", []))
        confidence  = model_output.get("time_confidence", "none")

        html = f"""
        <html><body style="font-family:Arial,sans-serif;
                           max-width:600px;margin:auto;padding:20px">
        <h2 style="color:#1a73e8">Meeting detected in your email</h2>
        <p>Your assistant found a meeting. Create a Calendar event?</p>
        <table style="width:100%;border-collapse:collapse;margin:16px 0">
          <tr><td style="padding:8px;color:#666;width:120px">Title</td>
              <td style="padding:8px;font-weight:bold">{title}</td></tr>
          <tr style="background:#f8f9fa">
              <td style="padding:8px;color:#666">Start</td>
              <td style="padding:8px">{start}</td></tr>
          <tr><td style="padding:8px;color:#666">End</td>
              <td style="padding:8px">{end}</td></tr>
          <tr style="background:#f8f9fa">
              <td style="padding:8px;color:#666">Location</td>
              <td style="padding:8px">{location}</td></tr>
          <tr><td style="padding:8px;color:#666">Attendees</td>
              <td style="padding:8px">{attendees}</td></tr>
          <tr style="background:#f8f9fa">
              <td style="padding:8px;color:#666">Confidence</td>
              <td style="padding:8px">{confidence}</td></tr>
        </table>
        <div style="margin:24px 0">
          <a href="{confirm_url}"
             style="background:#1a73e8;color:white;padding:12px 28px;
                    text-decoration:none;border-radius:4px;
                    margin-right:12px;font-weight:bold">
            Yes, create event
          </a>
          <a href="{cancel_url}"
             style="background:#ea4335;color:white;padding:12px 28px;
                    text-decoration:none;border-radius:4px;font-weight:bold">
            No, cancel
          </a>
        </div>
        <p style="color:#999;font-size:12px">
          Sent by your email-to-calendar assistant.
        </p>
        </body></html>
        """

        creds   = _get_google_creds()
        service = build("gmail", "v1", credentials=creds)

        msg            = email.mime.multipart.MIMEMultipart("alternative")
        msg["Subject"] = f"[Action needed] Meeting detected: {title}"
        # FIX: Always send FROM and TO the sender email (msadi account)
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = SENDER_EMAIL
        msg.attach(email.mime.text.MIMEText(html, "html"))

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        service.users().messages().send(
            userId="me",
            body={"raw": raw}
        ).execute()

        logger.info(f"Confirmation email sent to {SENDER_EMAIL}")

    except Exception as e:
        logger.error(f"Send email error: {e}")
        raise e


# ── GOOGLE CREDENTIALS ────────────────────────────────────────────────────────

def _get_google_creds():
    """Always gets a fresh token using refresh token — never expires."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_data = json.loads(GOOGLE_TOKEN)
    creds = Credentials(
        token=token_data.get("token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri=token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=token_data.get("client_id"),
        client_secret=token_data.get("client_secret"),
        scopes=token_data.get("scopes", [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/calendar.events",
        ]),
    )
    creds.refresh(Request())
    return creds


# ── HTML RESPONSE ─────────────────────────────────────────────────────────────

def _html(status: int, title: str, body: str) -> dict:
    html = f"""<!DOCTYPE html>
<html><head><title>{title}</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:500px;
       margin:60px auto;text-align:center;color:#333}}
  h2{{color:{"#1a73e8" if status==200 else "#ea4335"}}}
</style></head>
<body>
  <h2>{title}</h2>
  <p>{body}</p>
</body></html>"""
    return {
        "statusCode": status,
        "headers":    {"Content-Type": "text/html"},
        "body":       html
    }