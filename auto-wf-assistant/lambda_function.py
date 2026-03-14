import os
import json
import logging
import urllib.parse
import requests
from requests.auth import HTTPBasicAuth
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

# Setup Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- CONFIGURATION (Loaded from Environment Variables) ---
# See the "How to Update" section below for details.
JIRA_BASE_URL = os.environ.get("JIRA_BASE_URL")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY", "KAN")
JIRA_ISSUE_TYPE = os.environ.get("JIRA_ISSUE_TYPE", "Task")

# Team Map is now a JSON string in Lambda Env Vars
# Example: {"U0ALDPM2DK4": "712020:acc-id-xxx"}
TEAM_MAP = json.loads(os.environ.get("TEAM_MAP_JSON", "{}"))

# --- DATA SCHEMA ---
class JiraTicket(BaseModel):
    task_summary: str = Field(description="Brief summary of the task")
    assignee: str = Field(default="Unassigned", description="The Slack User ID or Name mentioned")
    no_action: bool = Field(default=False, description="True if no task is found")

def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    headers = {k.lower(): v for k, v in event.get('headers', {}).items()}
    
    # Ignore Slack retries to prevent duplicate tickets/messages
    if int(headers.get('x-slack-retry-num', 0)) > 0:
        return {"statusCode": 200, "body": "Ignoring retry"}

    body_raw = event.get('body', '')
    if event.get('isBase64Encoded'):
        import base64
        body_raw = base64.b64decode(body_raw).decode('utf-8')

    # Handle Interactivity (Buttons)
    if body_raw.startswith('payload='):
        payload_str = urllib.parse.unquote_plus(body_raw.split('payload=')[1])
        payload = json.loads(payload_str)
        return handle_interactivity(payload)
    
    body = json.loads(body_raw)

    # Handle Slack URL Verification
    if body.get("type") == "url_verification":
        return {"statusCode": 200, "body": json.dumps({"challenge": body.get("challenge")})}

    return handle_event(body)

def handle_event(body):
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    ec2_ip = os.environ.get("EC2_IP")
    slack_client = WebClient(token=slack_token)
    
    slack_event = body.get("event", {})
    if slack_event.get("bot_id") or slack_event.get("subtype") == "bot_message":
        return {"statusCode": 200, "body": "ok"}

    channel_id = slack_event.get("channel")
    user_text = slack_event.get("text", "")
    ts = slack_event.get("ts")
    
    # Strip brackets to help the LLM parse IDs cleanly
    clean_text = user_text.replace("<@", "").replace(">", "")

    llm = ChatOpenAI(
        model="slack", 
        openai_api_base=f"http://{ec2_ip}:8000/v1",
        openai_api_key="none",
        timeout=45, # High timeout for autonomous stability
        max_retries=0
    ).with_structured_output(JiraTicket)

    try:
        sys_msg = (
            "You are a Jira assistant. Extract task details. "
            "For 'assignee', return the ID or the name of the person mentioned."
        )
        ticket = llm.invoke([SystemMessage(content=sys_msg), HumanMessage(content=clean_text)])

        if ticket.no_action:
            return {"statusCode": 200, "body": "no action"}

        blocks = [
            {
                "type": "section", 
                "text": {"type": "mrkdwn", "text": f"🎫 *Proposed Jira Task*\n*Summary:* {ticket.task_summary}\n*Assignee:* {ticket.assignee}"}
            },
            {
                "type": "actions", 
                "elements": [
                    {
                        "type": "button", 
                        "text": {"type": "plain_text", "text": "✅ Create Ticket"}, 
                        "style": "primary", 
                        "value": json.dumps({"s": ticket.task_summary, "a": ticket.assignee}), 
                        "action_id": "create_jira"
                    },
                    {"type": "button", "text": {"type": "plain_text", "text": "❌ Cancel"}, "style": "danger", "action_id": "cancel_jira"}
                ]
            }
        ]
        slack_client.chat_postMessage(channel=channel_id, blocks=blocks, thread_ts=ts)

    except Exception as e:
        logger.error(f"LLM Error: {e}")
    
    return {"statusCode": 200, "body": "success"}

def handle_interactivity(payload):
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_client = WebClient(token=slack_token)
    
    action = payload['actions'][0]
    channel_id = payload['channel']['id']
    message_ts = payload['container']['message_ts']

    if action['action_id'] == "create_jira":
        try:
            data = json.loads(urllib.parse.unquote_plus(action['value']))
            raw_assignee = data.get('a', '')
            
            # Match the assignee to the Jira Account ID map
            jira_acc_id = TEAM_MAP.get(raw_assignee)
            if not jira_acc_id:
                for name, acc_id in TEAM_MAP.items():
                    if name.lower() in raw_assignee.lower():
                        jira_acc_id = acc_id
                        break

            jira_res = create_jira_issue(data['s'], "Created via Slack Agent", jira_acc_id)
            
            if "key" in jira_res:
                msg = f"✅ *Ticket Created:* <{JIRA_BASE_URL}/browse/{jira_res['key']}|{jira_res['key']}>"
            else:
                msg = "⚠️ Failed to create ticket. Check Jira permissions."
            
            slack_client.chat_update(channel=channel_id, ts=message_ts, text=msg, blocks=None)
        except Exception as e:
            slack_client.chat_update(channel=channel_id, ts=message_ts, text=f"❌ Error: {str(e)}", blocks=None)
    else:
        slack_client.chat_update(channel=channel_id, ts=message_ts, text="❌ Ticket creation cancelled.", blocks=None)

    return {"statusCode": 200, "body": ""}

def create_jira_issue(summary, description, assignee_id=None):
    url = f"{JIRA_BASE_URL}/rest/api/3/issue"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    
    fields = {
        "project": {"key": JIRA_PROJECT_KEY},
        "summary": summary,
        "description": {
            "type": "doc", "version": 1, 
            "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]
        },
        "issuetype": {"name": JIRA_ISSUE_TYPE}
    }

    if assignee_id:
        fields["assignee"] = {"id": assignee_id}

    payload = json.dumps({"fields": fields})
    response = requests.post(url, data=payload, headers=headers, auth=auth)
    return response.json()