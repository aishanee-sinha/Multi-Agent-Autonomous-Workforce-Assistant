# Email to Calendar — Autonomous Meeting Scheduler

An autonomous system that monitors Gmail inboxes, detects meeting emails using a fine-tuned Qwen2.5 LLM, and automatically creates Google Calendar events.

---

## How It Works

```
Email arrives in Gmail
  → Gmail Watch detects it instantly
  → Notifies Google Pub/Sub topic
  → Pub/Sub pushes to Cloud Run relay (GCP)
  → Cloud Run forwards to AWS Lambda
  → Lambda fetches email via Gmail API
  → Sends email to fine-tuned Qwen2.5 on EC2
  → Model extracts: title, time, attendees, location
  → Lambda sends confirmation email to sender
  → Sender clicks ✅ Yes
  → Google Calendar event created with invites sent
```

---

## Architecture

```
┌─────────────────── GCP ───────────────────┐  ┌────────────── AWS ──────────────────┐
│                                            │  │                                      │
│  Gmail Inbox → Gmail Watch → Pub/Sub ──────┼──→ Cloud Run Relay → AWS Lambda        │
│                                            │  │                        ↓             │
└────────────────────────────────────────────┘  │               EC2 (Qwen2.5 vLLM)    │
                                                │                        ↓             │
                                                │            Google Calendar API       │
                                                └──────────────────────────────────────┘
```

---

## Repository Structure

```
email_scheduler_agent/
├── lambda_function.py          # AWS Lambda — orchestrates everything
├── gmail-relay/
│   ├── main.py                 # Cloud Run relay — bridges GCP to AWS
│   ├── requirements.txt
│   └── Dockerfile
├── get_token.py                # OAuth token setup for Gmail/Calendar
├── register_watch.py           # Register Gmail push watch
├── refresh_token.py            # Refresh expired token
└── README.md
```

---

## Prerequisites

| Service | Purpose |
|---|---|
| AWS Account | Lambda, EC2 |
| Google Cloud Account | Pub/Sub, Cloud Run |
| Gmail Account(s) | Email monitoring |
| Google Calendar | Event creation |
| EC2 g5.xlarge (GPU) | vLLM inference server |

---

## Step 1 — EC2 Setup (Model Inference Server)

### 1.1 Start EC2 Instance

```
AWS Console → EC2 → Instances
→ Start your GPU instance (g5.xlarge recommended)
→ Note the Public IPv4 address
```

### 1.2 SSH Into EC2

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

### 1.3 Start vLLM Server

```bash
source llm_env/bin/activate

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --quantization awq \
    --dtype auto \
    --enable-lora \
    --max-loras 3 \
    --max-lora-rank 32 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 4096 \
    --lora-modules \
        slack=~/adapters/slack2jira \
        email=~/adapters/email-calender \
        meeting=~/adapters/transcript_summarizer \
    --host 0.0.0.0 \
    --port 8000 > ~/vllm.log 2>&1 &

tail -f ~/vllm.log
```

Wait for:
```
Application startup complete.
```

### 1.4 Verify Model is Running

```bash
# From your Mac
curl http://YOUR_EC2_IP:8000/v1/models
```

You should see `email`, `slack`, `meeting` adapters listed.

### 1.5 GPU Debugging (if model fails to start)

```bash
# Check GPU memory
nvidia-smi

# Kill ghost processes if GPU is stuck
sudo fuser -v /dev/nvidia* -k
```

---

## Step 2 — Google Cloud Setup

### 2.1 Create Pub/Sub Topic

```
Google Cloud Console → Pub/Sub → Topics
→ Create Topic
→ Topic ID: gmail-push
→ Uncheck "Add default subscription"
→ Create
```

### 2.2 Add Gmail Publisher Permission

```
Topics → gmail-push → Permissions
→ Add Principal: gmail-api-push@system.gserviceaccount.com
→ Role: Pub/Sub Publisher
→ Save
```

### 2.3 Deploy Cloud Run Relay

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Create relay folder
mkdir ~/gmail-relay && cd ~/gmail-relay

# Create the 3 files (main.py, requirements.txt, Dockerfile)
# — see gmail-relay/ folder in this repo

# Deploy
gcloud run deploy gmail-relay \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars LAMBDA_URL=https://YOUR_LAMBDA_URL.lambda-url.us-east-1.on.aws/
```

Note the Cloud Run URL: `https://gmail-relay-XXXXXX.us-central1.run.app`

### 2.4 Create Pub/Sub Subscription

```
Pub/Sub → Subscriptions → Create subscription
→ Subscription ID: gmail-push-sub
→ Topic: gmail-push
→ Delivery type: Push
→ Endpoint URL: https://gmail-relay-XXXXXX.us-central1.run.app
→ Acknowledgement deadline: 600 seconds
→ Create
```

---

## Step 3 — Gmail OAuth Setup

### 3.1 Create OAuth Credentials

```
Google Cloud Console → APIs & Services → Credentials
→ Create Credentials → OAuth 2.0 Client ID
→ Application type: Web application
→ Authorized redirect URIs: http://localhost:8080
→ Create → Download JSON → rename to credentials.json
```

### 3.2 Enable APIs

```
APIs & Services → Library
→ Enable: Gmail API
→ Enable: Google Calendar API
```

### 3.3 Get OAuth Token

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
python get_token.py
```

Browser opens → log in → token.json saved.

### 3.4 Register Gmail Watch

```bash
python register_watch.py
```

Output:
```
Watch registered! historyId: XXXXX
```

> **Note:** Gmail watch expires every 7 days. Re-run `register_watch.py` weekly or set up a cron job.

---

## Step 4 — AWS Lambda Setup

### 4.1 Create Lambda Function

```
AWS Console → Lambda → Create function
→ Author from scratch
→ Name: email-calendar-handler
→ Runtime: Python 3.11
→ Use existing role: qwen-agent-handler-role-jt96o2sw
→ Create function
```

### 4.2 Configure Timeout and Memory

```
Lambda → Configuration → General configuration
→ Timeout: 1 min 0 sec
→ Memory: 512 MB
→ Save
```

### 4.3 Enable Function URL

```
Lambda → Configuration → Function URL
→ Create function URL
→ Auth type: NONE
→ Save
→ Copy the URL
```

### 4.4 Deploy Lambda Code

```
Lambda → Code → Upload from → .zip file
```

Build the zip:
```bash
mkdir lambda-deploy && cd lambda-deploy
pip install \
  google-api-python-client \
  google-auth google-auth-httplib2 \
  google-auth-oauthlib \
  python-dateutil \
  --platform manylinux2014_x86_64 \
  --target . \
  --implementation cp \
  --python-version 3.11 \
  --only-binary=:all:

cp ../lambda_function.py .
zip -r ../lambda-deploy.zip .
```

Upload `lambda-deploy.zip` to Lambda.

### 4.5 Set Environment Variables

```
Lambda → Configuration → Environment variables → Edit
```

| Key | Value |
|---|---|
| `EC2_IP` | Your EC2 public IP |
| `CONFIRM_URL` | Your Lambda Function URL |
| `GROUP_EMAILS_JSON` | `["user1@gmail.com", "user2@gmail.com"]` |
| `SENDER_EMAIL` | Email to send confirmations from |

---

## Step 5 — End to End Test

### 5.1 Test Lambda Directly

```
Lambda → Test → Create test event:
```

```json
{
  "subject": "Team sync tomorrow at 3pm",
  "from_email": "sender@gmail.com",
  "to_emails": ["receiver@gmail.com"],
  "cc_emails": [],
  "date": "Wed, 20 Mar 2026 10:00:00 +0000",
  "body": "Hi, lets meet tomorrow at 3pm on Zoom: zoom.us/j/123456"
}
```

Expected: `{"statusCode": 200, "body": "confirmation sent"}`

### 5.2 Test Full Autonomous Flow

Send a meeting email from one group member to another:
```
From: member1@gmail.com
To:   member2@gmail.com
Subject: Q2 Planning - March 25 at 2pm to 3:30pm PST
Body: Hi team, let's meet on March 25 2026 from 2pm to 3:30pm PST on Zoom.
```

Within 30-60 seconds:
1. Confirmation email arrives in `SENDER_EMAIL` inbox
2. Click **Yes, create event**
3. Google Calendar event created with invites sent to all group members

---

## Step 6 — Troubleshooting

### EC2 model not responding

```bash
# Check if vLLM is running
curl http://YOUR_EC2_IP:8000/health

# If not running, SSH in and restart
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
source llm_env/bin/activate
# Run the start command from Step 1.3
```

### Lambda timeout error

```
Lambda → Configuration → General configuration
→ Increase timeout to 1 min
```

### No confirmation email

1. Check token is not expired — re-run `refresh_token.py`
2. Verify `gmail.send` scope is in token scopes
3. Check `SENDER_EMAIL` env var is set correctly

### Gmail watch expired

```bash
python register_watch.py
```

### EC2 IP changed after restart

```
AWS Console → EC2 → copy new Public IPv4
Lambda → Configuration → Environment variables
→ Update EC2_IP
```

---

## Maintenance

### Weekly Tasks

```bash
# Refresh Gmail watch (expires every 7 days)
python register_watch.py

# Refresh OAuth token
python refresh_token.py
```

### When EC2 Restarts

1. SSH in and start vLLM (Step 1.3)
2. Update EC2_IP in Lambda if IP changed

### Token Management

The system uses the OAuth refresh token which never expires. If you see auth errors, run:

```bash
python refresh_token.py
```

---

## Scaling to 5 Accounts

To add more Gmail accounts:

1. Run `get_token.py` for each account (browser opens once per account)
2. Add all emails to `GROUP_EMAILS_JSON` in Lambda env vars:
   ```json
   ["user1@gmail.com","user2@gmail.com","user3@gmail.com","user4@gmail.com","user5@gmail.com"]
   ```
3. Register Gmail watch for each account — update `register_watch.py` to loop through all accounts

---

## File Reference

| File | Location | Purpose |
|---|---|---|
| `lambda_function.py` | Root | AWS Lambda — main orchestration logic |
| `gmail-relay/main.py` | Root | Cloud Run — Pub/Sub → Lambda relay |
| `get_token.py` | Root | One-time OAuth setup (run locally) |
| `register_watch.py` | Root | Register Gmail push watch (run locally) |
| `refresh_token.py` | Root | Refresh expired token (run locally) |
| `adapter_model.safetensors` | S3: qwen-lora-weights/email-calender/ | Fine-tuned model weights |
| `adapter_config.json` | S3: qwen-lora-weights/email-calender/ | LoRA adapter config |

---

## Tech Stack

| Component | Technology |
|---|---|
| Email model | Qwen2.5-14B-Instruct-AWQ + LoRA fine-tune |
| Inference server | vLLM 0.17.0 |
| Serverless | AWS Lambda (Python 3.11) |
| Relay | Google Cloud Run |
| Message queue | Google Pub/Sub |
| Email API | Gmail API v1 |
| Calendar API | Google Calendar API v3 |
| GPU instance | AWS EC2 g5.xlarge |
| Model storage | AWS S3 |
