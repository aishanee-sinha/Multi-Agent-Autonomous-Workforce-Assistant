This is a comprehensive technical "Runbook" for your project. It captures the raw commands, the debugging "fixes" we found for the GPU, and the full deployment pipeline from Docker to Lambda.

---

# 🛠️ Project Runbook: Autonomous Slack-Jira Agent

**Target Environment:** AWS EC2 (GPU) + AWS Lambda (Serverless) + Slack API

---

## 1. EC2 Connectivity & Environment Setup

Before starting the model, you must connect to the instance and enter the correct Python environment.

**Connect to EC2:**

```bash
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip

```

**Prepare the Environment:**

```bash
# Enter the virtual environment where vLLM is installed
source llm_env/bin/activate

# Navigate to your model/adapter directory
cd ~/adapters

```

---

## 2. GPU Debugging & "Ghost" Process Management

If the model fails to start with a `ValueError` regarding free memory, the GPU is "clogged" by a previous crashed session.

**Check GPU Memory Status:**

```bash
nvidia-smi

```

**Kill Ghost Processes (Clear VRAM):**
If `nvidia-smi` shows memory is used but no process is active, force-clear the hardware:

```bash
sudo fuser -v /dev/nvidia* -k

```

---

## 3. Starting the vLLM Inference Engine

This command starts the "Brain" of the agent. It loads the base Qwen model and attaches your custom LoRA adapters.

**Start Command:**

```bash
python3 -m vllm.entrypoints.openai.api_server \
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
    --port 8000

```

*Note: Keep this terminal window open. If you close it, the model will stop.*

---

## 4. The Lambda "Docker Dance"

Since the Lambda requires heavy libraries like `slack_sdk` and `langchain`, we deploy it using a Docker container.

**Step-by-Step Deployment:**

1. **Build the Image:**
```bash
docker build -t slack-jira-agent .

```


2. **Authenticate Docker to AWS ECR:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com

```


3. **Tag and Push:**
```bash
docker tag slack-jira-agent:latest [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/slack-jira-repo:latest
docker push [YOUR_ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/slack-jira-repo:latest

```


4. **Update Lambda:**
Go to the AWS Lambda Console -> **Image** -> **Deploy New Image**.

---

## 5. API Gateway & Slack Integration

The Lambda needs a "front door" so Slack can talk to it.

1. **Lambda Function URL:** Enable "Function URL" in the Lambda configuration (Auth type: NONE for testing).
2. **Slack Event Subscriptions:** Paste the Lambda Function URL into the "Request URL" field in your Slack App settings.
3. **Interactivity:** Paste the same URL into the "Interactivity & Shortcuts" section so the "Create Ticket" button works.

---

## 6. Verification & Health Checks

**Test the EC2 API (Internal):**

```bash
# Run this on the EC2 itself in a second terminal
curl http://localhost:8000/v1/models

```

**Test the EC2 API (External):**

```bash
# Run this from your laptop
curl http://[YOUR-EC2-IP]:8000/v1/models

```

**Check Lambda Logs:**
If Slack isn't responding, check **AWS CloudWatch Logs** for your Lambda.

* **404 Error:** Lambda is hitting the wrong EC2 path (ensure it's `/v1/chat/completions`).
* **Timeout Error:** Increase Lambda timeout to **60 seconds**.
* **Connection Error:** Check if Port 8000 is open in the **EC2 Security Group**.

---

## 7. Configuration Checklist (Env Vars)

Ensure these are set in your Lambda Console:

* `JIRA_API_TOKEN` / `JIRA_EMAIL` / `JIRA_BASE_URL`
* `SLACK_BOT_TOKEN`
* `EC2_IP`
* `TEAM_MAP_JSON`: `{"SLACK_ID": "JIRA_ACCOUNT_ID"}`

---

This document is now ready to be shared with Ketki and the rest of the team! Would you like me to add a troubleshooting section specifically for common Slack API errors?