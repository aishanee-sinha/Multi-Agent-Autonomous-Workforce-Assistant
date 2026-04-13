# 🛠️ Project Runbook: Autonomous AI Workforce Assistant

This runbook contains the exact commands to maintain and deploy the production environment.

---

## 1. EC2 Connectivity (The Brain)

Before any automation works, your GPU "Brain" must be online and reachable.

**Connect to EC2:**
```bash
ssh -i "your-key.pem" ubuntu@your-ec2-public-ip
```

**Step 1: Start ChromaDB (Vector DB)**
ChromaDB must be running on Port 8001 to record RLHF telemetry.
```bash
# Run from the project root on EC2
bash rlhf/setup_chromadb.sh --bg
```

**Step 2: Start vLLM (Inference)**
```bash
# Enter your vllm environment
source llm_env/bin/activate

# Execute the engine (Port 8000)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --quantization awq \
    --dtype auto \
    --enable-lora \
    --max-loras 4 \
    --gpu-memory-utilization 0.8 \
    --lora-modules \
        slack=~/adapters/slack2jira \
        email=~/adapters/email-calender \
        meeting=~/adapters/transcript_summarizer \
    --host 0.0.0.0 \
    --port 8000
```

---

## 2. Infrastructure Checklist

### EC2 Security Group
Ensure the following **Inbound Rules** are active:
- **TCP 8000**: Allow from Anywhere (or Lambda IP) for AI Inferences.
- **TCP 8001**: Allow from Anywhere (or Lambda IP) for RLHF Telemetry.
- **TCP 22**: SSH Access.

---

## 3. Lambda Deployment (The Orchestrator)

We use Docker to bundle the agents and the machine learning runtime (`onnxruntime`).

**1. Build locally:**
```bash
# Ensure you are in the project root
docker build -t slack-jira-agent .
```

**2. Push to ECR:**
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com

# Tag and Push
docker tag slack-jira-agent:latest [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/slack-jira-repo:latest
docker push [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/slack-jira-repo:latest
```

**3. Update Lambda:**
In the AWS Console, navigate to your Lambda -> **Image** -> **Deploy New Image**.

---

## 4. Verification & Health Checks

**Verify Telemetry is landing:**
Run the check utility on your EC2 instance:
```bash
python3 rlhf/check_chroma_data.py
```

**Test vLLM Connectivity:**
```bash
curl http://localhost:8000/v1/models
```

---

## 5. Common Troubleshooting

| Symptom | Fix |
|---|---|
| `[Errno 30] Read-only file system` | Ensure `ENV HOME="/tmp"` is in your Dockerfile. |
| `onnxruntime not installed` | Ensure `onnxruntime==1.15.1` is in `requirements.txt`. |
| `ChromaDB connection failed` | Check if you ran `setup_chromadb.sh` and opened Port 8001. |
| `Redis session not found` | Check if your `REDIS_URL` in Lambda is expired or incorrect. |