# Meeting Summarizer — AWS Orchestration

Production deployment of the Qwen2.5-14B meeting summarizer. Triggered by Google Drive file uploads, orchestrated by AWS Lambda, inference served by vLLM on EC2, outputs delivered to Slack and email.

---

## Architecture

```
Google Drive (.txt upload)
        ↓  Google Apps Script polls every 1 min
AWS API Gateway  (HTTP POST)
        ↓
AWS Lambda  (Python orchestrator)
    ├── downloads transcript from Drive
    ├── calls vLLM /v1/chat/completions (EC2 :8000)
    ├── parses ABSTRACT / DECISIONS / ACTIONS_JSON
    ├── stores artifacts → S3
    ├── posts summary → Slack (Block Kit)
    └── sends email → AWS SES (HTML + ICS + CSV)
```

---

## Components

| Component | Service | Details |
|---|---|---|
| Trigger | Google Apps Script | Polls `transcript_summarizer/` folder every 1 min |
| Entry point | AWS API Gateway | REST endpoint, routes POST to Lambda |
| Orchestrator | AWS Lambda (Python) | Handles full pipeline, 15 min timeout |
| Inference | EC2 + vLLM | Qwen2.5-14B-Instruct-AWQ + LoRA adapter |
| Storage | AWS S3 | One folder per meeting |
| Messaging | Slack Block Kit | Structured summary with Confirm/Cancel buttons |
| Email | AWS SES | HTML summary + ICS invite + actions CSV |

---

## Prerequisites

- AWS CLI installed and configured (`aws configure`)
- EC2 instance with GPU (g4dn.xlarge or better) already created
- S3 bucket created
- Slack app with `chat:write` and `files:write` permissions
- AWS SES verified sender email
- Google service account credentials JSON for Drive access

---

## EC2 Instance Management

### Start the instance
```bash
aws ec2 start-instances \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --region us-east-1
```

### Wait until running
```bash
aws ec2 wait instance-running \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --region us-east-1

# Check status
aws ec2 describe-instances \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --query "Reservations[0].Instances[0].State.Name" \
  --output text
```

### Get public IP after start
```bash
aws ec2 describe-instances \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --query "Reservations[0].Instances[0].PublicIpAddress" \
  --output text
```

### SSH into instance
```bash
ssh -i /path/to/your-key.pem ubuntu@<PUBLIC_IP>
```

### Stop the instance (when not in use — saves cost)
```bash
aws ec2 stop-instances \
  --instance-ids i-XXXXXXXXXXXXXXXXX \
  --region us-east-1

# Wait until stopped
aws ec2 wait instance-stopped \
  --instance-ids i-XXXXXXXXXXXXXXXXX
```

### Reboot if unresponsive
```bash
aws ec2 reboot-instances \
  --instance-ids i-XXXXXXXXXXXXXXXXX
```

---

## vLLM Inference Server

### Install vLLM (first time only, run on EC2)
```bash
pip install vllm==0.4.2
pip install peft transformers accelerate
```

### Download and merge LoRA adapter with base model (first time only)
```python
# run_merge.py — run once to merge LoRA into base model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE  = "Qwen/Qwen2.5-14B-Instruct-AWQ"
LORA  = "/home/ubuntu/meeting-sum-adapter"   # upload adapter here
SAVE  = "/home/ubuntu/meeting-sum-merged"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.float16, device_map="auto"
)
print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA)
print("Merging and unloading...")
model = model.merge_and_unload()
model.save_pretrained(SAVE)
AutoTokenizer.from_pretrained(BASE).save_pretrained(SAVE)
print(f"Saved merged model to {SAVE}")
```

```bash
python run_merge.py
```

### Start vLLM server
```bash
# Standard start — OpenAI-compatible endpoint on port 8000
python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/meeting-sum-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --dtype float16 \
  --served-model-name meeting-summarizer
```

### Start vLLM server in background (persistent across SSH sessions)
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/meeting-sum-merged \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --dtype float16 \
  --served-model-name meeting-summarizer \
  > /home/ubuntu/vllm.log 2>&1 &

echo "vLLM PID: $!"
```

### Check vLLM is running
```bash
# Check process
ps aux | grep vllm

# Check logs
tail -f /home/ubuntu/vllm.log

# Test endpoint
curl http://localhost:8000/health

# Test inference
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meeting-summarizer",
    "messages": [{"role": "user", "content": "Hello, summarize this: Team discussed Q3 goals."}],
    "max_tokens": 100
  }'
```

### Stop vLLM server
```bash
# Find and kill the process
pkill -f "vllm.entrypoints"

# Or by PID
kill <PID>
```

### Check GPU utilization
```bash
nvidia-smi
nvidia-smi dmon -s u   # live utilization monitor
```

---

## Lambda Function

### Environment variables (set in Lambda console or via CLI)

```bash
aws lambda update-function-configuration \
  --function-name meeting-summarizer-trigger \
  --environment Variables="{
    VLLM_ENDPOINT=http://<EC2_PRIVATE_IP>:8000,
    S3_BUCKET=your-bucket-name,
    SLACK_BOT_TOKEN=xoxb-your-token,
    SLACK_CHANNEL_ID=C0XXXXXXXXX,
    SES_SENDER=your-verified@email.com,
    WEBHOOK_SECRET=meeting-summarizer-secret-2026,
    GOOGLE_CREDS_SECRET=meeting-summarizer-google-creds,
    PROCESSED_FOLDER_ID=your-drive-processed-folder-id
  }"
```

### Deploy Lambda function
```bash
# Zip and deploy
zip -r function.zip lambda_function.py requirements.txt

aws lambda update-function-code \
  --function-name meeting-summarizer-trigger \
  --zip-file fileb://function.zip \
  --region us-east-1
```

### Increase Lambda timeout (required — inference takes time)
```bash
aws lambda update-function-configuration \
  --function-name meeting-summarizer-trigger \
  --timeout 900 \
  --memory-size 512
```

### Invoke Lambda manually for testing
```bash
aws lambda invoke \
  --function-name meeting-summarizer-trigger \
  --payload '{
    "body": "{\"type\":\"new_transcript\",\"file_id\":\"YOUR_FILE_ID\",\"file_name\":\"test.txt\",\"secret\":\"meeting-summarizer-secret-2026\"}"
  }' \
  --cli-binary-format raw-in-base64-out \
  response.json

cat response.json
```

### Check Lambda logs
```bash
aws logs tail /aws/lambda/meeting-summarizer-trigger \
  --follow \
  --region us-east-1
```

---

## API Gateway

### Get endpoint URL
```bash
aws apigateway get-rest-apis \
  --query "items[?name=='meeting-summarizer-api'].id" \
  --output text
```

### Test endpoint manually
```bash
curl -X POST \
  https://ftpzdcgehe.execute-api.us-east-1.amazonaws.com/prod/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "type": "new_transcript",
    "file_id": "YOUR_GOOGLE_DRIVE_FILE_ID",
    "file_name": "test_meeting.txt",
    "secret": "meeting-summarizer-secret-2026",
    "timestamp": "2026-03-23T00:00:00.000Z"
  }'
```

---

## S3 Storage

### List meeting artifacts
```bash
aws s3 ls s3://your-bucket-name/meetings/ --recursive
```

### Download artifacts for a specific meeting
```bash
aws s3 cp s3://your-bucket-name/meetings/<meeting_id>/ ./local_output/ --recursive
```

### Check bucket size
```bash
aws s3 ls s3://your-bucket-name --recursive --human-readable --summarize
```

### Per-meeting folder structure in S3
```
meetings/
└── <meeting_id>/
    ├── transcript.txt       original transcript
    ├── raw_output.txt       raw model output
    ├── meta.json            parsed structured summary
    ├── invite.ics           calendar invite
    └── actions.csv          action items with owner/deadline
```

---

## Google Apps Script

The Apps Script monitors `transcript_summarizer/` in Google Drive and calls the API Gateway when a new `.txt` file is detected.

### Setup (one time)
1. Go to [script.google.com](https://script.google.com)
2. Create project named `MeetingSummarizerTrigger`
3. Paste `google_apps_script.js` into `Code.gs`
4. Update `CONFIG.DRIVE_FOLDER_ID` and `CONFIG.API_GATEWAY_URL`
5. Run `setupTrigger()` once manually
6. Authorize when prompted

### Useful functions to run manually

```javascript
setupTrigger()           // install 1-min polling trigger
checkForNewTranscripts() // run immediately (for testing)
debugFolder()            // list all files + MIME types in folder
debugDetailed()          // show why each file is being skipped
resetProcessedFiles()    // clear processed list (reprocess all files)
showStatus()             // show trigger status and tracked file count
testWithFile()           // trigger Lambda for a specific file ID
```

---

## End-to-End Test

Run these steps in order to verify the full pipeline:

```bash
# 1. Start EC2
aws ec2 start-instances --instance-ids i-XXXXXXXXXXXXXXXXX --region us-east-1
aws ec2 wait instance-running --instance-ids i-XXXXXXXXXXXXXXXXX

# 2. SSH and start vLLM
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/meeting-sum-merged \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 --dtype float16 \
  --served-model-name meeting-summarizer > vllm.log 2>&1 &

# 3. Wait for vLLM to load (~2-3 min), then verify
curl http://localhost:8000/health

# 4. Upload a .txt transcript to Google Drive transcript_summarizer/ folder

# 5. Wait 1 minute for Apps Script to trigger, or run manually:
#    In Apps Script editor → Run → checkForNewTranscripts

# 6. Watch Lambda logs
aws logs tail /aws/lambda/meeting-summarizer-trigger --follow

# 7. Verify S3 output
aws s3 ls s3://your-bucket-name/meetings/ --recursive

# 8. Check Slack channel for Block Kit summary

# 9. Check email inbox for HTML summary + ICS + CSV attachment
```

---

## Cost Management

The EC2 GPU instance is the main cost driver. Always stop it when not in use.

```bash
# Stop after testing
aws ec2 stop-instances --instance-ids i-XXXXXXXXXXXXXXXXX --region us-east-1

# Set up auto-stop via CloudWatch (stops after 30 min idle)
aws cloudwatch put-metric-alarm \
  --alarm-name "MeetingSummarizer-IdleStop" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 1800 \
  --threshold 5 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:swf:us-east-1:<ACCOUNT_ID>:action/actions/AWS_EC2.InstanceId.Stop/1.0 \
  --dimensions Name=InstanceId,Value=i-XXXXXXXXXXXXXXXXX
```

### Estimated costs

| Service | Usage | Est. monthly |
|---|---|---|
| EC2 g4dn.xlarge | ~2 hrs/day | ~$42 |
| S3 storage | ~1 GB meetings | ~$0.02 |
| Lambda | 1000 invocations | ~$0.00 |
| API Gateway | 1000 requests | ~$0.01 |
| AWS SES | 1000 emails | ~$0.10 |
| **Total** | | **~$42/month** |

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Connection refused` on vLLM endpoint | Server not started or still loading | Wait 2-3 min, check `vllm.log` |
| Lambda timeout | EC2 not running or vLLM down | Start EC2, start vLLM server |
| `New files found: 0` in Apps Script | File already processed or too old | Run `resetProcessedFiles()`, increase age limit to 72 hrs |
| File detected but not sent to Lambda | `text/plain` MIME type mismatch | Run `debugFolder()` to check actual MIME type |
| Slack message not appearing | Wrong channel ID or missing bot permission | Verify `SLACK_CHANNEL_ID` and `chat:write` scope |
| Email not delivered | SES sender not verified | Verify email in SES console |
| CUDA OOM on vLLM | Model too large for GPU | Lower `--gpu-memory-utilization` to 0.85 |
| Lambda cold start slow | First invocation after idle | Increase Lambda memory or use provisioned concurrency |
