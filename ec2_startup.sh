#!/bin/bash
# =============================================================================
# ec2_startup.sh
# EC2 Startup Script - Meeting Summarizer Inference Server
# Instance: g4dn.2xlarge (1x T4 16GB) or g5.2xlarge (1x A10G 24GB)
# OS: Deep Learning AMI (Ubuntu 22.04) - already has CUDA + Python
# =============================================================================

set -e
LOG_FILE="/var/log/meeting_summarizer_startup.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================="
echo "Meeting Summarizer EC2 Startup"
echo "$(date)"
echo "=============================="

# --- CONFIGURATION ---
# These are set by AWS Instance User Data or hardcoded here
S3_BUCKET="qwen-lora-weights"
S3_ADAPTER_PREFIX="transcript_summarizer"
BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
ADAPTER_LOCAL_PATH="/opt/model/adapter"
HF_CACHE_DIR="/opt/model/hf_cache"
VLLM_PORT=8000
VLLM_LOG="/var/log/vllm_server.log"

# --- STEP 1: Install dependencies ---
echo ""
echo "Step 1: Installing dependencies..."

pip install --upgrade pip --quiet
pip install vllm==0.4.3 --quiet
pip install boto3 --quiet
echo "  Dependencies installed."

# --- STEP 2: Download LoRA adapter from S3 ---
echo ""
echo "Step 2: Downloading LoRA adapter from S3..."
mkdir -p "$ADAPTER_LOCAL_PATH"

# Download all adapter files
ADAPTER_FILES=(
    "adapter_config.json"
    "adapter_model.safetensors"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "added_tokens.json"
    "special_tokens_map.json"
)

for filename in "${ADAPTER_FILES[@]}"; do
    S3_KEY="${S3_ADAPTER_PREFIX}/${filename}"
    LOCAL_PATH="${ADAPTER_LOCAL_PATH}/${filename}"
    if [ -f "$LOCAL_PATH" ]; then
        echo "  Skipping (already exists): $filename"
    else
        echo "  Downloading: $filename"
        aws s3 cp "s3://${S3_BUCKET}/${S3_KEY}" "$LOCAL_PATH"
    fi
done

echo "  All adapter files downloaded to: $ADAPTER_LOCAL_PATH"

# --- STEP 3: Download base model from HuggingFace ---
echo ""
echo "Step 3: Downloading base model from HuggingFace..."
echo "  Model: $BASE_MODEL"
echo "  Cache: $HF_CACHE_DIR"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"

# Pre-download model weights using huggingface_hub
python3 - <<EOF
import os
os.environ["HF_HOME"] = "$HF_CACHE_DIR"
from huggingface_hub import snapshot_download
print("  Starting model download (this takes 10-20 mins on first boot)...")
snapshot_download(
    repo_id="$BASE_MODEL",
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
)
print("  Base model downloaded.")
EOF

# --- STEP 4: Start vLLM server ---
echo ""
echo "Step 4: Starting vLLM server on port $VLLM_PORT..."
echo "  Base model    : $BASE_MODEL"
echo "  LoRA adapter  : $ADAPTER_LOCAL_PATH"
echo "  GPU memory    : 0.90 utilization"

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --enable-lora \
    --lora-modules "transcript-summarizer=${ADAPTER_LOCAL_PATH}" \
    --port "$VLLM_PORT" \
    --host "0.0.0.0" \
    --dtype bfloat16 \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --max-lora-rank 64 \
    --trust-remote-code \
    --served-model-name "transcript-summarizer" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "  vLLM server started with PID: $VLLM_PID"
echo "$VLLM_PID" > /var/run/vllm.pid

# --- STEP 5: Wait for server to be ready ---
echo ""
echo "Step 5: Waiting for vLLM server to become ready..."
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "  vLLM server is ready after ${ELAPSED}s"
        break
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... (${ELAPSED}s elapsed)"
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "  ERROR: vLLM server did not start within ${MAX_WAIT}s"
    echo "  Check logs at: $VLLM_LOG"
    exit 1
fi

# --- STEP 6: Test endpoint ---
echo ""
echo "Step 6: Testing inference endpoint..."

TEST_RESPONSE=$(curl -s -X POST \
    "http://localhost:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "transcript-summarizer",
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 5
    }')

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    echo "  Endpoint test passed."
else
    echo "  WARNING: Endpoint test returned unexpected response."
    echo "  Response: $TEST_RESPONSE"
fi

# --- DONE ---
echo ""
echo "=============================="
echo "Startup complete: $(date)"
echo "vLLM endpoint: http://$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4):${VLLM_PORT}/v1"
echo "Logs: $VLLM_LOG"
echo "=============================="
