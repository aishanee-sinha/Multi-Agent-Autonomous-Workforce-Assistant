#!/bin/bash
# deploy_adapters.sh — Hot-swap LoRA adapters on EC2 vLLM
# =========================================================
# Downloads new DPO-trained LoRA adapter from S3, stops vLLM,
# replaces adapter files, restarts vLLM, runs smoke test.
#
# Usage:
#   bash rlhf/deploy_adapters.sh --flow slack --adapter-s3 s3://bucket/rlhf/adapters/slack_dpo/
#   bash rlhf/deploy_adapters.sh --flow slack --adapter-local rlhf/adapters/slack_dpo/
#
# Rollback: If smoke test fails, automatically reverts to previous adapter.

set -euo pipefail

FLOW="${1:---help}"
ADAPTER_SOURCE="${2:-}"
VLLM_PORT="${VLLM_PORT:-8000}"
ADAPTERS_DIR="${ADAPTERS_DIR:-/data/lora_adapters}"

usage() {
    echo "Usage: $0 <flow> <adapter-path>"
    echo "  flow:         slack | email | meeting"
    echo "  adapter-path: local path or s3:// URI to new adapter"
    echo ""
    echo "Example:"
    echo "  $0 slack rlhf/adapters/slack_dpo/"
    echo "  $0 email s3://qwen-lora-weights/rlhf/adapters/email_dpo/"
    exit 1
}

if [[ "$FLOW" == "--help" || -z "$ADAPTER_SOURCE" ]]; then
    usage
fi

ADAPTER_DIR="${ADAPTERS_DIR}/${FLOW}"
BACKUP_DIR="${ADAPTERS_DIR}/${FLOW}_backup_$(date +%Y%m%d_%H%M%S)"
VLLM_PID_FILE="/tmp/vllm.pid"

echo "=== LoRA Adapter Deployment ==="
echo "  Flow:          ${FLOW}"
echo "  Source:        ${ADAPTER_SOURCE}"
echo "  Target:        ${ADAPTER_DIR}"
echo "  Backup:        ${BACKUP_DIR}"

# 1. Backup current adapter
echo ""
echo "Step 1: Backing up current adapter..."
if [ -d "$ADAPTER_DIR" ]; then
    cp -r "$ADAPTER_DIR" "$BACKUP_DIR"
    echo "  Backed up to ${BACKUP_DIR}"
else
    echo "  No existing adapter found — fresh install"
fi

# 2. Download/copy new adapter
echo ""
echo "Step 2: Installing new adapter..."
mkdir -p "$ADAPTER_DIR"
if [[ "$ADAPTER_SOURCE" == s3://* ]]; then
    echo "  Downloading from S3..."
    aws s3 sync "$ADAPTER_SOURCE" "$ADAPTER_DIR/" --quiet
else
    echo "  Copying from local path..."
    cp -r "$ADAPTER_SOURCE"/* "$ADAPTER_DIR/"
fi
echo "  Installed to ${ADAPTER_DIR}"

# 3. Stop vLLM
echo ""
echo "Step 3: Stopping vLLM..."
if pgrep -f "vllm" > /dev/null 2>&1; then
    pkill -f "vllm" || true
    sleep 3
    echo "  vLLM stopped"
else
    echo "  vLLM was not running"
fi

# 4. Start vLLM with new adapter
echo ""
echo "Step 4: Starting vLLM with new adapter..."
# The exact command depends on your vLLM setup — adjust as needed
nohup python -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-14B-Instruct-AWQ" \
    --enable-lora \
    --lora-modules \
        "slack=${ADAPTERS_DIR}/slack" \
        "email=${ADAPTERS_DIR}/email" \
        "meeting=${ADAPTERS_DIR}/meeting" \
    --port "${VLLM_PORT}" \
    --quantization awq \
    --max-model-len 4096 \
    > /var/log/vllm.log 2>&1 &

echo "  vLLM PID: $!"
echo "  Waiting for vLLM to start..."
sleep 15  # Give vLLM time to load

# 5. Smoke test
echo ""
echo "Step 5: Running smoke test..."
SMOKE_TEST_PASSED=true

# Test that vLLM is responding
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null || echo "000")

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "  ❌ vLLM health check failed (HTTP $HTTP_CODE)"
    SMOKE_TEST_PASSED=false
else
    echo "  ✅ vLLM is responding (HTTP $HTTP_CODE)"

    # Test inference with a simple prompt
    RESPONSE=$(curl -s "http://localhost:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${FLOW}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"test\"}],
            \"max_tokens\": 10
        }" 2>/dev/null || echo "error")

    if echo "$RESPONSE" | grep -q "choices"; then
        echo "  ✅ Inference test passed"
    else
        echo "  ❌ Inference test failed: $RESPONSE"
        SMOKE_TEST_PASSED=false
    fi
fi

# 6. Rollback if smoke test failed
if [[ "$SMOKE_TEST_PASSED" == "false" ]]; then
    echo ""
    echo "Step 6: ROLLING BACK to previous adapter..."
    if [ -d "$BACKUP_DIR" ]; then
        pkill -f "vllm" || true
        sleep 2
        rm -rf "$ADAPTER_DIR"
        mv "$BACKUP_DIR" "$ADAPTER_DIR"
        echo "  Restored from backup. Please restart vLLM manually."
    else
        echo "  No backup available — manual intervention required"
    fi
    echo ""
    echo "❌ DEPLOYMENT FAILED — rolled back to previous adapter"
    exit 1
fi

echo ""
echo "✅ DEPLOYMENT SUCCESSFUL"
echo "  Flow: ${FLOW}"
echo "  New adapter active at: ${ADAPTER_DIR}"
echo "  Backup saved at: ${BACKUP_DIR}"
