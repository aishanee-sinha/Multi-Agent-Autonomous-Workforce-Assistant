#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# update_lambda_image.sh
# Rebuilds and pushes the meeting-agent Lambda container image, then
# updates the function to use the new image.
#
# Run this from the PROJECT ROOT (where your Dockerfile lives).
# Usage: bash update_lambda_image.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Your values from the AWS console screenshots ──────────────────────────────
ACCOUNT_ID="166329267466"
REGION="us-east-1"
ECR_REPO="worker-lambda"
IMAGE_TAG="meeting-agent"
LAMBDA_NAME="meeting-agent-handler"

# Full image URI
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Meeting Agent — Lambda Image Update                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Account  : $ACCOUNT_ID"
echo "  Region   : $REGION"
echo "  ECR repo : $ECR_REPO"
echo "  Tag      : $IMAGE_TAG"
echo "  Lambda   : $LAMBDA_NAME"
echo "  Full URI : $IMAGE_URI"
echo ""

# ── Step 1: Confirm changed files are in place ────────────────────────────────
echo "▶ Step 1: Checking updated source files..."
for f in src/meeting_agent.py src/state.py src/orchestrator.py; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f NOT FOUND — copy the updated files here before proceeding"
        exit 1
    fi
done

# ── Step 2: Authenticate Docker to ECR ───────────────────────────────────────
echo ""
echo "▶ Step 2: Authenticating Docker to ECR..."
aws ecr get-login-password --region "$REGION" \
  | docker login \
      --username AWS \
      --password-stdin \
      "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo "  ✓ Docker authenticated to ECR"

# ── Step 3: Build the Docker image ───────────────────────────────────────────
echo ""
echo "▶ Step 3: Building Docker image (platform linux/amd64, BuildKit OFF for Lambda compatibility)..."
DOCKER_BUILDKIT=0 docker build \
  --platform linux/amd64 \
  --tag "${ECR_REPO}:${IMAGE_TAG}" \
  .
echo "  ✓ Image built: ${ECR_REPO}:${IMAGE_TAG}"

# ── Step 4: Tag for ECR ───────────────────────────────────────────────────────
echo ""
echo "▶ Step 4: Tagging image for ECR..."
docker tag "${ECR_REPO}:${IMAGE_TAG}" "$IMAGE_URI"
echo "  ✓ Tagged: $IMAGE_URI"

# ── Step 5: Push to ECR ───────────────────────────────────────────────────────
echo ""
echo "▶ Step 5: Pushing image to ECR (this takes 1-3 minutes)..."
docker push "$IMAGE_URI"
echo "  ✓ Image pushed to ECR"

# ── Step 6: Update Lambda to use new image ────────────────────────────────────
echo ""
echo "▶ Step 6: Updating Lambda function to use new image..."
aws lambda update-function-code \
  --function-name "$LAMBDA_NAME" \
  --image-uri "$IMAGE_URI" \
  --region "$REGION" \
  --query '{FunctionName:FunctionName, LastUpdateStatus:LastUpdateStatus, CodeSize:CodeSize}' \
  --output table
echo "  ✓ Lambda update triggered"

# ── Step 7: Wait for update to complete ──────────────────────────────────────
echo ""
echo "▶ Step 7: Waiting for Lambda update to finish (polling every 5s)..."
for i in $(seq 1 24); do
  STATUS=$(aws lambda get-function-configuration \
    --function-name "$LAMBDA_NAME" \
    --region "$REGION" \
    --query 'LastUpdateStatus' \
    --output text)
  echo "  [${i}/24] Status: $STATUS"
  if [ "$STATUS" = "Successful" ]; then
    echo "  ✓ Lambda is live with the new image"
    break
  elif [ "$STATUS" = "Failed" ]; then
    echo "  ✗ Lambda update failed — check CloudWatch logs"
    exit 1
  fi
  sleep 5
done

# ── Step 8: Also bump timeout and SESSION_TTL if not done yet ─────────────────
echo ""
echo "▶ Step 8: Ensuring timeout=60s and SESSION_TTL=7200..."
aws lambda update-function-configuration \
  --function-name "$LAMBDA_NAME" \
  --timeout 60 \
  --region "$REGION" \
  --query '{Timeout:Timeout}' \
  --output text

# Give it a moment before updating env vars (can't update concurrently)
sleep 5

aws lambda update-function-configuration \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION" \
  --environment "Variables={SESSION_TTL_SECONDS=7200}" \
  --query '{LastUpdateStatus:LastUpdateStatus}' \
  --output text
echo "  ✓ Timeout=60, SESSION_TTL=7200 set"

# ── Step 9: Confirm final configuration ───────────────────────────────────────
echo ""
echo "▶ Step 9: Final configuration check..."
aws lambda get-function-configuration \
  --function-name "$LAMBDA_NAME" \
  --region "$REGION" \
  --query '{
    Function:FunctionName,
    Timeout:Timeout,
    Memory:MemorySize,
    Status:LastUpdateStatus,
    ImageUri:Code.ImageUri
  }' \
  --output table

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✅  Update complete — ready to test!                  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Next: Upload demo_meeting_full_test.txt to Google Drive"
echo "  and follow the runbook from Section 3."
echo ""
