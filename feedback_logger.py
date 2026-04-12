"""
feedback_logger.py — Feedback collection for RLHF
===================================================
Two storage targets:
  1. S3 (JSONL)   — raw logs for DPO dataset construction
  2. ChromaDB     — embeddings for BOTH RAG retrieval AND semantic pairing

Stores ALL feedback (approved + cancelled) in ChromaDB with a
"decision" metadata tag. Consumers filter at query time:
  - RAG retriever:  where={"decision": "approved"}   (only good examples)
  - DPO builder:    where={"decision": "cancelled"}  (find rejected outputs)
                    then find similar approved for pairing

Called by all three subgraphs when user clicks Approve/Cancel.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
S3_BUCKET       = os.environ.get("S3_BUCKET", "qwen-lora-weights")
RLHF_S3_PREFIX  = os.environ.get("RLHF_S3_PREFIX", "rlhf/feedback")
CHROMADB_HOST   = os.environ.get("CHROMADB_HOST", os.environ.get("EC2_IP", "localhost"))
CHROMADB_PORT   = int(os.environ.get("CHROMADB_PORT", "8001"))

# Lazy-loaded clients to avoid cold start overhead
_s3_client = None
_chroma_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3")
    return _s3_client


def _get_chromadb():
    """Lazy-load ChromaDB HTTP client — only imported when first used."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            _chroma_client = chromadb.HttpClient(
                host=CHROMADB_HOST,
                port=CHROMADB_PORT,
            )
            logger.info("feedback_logger: connected to ChromaDB at %s:%d", CHROMADB_HOST, CHROMADB_PORT)
        except Exception as e:
            logger.warning("feedback_logger: ChromaDB connection failed: %s", e)
            return None
    return _chroma_client


def _get_collection(flow: str):
    """Get or create the ChromaDB collection for a given flow."""
    client = _get_chromadb()
    if client is None:
        return None
    try:
        return client.get_or_create_collection(
            name=f"{flow}_feedback",
            metadata={"description": f"RLHF feedback for {flow} flow"},
        )
    except Exception as e:
        logger.warning("feedback_logger: failed to get collection '%s_feedback': %s", flow, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────
def log_feedback(
    flow: str,
    prompt: str,
    response: str,
    human_decision: str,
    metadata: dict | None = None,
) -> None:
    """
    Log a single LLM interaction + human feedback to S3 and ChromaDB.

    Parameters
    ----------
    flow : str
        One of "slack", "email", "meeting".
    prompt : str
        Serialized LLM input (system + human messages as JSON string).
    response : str
        Serialized LLM output (structured output as JSON string).
    human_decision : str
        "approved" or "cancelled".
    metadata : dict, optional
        Additional context (channel_id, jira_key, email_subject, etc.).
    """
    feedback_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "id": feedback_id,
        "timestamp": timestamp,
        "flow": flow,
        "prompt": prompt,
        "response": response,
        "human_decision": human_decision,
        "metadata": metadata or {},
        "adapter_version": "v1",
    }

    # ── 1. Write to S3 (JSONL — append) ─────────────────────────────────
    _write_s3_jsonl(flow, record)

    # ── 2. Write to ChromaDB (BOTH approved and cancelled) ──────────────
    _write_chromadb(flow, feedback_id, prompt, response, human_decision, timestamp, metadata)

    logger.info(
        "feedback_logger: logged %s feedback for flow=%s id=%s",
        human_decision, flow, feedback_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# S3 JSONL writer
# ─────────────────────────────────────────────────────────────────────────────
def _write_s3_jsonl(flow: str, record: dict) -> None:
    """Append a feedback record to the daily JSONL file in S3."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3_key = f"{RLHF_S3_PREFIX}/{flow}/{today}.jsonl"
    line = json.dumps(record, default=str) + "\n"

    s3 = _get_s3()
    try:
        # Try to append to existing file
        existing = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        existing_body = existing["Body"].read().decode("utf-8")
        updated_body = existing_body + line
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            updated_body = line
        else:
            logger.error("feedback_logger: S3 read error: %s", e)
            return

    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=updated_body.encode("utf-8"),
            ContentType="application/jsonl",
        )
        logger.info("feedback_logger: wrote to s3://%s/%s", S3_BUCKET, s3_key)
    except Exception as e:
        logger.error("feedback_logger: S3 write error: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB writer
# ─────────────────────────────────────────────────────────────────────────────
def _write_chromadb(
    flow: str,
    feedback_id: str,
    prompt: str,
    response: str,
    human_decision: str,
    timestamp: str,
    metadata: dict | None,
) -> None:
    """
    Store feedback embedding in ChromaDB.

    Both approved AND cancelled interactions are stored, tagged with
    the decision in metadata. Consumers filter at query time:
      - RAG: where={"decision": "approved"}
      - DPO: where={"decision": "cancelled"} then similarity search
    """
    collection = _get_collection(flow)
    if collection is None:
        return

    # The document text is what ChromaDB embeds and searches on.
    # We use the prompt (user input) as the primary document for
    # semantic similarity search.
    # The response is stored in metadata so it can be retrieved
    # alongside the prompt.
    #
    # Truncate response to fit ChromaDB metadata limits (max ~32KB)
    response_truncated = response[:8000] if response else ""

    try:
        collection.add(
            documents=[prompt],
            metadatas=[{
                "flow": flow,
                "decision": human_decision,
                "response": response_truncated,
                "timestamp": timestamp,
                **({"extra": json.dumps(metadata)} if metadata else {}),
            }],
            ids=[feedback_id],
        )
        logger.info(
            "feedback_logger: ChromaDB add flow=%s decision=%s id=%s",
            flow, human_decision, feedback_id,
        )
    except Exception as e:
        logger.warning("feedback_logger: ChromaDB write failed: %s", e)
