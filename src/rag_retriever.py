"""
rag_retriever.py — RAG retrieval for RLHF
===========================================
Queries ChromaDB for similar past APPROVED interactions.
Returns formatted few-shot examples to inject into LLM prompts.

At inference time:
  1. Embed the new input text
  2. Search ChromaDB collection for similar past prompts
     where decision = "approved" (only good examples)
  3. Return the prompt+response pairs as few-shot examples
  4. Caller injects these into the system prompt

Handles failures gracefully — if ChromaDB is down or no examples
found, returns empty list and the system works without RAG.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", os.environ.get("EC2_IP", "localhost"))
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8001"))

# Minimum similarity score (0-1) to consider a result relevant.
# ChromaDB returns distances (lower = more similar for default L2).
# We filter by max distance instead.
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "1.5"))

# Lazy-loaded client
_chroma_client = None


def _get_chromadb():
    """Lazy-load ChromaDB HTTP client."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            _chroma_client = chromadb.HttpClient(
                host=CHROMADB_HOST,
                port=CHROMADB_PORT,
            )
        except Exception as e:
            logger.warning("rag_retriever: ChromaDB connection failed: %s", e)
            return None
    return _chroma_client


def _get_collection(flow: str):
    """Get the ChromaDB collection for a flow. Returns None if unavailable."""
    client = _get_chromadb()
    if client is None:
        return None
    try:
        return client.get_or_create_collection(name=f"{flow}_feedback")
    except Exception as e:
        logger.warning("rag_retriever: collection error: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────
def get_similar_approved(
    flow: str,
    query_text: str,
    n_results: int = 2,
) -> list[dict]:
    """
    Search ChromaDB for the most similar past APPROVED interactions.

    Parameters
    ----------
    flow : str
        One of "slack", "email", "meeting".
    query_text : str
        The new input to find similar examples for.
    n_results : int
        Maximum number of examples to retrieve.

    Returns
    -------
    list[dict]
        Each dict has:
          - "input":  the past prompt text
          - "output": the approved LLM response (as string)
          - "distance": similarity distance (lower = more similar)
        Returns empty list if ChromaDB unavailable or no matches.
    """
    if not query_text or not query_text.strip():
        return []

    collection = _get_collection(flow)
    if collection is None:
        return []

    try:
        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            logger.info("rag_retriever: collection '%s_feedback' is empty — no RAG examples", flow)
            return []

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
            where={"decision": "approved"},
        )

        examples = []
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results.get("distances") else 0
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

                # Skip results that are too dissimilar
                if distance > MAX_DISTANCE:
                    continue

                examples.append({
                    "input": doc,
                    "output": metadata.get("response", ""),
                    "distance": distance,
                })

        logger.info(
            "rag_retriever: flow=%s query_len=%d found=%d examples (of %d total in collection)",
            flow, len(query_text), len(examples), count,
        )
        return examples

    except Exception as e:
        logger.warning("rag_retriever: query failed: %s", e)
        return []


def format_as_few_shot(examples: list[dict], max_examples: int = 3) -> str:
    """
    Format retrieved examples into a few-shot block for LLM prompts.

    Parameters
    ----------
    examples : list[dict]
        Output of get_similar_approved().
    max_examples : int
        Cap the number of examples to avoid prompt bloat.

    Returns
    -------
    str
        Formatted string to append to the system prompt.
        Returns empty string if no examples.
    """
    if not examples:
        return ""

    lines = [
        "\n--- Previously Approved Examples ---",
        "The following are similar past interactions that were approved by a human.",
        "Use them as reference for the expected output format and quality.\n",
    ]

    for i, ex in enumerate(examples[:max_examples]):
        # Try to pretty-print the response if it's JSON
        output = ex.get("output", "")
        try:
            parsed = json.loads(output)
            output = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass

        lines.append(f"Example {i + 1}:")
        lines.append(f"  Input:  {ex.get('input', '')[:300]}")
        lines.append(f"  Approved Output: {output[:500]}")
        lines.append("")

    lines.append("--- End of Examples ---\n")
    return "\n".join(lines)
