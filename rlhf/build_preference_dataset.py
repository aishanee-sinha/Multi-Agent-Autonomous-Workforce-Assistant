"""
build_preference_dataset.py — Build DPO pairs from production feedback
=======================================================================
Reads feedback from ChromaDB (both approved and cancelled entries),
uses semantic similarity to pair cancelled outputs with the most
similar approved outputs as preference pairs for DPO training.

All data lives in ChromaDB (both approved and cancelled, tagged by decision).

Pairing strategy:
  1. Query ChromaDB: get ALL cancelled entries
     → collection.get(where={"decision": "cancelled"})

  2. For each CANCELLED output:
     → Query ChromaDB: "find the most similar APPROVED prompt"
       collection.query(query_texts=[cancelled_prompt],
                        where={"decision": "approved"}, n_results=1)
     → chosen  = the approved output
     → rejected = the cancelled output

  3. If no similar approved prompt exists (similarity score too low):
     → Skip this pair (or optionally generate synthetic)

Output: JSONL file with {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
  python rlhf/build_preference_dataset.py --flow slack --output datasets/slack_pairs.jsonl
  python rlhf/build_preference_dataset.py --flow all --output-dir datasets/
"""

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)

# Max similarity distance to accept a pair (lower = more similar)
MAX_PAIR_DISTANCE = float(os.environ.get("MAX_PAIR_DISTANCE", "1.5"))


def build_pairs_from_chromadb(
    flow: str,
    chromadb_host: str = "localhost",
    chromadb_port: int = 8001,
    output_path: str = None,
) -> list[dict]:
    """
    Build DPO preference pairs by pairing cancelled outputs with
    the most similar approved outputs from ChromaDB.
    """
    import chromadb

    client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
    collection_name = f"{flow}_feedback"

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        logger.error("Collection '%s' not found: %s", collection_name, e)
        return []

    # Step 1: Get all cancelled entries
    logger.info("Fetching cancelled entries from '%s'...", collection_name)
    cancelled = collection.get(
        where={"decision": "cancelled"},
        include=["documents", "metadatas"],
    )

    n_cancelled = len(cancelled["ids"]) if cancelled["ids"] else 0
    logger.info("Found %d cancelled entries", n_cancelled)

    if n_cancelled == 0:
        logger.info("No cancelled entries found — nothing to pair")
        return []

    # Check how many approved entries exist
    approved_count = collection.count() - n_cancelled
    logger.info("Approved entries available: ~%d", approved_count)

    if approved_count == 0:
        logger.warning("No approved entries to pair with — cannot build preference pairs")
        return []

    # Step 2: For each cancelled entry, find the most similar approved
    pairs = []
    for i in range(n_cancelled):
        cancelled_prompt = cancelled["documents"][i]
        cancelled_meta = cancelled["metadatas"][i]
        cancelled_response = cancelled_meta.get("response", "")

        # Query for most similar APPROVED
        try:
            similar = collection.query(
                query_texts=[cancelled_prompt],
                where={"decision": "approved"},
                n_results=1,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning("ChromaDB query failed for cancelled entry %d: %s", i, e)
            continue

        if not similar["documents"] or not similar["documents"][0]:
            logger.debug("No approved match for cancelled entry %d", i)
            continue

        distance = similar["distances"][0][0] if similar.get("distances") else float("inf")

        # Filter by similarity threshold
        if distance > MAX_PAIR_DISTANCE:
            logger.debug(
                "Skipping pair %d: distance %.3f > threshold %.3f",
                i, distance, MAX_PAIR_DISTANCE,
            )
            continue

        approved_response = similar["metadatas"][0][0].get("response", "")

        pair = {
            "prompt": cancelled_prompt,
            "chosen": approved_response,
            "rejected": cancelled_response,
            "metadata": {
                "distance": round(distance, 4),
                "flow": flow,
                "cancelled_id": cancelled["ids"][i],
                "approved_id": similar["ids"][0][0],
            },
        }
        pairs.append(pair)
        logger.info(
            "Pair %d: distance=%.3f cancelled_id=%s matched_approved_id=%s",
            len(pairs), distance, cancelled["ids"][i], similar["ids"][0][0],
        )

    logger.info("Built %d preference pairs for flow='%s'", len(pairs), flow)

    # Step 3: Write output
    if output_path and pairs:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info("Wrote %d pairs to %s", len(pairs), output_path)

    return pairs


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build DPO preference pairs from ChromaDB feedback")
    parser.add_argument("--flow", choices=["slack", "email", "meeting", "all"], default="all")
    parser.add_argument("--chromadb-host", default=os.environ.get("CHROMADB_HOST", "localhost"))
    parser.add_argument("--chromadb-port", type=int, default=int(os.environ.get("CHROMADB_PORT", "8001")))
    parser.add_argument("--output-dir", default="rlhf/datasets")
    parser.add_argument("--max-distance", type=float, default=MAX_PAIR_DISTANCE)
    args = parser.parse_args()

    global MAX_PAIR_DISTANCE
    MAX_PAIR_DISTANCE = args.max_distance

    flows = ["slack", "email", "meeting"] if args.flow == "all" else [args.flow]
    total = 0
    for flow in flows:
        output_path = os.path.join(args.output_dir, f"{flow}_pairs.jsonl")
        pairs = build_pairs_from_chromadb(
            flow=flow,
            chromadb_host=args.chromadb_host,
            chromadb_port=args.chromadb_port,
            output_path=output_path,
        )
        total += len(pairs)

    print(f"\nTotal: {total} preference pairs generated across {len(flows)} flow(s)")


if __name__ == "__main__":
    main()
