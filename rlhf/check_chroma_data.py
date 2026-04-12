#!/usr/bin/env python3
"""
Utility script to quickly check if RLHF telemetry is landing in ChromaDB.
"""
import os
import chromadb
import json

CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8001"))

def main():
    print(f"Connecting to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT}...")
    try:
        client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        collections = client.list_collections()
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")
        return

    if not collections:
        print("\nNo collections found! Either no feedback has been logged yet, or you are connected to the wrong database.")
        return

    print(f"\nFound {len(collections)} collections:")
    for col in collections:
        print(f" - {col.name} (Count: {col.count()})")

    # Let's peek into the slack_feedback collection if it exists
    slack_col = [c for c in collections if c.name == "slack_feedback"]
    if slack_col:
        print("\n--- Latest 3 Slack Feedback Items ---")
        col = slack_col[0]
        results = col.peek(limit=3)
        for i, (doc, meta, id) in enumerate(zip(results["documents"], results["metadatas"], results["ids"])):
            decision = meta.get("decision", "unknown")
            timestamp = meta.get("timestamp", "unknown")
            print(f"\n[ID: {id}] | Decision: {decision} | Time: {timestamp}")
            print(f" Prompt Snippet: {doc[:100]}...")
            
            # Print parsed metadata if available
            extra_meta = meta.get("extra", {})
            if isinstance(extra_meta, str) and extra_meta.startswith("{"):
                try:
                    parsed = json.loads(extra_meta)
                    if "jira_key" in parsed:
                        print(f" Generated Jira: {parsed['jira_key']} ({parsed.get('jira_url', '')})")
                except:
                    pass

if __name__ == "__main__":
    main()
