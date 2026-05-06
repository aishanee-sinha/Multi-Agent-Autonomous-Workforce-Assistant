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

    # Let's read ALL data from ALL collections
    for col in collections:
        if col.count() == 0:
            continue
            
        print(f"\n--- ALL ITEMS IN {col.name.upper()} ---")
        results = col.get()
        for i, (doc, meta, id) in enumerate(zip(results["documents"], results["metadatas"], results["ids"])):
            decision = meta.get("decision", "unknown")
            timestamp = meta.get("timestamp", "unknown")
            response_full = meta.get("response", "No response saved")
            extra_meta = meta.get("extra", "No extra meta")
            
            print(f"\n[ID: {id}] | Decision: {decision} | Time: {timestamp}")
            print(f"Prompt: {doc}")
            print(f"Response: {response_full}")
            print(f"Metadata: {extra_meta}")
            print("-" * 50)

if __name__ == "__main__":
    main()
