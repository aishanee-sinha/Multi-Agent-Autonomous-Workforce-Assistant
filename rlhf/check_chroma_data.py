#!/usr/bin/env python3
"""
Utility script to quickly check if RLHF telemetry is landing in ChromaDB.
"""
import os
import chromadb
import json
from dotenv import load_dotenv

load_dotenv()

CHROMADB_HOST = os.environ.get("CHROMADB_HOST", os.environ.get("EC2_IP", "localhost"))
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
        for doc, meta, id in zip(results["documents"], results["metadatas"], results["ids"]):
            decision = meta.get("decision", "unknown")
            timestamp = meta.get("timestamp", "unknown")
            response_full = meta.get("response", "No response saved")
            extra_meta = meta.get("extra", "No extra meta")
            
            response_full = meta.get("response", "No response saved")
            extra_meta = meta.get("extra", "No extra meta")
            
            print(f"\n[ID: {id}] | Decision: {decision} | Time: {timestamp}")
            print(f"Prompt: {doc}")
            print(f"Response: {response_full}")
            print(f"Metadata: {extra_meta}")
            print("-" * 50)

def summarize_calendar_feedback():
    from collections import Counter
    from datetime import datetime

    print(f"\nConnecting to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT} for RLHF summary...")
    try:
        client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        col = client.get_collection("calendar_feedback")
    except Exception as e:
        print(f"Failed: {e}")
        return

    results = col.get()
    if not results["documents"]:
        print("No items found in calendar_feedback.")
        return

    hour_counter = Counter()
    total = 0

    for doc in results["documents"]:
        try:
            data = json.loads(doc)
        except json.JSONDecodeError:
            continue

        meeting_start = data.get("meeting_start")
        if not meeting_start:
            continue

        try:
            dt = datetime.fromisoformat(meeting_start)
        except ValueError:
            continue

        total += 1
        hour_counter[dt.hour] += 1

    if not total:
        print("No valid meeting_start times found.")
        return

    print(f"\n{'='*45}")
    print(f" RLHF Time Preference Summary  ({total} meetings)")
    print(f"{'='*45}")
    print(f"{'Hour':<10} {'Time':<12} {'Count':<8} {'Rate'}")
    print(f"{'-'*45}")
    for hour in sorted(hour_counter):
        count = hour_counter[hour]
        rate = count / total * 100
        label = datetime(2000, 1, 1, hour).strftime("%-I %p") if hasattr(datetime, "%-I") else f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}"
        print(f"  {hour:<8} {label:<12} {count:<8} {rate:.1f}%")


if __name__ == "__main__":
    summarize_calendar_feedback()
