import os
import json
import base64
import urllib.request
from flask import Flask, request, jsonify

app = Flask(__name__)

LAMBDA_URL = os.environ.get("LAMBDA_URL", "YOUR_LAMBDA_FUNCTION_URL")

@app.route("/", methods=["POST"])
def relay():
    try:
        envelope = request.get_json(silent=True)
        if not envelope:
            return jsonify({"error": "no json"}), 200

        pubsub_message = envelope.get("message", {})
        if not pubsub_message:
            return jsonify({"error": "no message"}), 200

        print(f"Received Pub/Sub message, forwarding to Lambda...")

        lambda_payload = json.dumps({
            "message": {
                "data": pubsub_message["data"],
                "messageId": pubsub_message.get("messageId", "")
            },
            "subscription": envelope.get("subscription", "")
        }).encode("utf-8")

        req = urllib.request.Request(
            LAMBDA_URL,
            data=lambda_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = resp.read().decode()
            print(f"Lambda response: {result}")

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        print(f"Relay error: {e}")
        return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
