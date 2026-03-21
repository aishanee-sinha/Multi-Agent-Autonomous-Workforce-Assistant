"""
refresh_token.py
Refreshes the OAuth access token using the refresh token.
Run this if you see authentication errors.
"""

import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

def refresh_token():
    token_data = json.loads(open("token.json").read())
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data["scopes"],
    )
    creds.refresh(Request())
    with open("token.json", "w") as f:
        f.write(creds.to_json())

    data = json.loads(open("token.json").read())
    print(f"Token refreshed!")
    print(f"New expiry: {data['expiry']}")
    print(f"Scopes: {data['scopes']}")
    print()
    print("Paste this into Lambda GOOGLE_TOKEN_JSON env var:")
    print(open("token.json").read())

if __name__ == "__main__":
    refresh_token()
