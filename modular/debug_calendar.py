# debug_calendar.py
import json, os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_TOKEN = os.environ.get("GOOGLE_TOKEN_JSON", "")

print("=" * 50)
print("STEP 1: Check GOOGLE_TOKEN_JSON in .env")
print("=" * 50)
print(f"Length     : {len(GOOGLE_TOKEN)} chars")
print(f"First 50   : {GOOGLE_TOKEN[:50]}")
print(f"Last 20    : {GOOGLE_TOKEN[-20:]}")

if not GOOGLE_TOKEN:
    print("❌ GOOGLE_TOKEN_JSON is empty — check your .env")
    exit(1)

print("\n" + "=" * 50)
print("STEP 2: Parse JSON")
print("=" * 50)
try:
    token_data = json.loads(GOOGLE_TOKEN)
    print("✅ JSON parsed successfully")
    print(f"  token          : {str(token_data.get('token', ''))[:30]}...")
    print(f"  refresh_token  : {str(token_data.get('refresh_token', ''))[:30]}...")
    print(f"  client_id      : {token_data.get('client_id', 'MISSING')}")
    print(f"  client_secret  : {token_data.get('client_secret', 'MISSING')[:10]}...")
    print(f"  token_uri      : {token_data.get('token_uri', 'MISSING')}")
    print(f"  scopes         : {token_data.get('scopes', 'MISSING')}")
except json.JSONDecodeError as e:
    print(f"❌ JSON parse failed: {e}")
    print("   Common causes:")
    print("   - Line breaks inside the JSON value in .env")
    print("   - Unescaped quotes")
    print("   - Copied with extra spaces or newlines")
    exit(1)

print("\n" + "=" * 50)
print("STEP 3: Build Google credentials object")
print("=" * 50)
try:
    from google.oauth2.credentials import Credentials
    creds = Credentials(
        token=token_data["token"],
        refresh_token=token_data["refresh_token"],
        token_uri=token_data["token_uri"],
        client_id=token_data["client_id"],
        client_secret=token_data["client_secret"],
        scopes=token_data.get("scopes", []),
    )
    print(f"✅ Credentials object created")
    print(f"   expired  : {creds.expired}")
    print(f"   valid    : {creds.valid}")
except Exception as e:
    print(f"❌ Failed to build credentials: {e}")
    exit(1)

print("\n" + "=" * 50)
print("STEP 4: Refresh token if expired")
print("=" * 50)
try:
    from google.auth.transport.requests import Request
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        print("✅ Token refreshed successfully")
    else:
        print("✅ Token is still valid — no refresh needed")
except Exception as e:
    print(f"❌ Token refresh failed: {e}")
    exit(1)

print("\n" + "=" * 50)
print("STEP 5: Test Gmail API")
print("=" * 50)
try:
    from googleapiclient.discovery import build
    gmail = build("gmail", "v1", credentials=creds)
    profile = gmail.users().getProfile(userId="me").execute()
    print(f"✅ Gmail connected — logged in as: {profile['emailAddress']}")
except Exception as e:
    print(f"❌ Gmail API failed: {e}")
    exit(1)

print("\n" + "=" * 50)
print("STEP 6: Test Calendar API")
print("=" * 50)
try:
    from googleapiclient.discovery import build
    calendar = build("calendar", "v3", credentials=creds)
    cal_list = calendar.calendarList().list().execute()
    calendars = cal_list.get("items", [])
    print(f"✅ Calendar connected — found {len(calendars)} calendars:")
    for c in calendars[:3]:
        print(f"   - {c['summary']} ({c['id']})")
except Exception as e:
    print(f"❌ Calendar API failed: {e}")
    exit(1)

print("\n" + "=" * 50)
print("✅ ALL CHECKS PASSED — Google credentials are working")
print("=" * 50)