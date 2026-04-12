from dotenv import load_dotenv; load_dotenv()
from calendar_agent import _get_google_creds
from googleapiclient.discovery import build
creds = _get_google_creds()
svc = build('gmail', 'v1', credentials=creds)
profile = svc.users().getProfile(userId='me').execute()
print('historyId:', profile['historyId'])
