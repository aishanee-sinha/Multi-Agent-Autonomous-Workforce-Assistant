import os, csv
from datetime import datetime, date, time, timezone
from typing import Iterator, Dict, Any, Optional, List, Tuple, Set

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
TIME_MIN = "1900-01-01T00:00:00Z"
TIME_MAX = "2100-01-01T00:00:00Z"
OUTPUT_CSV = "events_table.csv"

def get_creds() -> Credentials:
    if not os.path.exists("credentials.json"):
        raise FileNotFoundError("Place your OAuth client file as credentials.json in this folder.")
    creds: Optional[Credentials] = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return creds

def iter_calendar_list(service) -> Iterator[Dict[str, Any]]:
    page_token = None
    while True:
        resp = service.calendarList().list(
            maxResults=250, pageToken=page_token, minAccessRole="reader"
        ).execute()
        for cal in resp.get("items", []):
            yield cal
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

def iter_events(service, calendar_id: str) -> Iterator[Dict[str, Any]]:
    page_token = None
    while True:
        resp = (
            service.events()
            .list(
                calendarId=calendar_id,
                timeMin=TIME_MIN,
                timeMax=TIME_MAX,
                singleEvents=True,
                orderBy="startTime",
                maxResults=2500,
                pageToken=page_token,
                showDeleted=False,
            )
            .execute()
        )
        for ev in resp.get("items", []):
            yield ev
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

def parse_rfc3339(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def normalize_times(ev: Dict[str, Any]) -> Tuple[str, str]:
    """Return (start_iso, end_iso) for timed or all-day events."""
    s, e = ev.get("start", {}), ev.get("end", {})
    if "dateTime" in s or "dateTime" in e:
        sdt = parse_rfc3339(s.get("dateTime"))
        edt = parse_rfc3339(e.get("dateTime"))
        return (sdt.isoformat() if sdt else "", edt.isoformat() if edt else "")
    # all-day (end.date is exclusive -> we keep 00:00 UTC)
    sd, ed = s.get("date"), e.get("date")
    if sd and ed:
        sdt = datetime.combine(date.fromisoformat(sd), time(0, 0)).replace(tzinfo=timezone.utc)
        edt = datetime.combine(date.fromisoformat(ed), time(0, 0)).replace(tzinfo=timezone.utc)
        return (sdt.isoformat(), edt.isoformat())
    return ("", "")

def flatten_attendees(att: Optional[List[Dict[str, Any]]]) -> str:
    if not att:
        return ""
    seen: Set[str] = set()
    out: List[str] = []
    for a in att:
        email = a.get("email")
        if email and email not in seen:
            seen.add(email); out.append(email)
    return "; ".join(out)

def print_table(rows: List[Dict[str, str]], headers: List[str]) -> None:
    # simple fixed-width table without extra deps
    widths = [max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers]
    def fmt_row(vals): return " | ".join(str(v).ljust(w) for v, w in zip(vals, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(fmt_row(headers)); print(sep)
    for r in rows:
        print(fmt_row([r.get(h, "") for h in headers]))

def main():
    try:
        creds = get_creds()
        service = build("calendar", "v3", credentials=creds)

        # Collect data rows
        headers = ["start_datetime", "end_datetime", "summary", "attendees"]
        rows: List[Dict[str, str]] = []

        for cal in iter_calendar_list(service):
            cal_id = cal.get("id")
            cal_sum = cal.get("summary", cal_id)
            print(f"Fetching: {cal_sum} ({cal_id}) …")
            for ev in iter_events(service, cal_id):
                start_iso, end_iso = normalize_times(ev)
                rows.append({
                    "start_datetime": start_iso,
                    "end_datetime": end_iso,
                    "summary": ev.get("summary", "") or "",
                    "attendees": flatten_attendees(ev.get("attendees")),
                })

        # Write CSV (table format file)
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(rows)

        # Print pretty table in terminal
        if rows:
            print("\n=== Events Table ===")
            print_table(rows, headers)
        else:
            print("\nNo events found in the selected time window/calendars.")

        print(f"\nDone. Wrote {len(rows)} rows to {OUTPUT_CSV}")

    except HttpError as e:
        print("API error:", e)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
