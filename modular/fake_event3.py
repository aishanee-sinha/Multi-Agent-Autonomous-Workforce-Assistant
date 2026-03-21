import json, time, uuid
from dotenv import load_dotenv
load_dotenv()

from state import OrchestratorState
from orchestrator import _graph, handler

print("=" * 50)
print("STEP 1: Incoming email with meeting intent...")
print("=" * 50)

step1_event = {
    "headers": {},
    "isBase64Encoded": False,
    "body": json.dumps({
        "from_email": "boss@company.com",
        "subject": "Team Sync Friday 3pm",
        "to_emails": ["team@company.com"],
        "cc_emails": [],
        "date": "Fri, 21 Mar 2026 10:00:00 +0000",
        "body": """Hi team,

Let's have a sync meeting this Friday March 27th 2026 at 3:00 PM UTC.
We'll discuss the Q2 roadmap and sprint planning.

Location: Google Meet - meet.google.com/abc-defg-hij
Duration: 1 hour

Please confirm your availability.

Thanks,
Boss""",
        "snippet": "Let's have a sync meeting this Friday March 27th at 3:00 PM UTC."
    })
}

# Use a unique thread_id each run so MemorySaver never carries over stale state
thread1 = f"email-step1-{uuid.uuid4()}"
initial = OrchestratorState(raw_event=step1_event)
config  = {"configurable": {"thread_id": thread1}}
raw     = _graph.invoke(initial, config=config)
step1_state = OrchestratorState(**raw) if isinstance(raw, dict) else raw

print(f"Intent          : {step1_state.intent}")
print(f"Email source    : {step1_state.email_source}")
print(f"Is meeting      : {step1_state.is_meeting}")
print(f"Meeting title   : {step1_state.meeting_title}")
print(f"Meeting start   : {step1_state.meeting_start}")
print(f"Meeting end     : {step1_state.meeting_end}")
print(f"Meeting location: {step1_state.meeting_location}")
print(f"Pending meeting : {'✅ set' if step1_state.pending_meeting else '❌ not set'}")
print(f"Error           : {step1_state.error}")

if not step1_state.is_meeting or not step1_state.pending_meeting:
    print("\n❌ Meeting not detected or pending_meeting not set")
    print("   Common causes:")
    print("   - GROUP_EMAILS_JSON not set to [] in .env (must allow all senders)")
    print("   - LLM returned is_meeting=False for this email text")
    exit(1)

print("\n✅ Step 1 done — confirmation email sent to boss@company.com")

print("\nWaiting 3 seconds (simulating user clicking Yes)...")
time.sleep(3)

print("\n" + "=" * 50)
print("STEP 2: User clicks Yes — create calendar event...")
print("=" * 50)

step2_event = {
    "headers": {},
    "isBase64Encoded": False,
    "body": "",
    "queryStringParameters": {
        "action": "confirm"     # change to "cancel" to test cancellation
    }
}

# Fresh thread + carry pending_meeting from Step 1 — no DynamoDB needed
thread2 = f"email-step2-{uuid.uuid4()}"
initial2 = OrchestratorState(
    raw_event=step2_event,
    pending_meeting=step1_state.pending_meeting,
    meeting_title=step1_state.meeting_title,
)
raw2        = _graph.invoke(initial2, config={"configurable": {"thread_id": thread2}})
step2_state = OrchestratorState(**raw2) if isinstance(raw2, dict) else raw2

print(f"Calendar link : {step2_state.calendar_link}")
print(f"Error         : {step2_state.error}")

if step2_state.calendar_link:
    print(f"\n✅ Calendar event created!")
    print(f"   Open here: {step2_state.calendar_link}")
elif step2_state.error:
    print(f"\n❌ Failed: {step2_state.error}")
else:
    print("\n❌ No calendar link returned — check Google credentials in .env")