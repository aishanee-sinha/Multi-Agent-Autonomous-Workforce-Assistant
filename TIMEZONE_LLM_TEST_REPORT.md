# Timezone LLM Call Test Report
**Date:** 2026-04-09
**Model:** Qwen/Qwen2.5-14B-Instruct-AWQ (base, no LoRA)
**System timezone:** PDT
**UTC offset:** -0700

---

## 1. System Timezone at Runtime

```
======================================================================
SYSTEM TIMEZONE SNAPSHOT
  datetime.now().astimezone() : 2026-04-09 12:31:10.059575-07:00
  UTC offset                  : -0700
  TZ name                     : PDT
  isoformat()                 : 2026-04-09T12:31:10.059575-07:00
======================================================================
```

---

## 2. Timezone Constants in Code

```
======================================================================
TIMEZONE CONSTANTS CHECK
  PDT_TZ found     : PDT
  PDT_TZ offset    : -1 day, 17:00:00
  Is hardcoded -7h : True
  LOCAL_TZ found   : PDT
  LOCAL_TZ offset  : -1 day, 17:00:00
  Match: LOCAL == PDT_TZ (-7.0h) — OK now, breaks in winter
======================================================================
```

**PDT_TZ hardcoded to -7h?** YES
**LOCAL_TZ dynamic?** YES
**Do they match right now?** YES
**When will they diverge?** November (PST = -8h)

---

## 3. System Prompt Timezone Injection

```
All datetimes use the local timezone: PDT (UTC-07:00).

4. START_WINDOW — earliest datetime the scheduler should begin searching (YYYY-MM-DDTHH:MM:SS-07:00):
   TODAY is 2026-04-09 (Thursday). "tomorrow" is 2026-04-10.
   Use the exact dates below for "next <weekday>" references:
     "next Monday" -> "2026-04-13"
     "next Tuesday" -> "2026-04-14"
     "next Wednesday" -> "2026-04-15"
     "next Thursday" -> "2026-04-16"
     "next Friday" -> "2026-04-10"
     "next Saturday" -> "2026-04-11"
     "next Sunday" -> "2026-04-12"
   - "next week" (no specific day) -> next Monday = "2026-04-13"
   - "this week" (no specific day) -> tomorrow = "2026-04-10"
   - No date reference at all      -> null

5. END_WINDOW — latest datetime the scheduler should stop searching (YYYY-MM-DDTHH:MM:SS-07:00):
   ...
   - "next week" (no specific day) -> next Friday = "2026-04-10"
   - "this week" (no specific day) -> this Friday = "2026-04-10"
   - No date reference at all      -> null

   EXAMPLES (correct search windows):
   - "next Monday" -> start="2026-04-13T09:00:00-07:00"  end="2026-04-13T17:00:00-07:00"
   - "next Monday or Tuesday" -> start="2026-04-13T09:00:00-07:00"  end="2026-04-14T17:00:00-07:00"
   - "next week" -> start="2026-04-13T09:00:00-07:00"  end="2026-04-10T17:00:00-07:00"
   - "Monday at 2pm" -> start="2026-04-13T14:00:00-07:00"  end="2026-04-13T17:00:00-07:00"
```

**Offset injected into prompt:** -07:00
**Name injected into prompt:** PDT
**Next Monday date injected:** 2026-04-13
**Next Friday date injected:** 2026-04-10

**CRITICAL BUG IN SYSTEM PROMPT:** The "next week" example in the prompt ITSELF has end_window = `2026-04-10` (this Friday), not `2026-04-17` (next Friday). Line 85 of the prompt:
```
   - "next week" -> start="2026-04-13T09:00:00-07:00"  end="2026-04-10T17:00:00-07:00"
```
This means the LLM is being **taught to produce an inverted date range** for "next week". The bug is in the prompt, not the model.

Similarly, the rule text at line 73-74 says:
```
   - "next week" (no specific day) -> next Friday = "2026-04-10"
```
`"next Friday" -> "2026-04-10"` is this Friday (tomorrow), not next Friday (2026-04-17). The day-of-week calculation for "next Friday" is wrong — it picks the nearest upcoming Friday rather than the Friday of next week.

---

## 4. Test 1 — "Quick sync next Monday at 2pm?"

### 4a. email_classify LLM Call

**System prompt sent (INPUT[0]):**
Full system prompt as shown in Section 3 above (identical for all tests).

**User message sent (INPUT[1]):**
```
Subject: Quick sync next Monday at 2pm?
From: leelaprasad.dammalapati@sjsu.edu
To: msadi.finalproject@gmail.com
Cc: 

Hi,

Can we meet next Monday at 2pm to go over the project updates?

Thanks,
Leela
```

**Raw model output:**
```
RAW OUTPUT type  : EmailMeetingDetails
RAW OUTPUT str   : is_meeting=True title='Quick sync to go over the project updates' attendees=['leelaprasad.dammalapati@sjsu.edu', 'msadi.finalproject@gmail.com'] start_window='2026-04-13T14:00:00-07:00' end_window='2026-04-13T17:00:00-07:00' time_confidence='high'
RAW OUTPUT .is_meeting           = True
RAW OUTPUT .title                = Quick sync to go over the project updates
RAW OUTPUT .attendees            = ['leelaprasad.dammalapati@sjsu.edu', 'msadi.finalproject@gmail.com']
RAW OUTPUT .start_window         = 2026-04-13T14:00:00-07:00
RAW OUTPUT .end_window           = 2026-04-13T17:00:00-07:00
RAW OUTPUT .time_confidence      = high
```

**Parsed result:**
| Field | Raw value from log |
|---|---|
| `is_meeting` | True |
| `time_confidence` | high |
| `start_window` | 2026-04-13T14:00:00-07:00 |
| `end_window` | 2026-04-13T17:00:00-07:00 |
| Offset in start_window | -07:00 |
| Offset in end_window | -07:00 |

**Timezone correct?** YES
**Notes:** All fields correct. Model correctly identified Monday 2pm, used -07:00 offset.

---

### 4b. CoD Proposer LLM Call

**Full input sent:**
```
INPUT[0] [SystemMessage]:
You are the Proposer in a meeting scheduling debate.
You are given the original email intent AND the calendar availability of all attendees.

Your job: pick the single best 1-hour slot from the candidate list.
Rules (in priority order):
  1. Respect any time preferences in the email (e.g. 'morning', 'after 2pm', 'before noon') — these take highest priority
  2. Prefer [ALL FREE] slots over conflict slots
  3. Among free slots, prefer mid-morning (10–11am) unless email says otherwise
  4. Only consider NON-URGENT/displaceable conflicts if no free slot exists
  5. Never pick a slot with an URGENT conflict
Argue your choice in 1–3 sentences referencing the email context.
Return a SlotProposal with proposed_slot as the ISO 8601 start datetime from the list.

INPUT[1] [HumanMessage]:
=== EMAIL INTENT ===
Meeting title    : Quick sync to go over the project updates
Attendees        : leelaprasad.dammalapati@sjsu.edu, msadi.finalproject@gmail.com
Requested window : 2026-04-13T14:00:00-07:00 to 2026-04-13T17:00:00-07:00
Time confidence  : high
Email snippet    : Hi,  Can we meet next Monday at 2pm to go over the project updates?  Thanks, Leela

=== CALENDAR AVAILABILITY (within search window) ===
Search window : Monday, April 13 2026
Participants  : leelaprasad.dammalapati@sjsu.edu, msadi.finalproject@gmail.com
Candidate slots (free or single displaceable conflict):
  - Monday, Apr 13  02:00 PM - 03:00 PM PDT  [ALL FREE]  (start=2026-04-13T14:00:00-07:00)
  - Monday, Apr 13  03:00 PM - 04:00 PM PDT  [ALL FREE]  (start=2026-04-13T15:00:00-07:00)
  - Monday, Apr 13  04:00 PM - 05:00 PM PDT  [ALL FREE]  (start=2026-04-13T16:00:00-07:00)
```

**Raw output:**
```
RAW OUTPUT type  : SlotProposal
RAW OUTPUT str   : proposed_slot='2026-04-13T14:00:00-07:00' argument='The preferred meeting time is specified as 2pm by Leela in the email. All available slots from 2pm to 5pm are free for both attendees.'
RAW OUTPUT .proposed_slot        = 2026-04-13T14:00:00-07:00
RAW OUTPUT .argument             = The preferred meeting time is specified as 2pm by Leela in the email. All available slots from 2pm to 5pm are free for both attendees.
```

**proposed_slot value:** 2026-04-13T14:00:00-07:00
**argument value:** The preferred meeting time is specified as 2pm by Leela in the email. All available slots from 2pm to 5pm are free for both attendees.
**Timezone offset in proposed_slot:** -07:00

---

### 4c. CoD Challenger LLM Call

**Full input sent:**
Same email intent + calendar availability as Proposer, plus:
```
Proposer chose: 2026-04-13T14:00:00-07:00
Proposer argument: The preferred meeting time is specified as 2pm by Leela in the email. All available slots from 2pm to 5pm are free for both attendees.
```

**Raw output:**
```
RAW OUTPUT type  : SlotChallenge
RAW OUTPUT str   : agrees=True counter_slot='' argument=''
RAW OUTPUT .agrees               = True
RAW OUTPUT .counter_slot         = 
RAW OUTPUT .argument             = 
```

**agrees value:** True
**counter_slot value:** (empty)
**argument value:** (empty)

---

### 4d. CoD Judge LLM Call

**Full input sent:**
Same email intent + calendar availability, plus:
```
Proposer chose: 2026-04-13T14:00:00-07:00
Proposer argument: The preferred meeting time is specified as 2pm by Leela in the email. All available slots from 2pm to 5pm are free for both attendees.

Challenger agreed: 
Challenger argument: 
```

**Raw output:**
```
RAW OUTPUT type  : SlotVerdict
RAW OUTPUT str   : final_slot_start='2026-04-13T14:00:00-07:00' final_slot_end='2026-04-13T15:00:00-07:00' reason='The slot at 2pm aligns with the requested time in the email and is the earliest available time within the specified window.'
RAW OUTPUT .final_slot_start     = 2026-04-13T14:00:00-07:00
RAW OUTPUT .final_slot_end       = 2026-04-13T15:00:00-07:00
RAW OUTPUT .reason               = The slot at 2pm aligns with the requested time in the email and is the earliest available time within the specified window.
```

**final_slot_start value:** 2026-04-13T14:00:00-07:00
**final_slot_end value:** 2026-04-13T15:00:00-07:00
**reason value:** The slot at 2pm aligns with the requested time in the email and is the earliest available time within the specified window.
**Timezone offset in final_slot_start:** -07:00
**Correct day and time?** YES

---

### 4e. Timezone Analysis (from log)

```
TIMEZONE ANALYSIS after classify:
  meeting_start = 2026-04-13T14:00:00-07:00
    OK: -07:00 PDT offset present
    parsed hour (as-is)  : 14
    local hour           : 14
    local datetime       : 2026-04-13T14:00:00-07:00
  meeting_end = 2026-04-13T17:00:00-07:00
    OK: -07:00 PDT offset present
    parsed hour (as-is)  : 17
    local hour           : 17
    local datetime       : 2026-04-13T17:00:00-07:00

TIMEZONE ANALYSIS after CoD:
  meeting_start = 2026-04-13T14:00:00-07:00
    OK: -07:00 PDT
    local time: Mon Apr 13 02:00 PM PDT
  meeting_end = 2026-04-13T15:00:00-07:00
    OK: -07:00 PDT
    local time: Mon Apr 13 03:00 PM PDT
```

---

## 5. Test 2 — "Sync next week?"

### 5a. email_classify LLM Call

**Raw model output:**
```
RAW OUTPUT type  : EmailMeetingDetails
RAW OUTPUT str   : is_meeting=True title='Sync next week' attendees=['leelaprasad.dammalapati@sjsu.edu', 'msadi.finalproject@gmail.com'] start_window='2026-04-13T09:00:00-07:00' end_window='2026-04-10T17:00:00-07:00' time_confidence='low'
RAW OUTPUT .is_meeting           = True
RAW OUTPUT .title                = Sync next week
RAW OUTPUT .attendees            = ['leelaprasad.dammalapati@sjsu.edu', 'msadi.finalproject@gmail.com']
RAW OUTPUT .start_window         = 2026-04-13T09:00:00-07:00
RAW OUTPUT .end_window           = 2026-04-10T17:00:00-07:00
RAW OUTPUT .time_confidence      = low
```

**Parsed result:**
| Field | Raw value |
|---|---|
| `time_confidence` | low |
| `start_window` | 2026-04-13T09:00:00-07:00 |
| `end_window` | 2026-04-10T17:00:00-07:00 |
| end_window date | 2026-04-10 (this Friday) |
| start_window date | 2026-04-13 (next Monday) |
| end > start? | **NO** |

**Bug observed (end < start)?** YES
**end_window value:** 2026-04-10T17:00:00-07:00
**start_window value:** 2026-04-13T09:00:00-07:00

**Root cause:** The system prompt itself teaches the LLM this wrong example:
```
   - "next week" -> start="2026-04-13T09:00:00-07:00"  end="2026-04-10T17:00:00-07:00"
```
The end date `2026-04-10` is calculated from `"next Friday" -> "2026-04-10"`, which is actually THIS Friday (tomorrow), not the Friday of next week (`2026-04-17`). **The model is correctly following the prompt — the prompt is wrong.**

---

### 5b. CoD Proposer, Challenger, Judge

**Proposer raw output:**
```
RAW OUTPUT type  : SlotProposal
RAW OUTPUT str   : proposed_slot='2026-04-13T10:00:00-07:00' argument='Given the email does not specify a preferred time and all suggested slots are available, the mid-morning slot at 10:00 AM is chosen as it aligns with the preference for this time frame.'
RAW OUTPUT .proposed_slot        = 2026-04-13T10:00:00-07:00
RAW OUTPUT .argument             = Given the email does not specify a preferred time and all suggested slots are available, the mid-morning slot at 10:00 AM is chosen as it aligns with the preference for this time frame.
```

**Challenger raw output:**
```
RAW OUTPUT type  : SlotChallenge
RAW OUTPUT str   : agrees=True counter_slot='' argument=''
RAW OUTPUT .agrees               = True
RAW OUTPUT .counter_slot         = 
RAW OUTPUT .argument             = 
```

**Judge raw output:**
```
RAW OUTPUT type  : SlotVerdict
RAW OUTPUT str   : final_slot_start='2026-04-13T09:00:00-07:00' final_slot_end='2026-04-13T10:00:00-07:00' reason='The earliest available slot on the requested day is preferred.'
RAW OUTPUT .final_slot_start     = 2026-04-13T09:00:00-07:00
RAW OUTPUT .final_slot_end       = 2026-04-13T10:00:00-07:00
RAW OUTPUT .reason               = The earliest available slot on the requested day is preferred.
```

**Final verdict start:** 2026-04-13T09:00:00-07:00
**Final verdict end:** 2026-04-13T10:00:00-07:00
**Days actually searched (from slot_cod log):** Mon Apr 13 only (1 day, 8 free slots)
**Correct (should be Mon–Fri)?** NO — should have searched Mon Apr 13 through Fri Apr 17

---

### 5c. Timezone Analysis (from log)

```
TIMEZONE ANALYSIS after classify:
  meeting_start = 2026-04-13T09:00:00-07:00
    OK: -07:00 PDT offset present
    parsed hour (as-is)  : 9
    local hour           : 9
    local datetime       : 2026-04-13T09:00:00-07:00
  meeting_end = 2026-04-10T17:00:00-07:00
    OK: -07:00 PDT offset present
    parsed hour (as-is)  : 17
    local hour           : 17
    local datetime       : 2026-04-10T17:00:00-07:00

TIMEZONE ANALYSIS after CoD:
  meeting_start = 2026-04-13T09:00:00-07:00
    OK: -07:00 PDT
    local time: Mon Apr 13 09:00 AM PDT
  meeting_end = 2026-04-13T10:00:00-07:00
    OK: -07:00 PDT
    local time: Mon Apr 13 10:00 AM PDT
```

---

## 6. Test 3 — "Let's connect"

### 6a. email_classify LLM Call

**Raw model output:**
```
RAW OUTPUT type  : EmailMeetingDetails
RAW OUTPUT str   : is_meeting=True title='Discuss roadmap' attendees=['leelaprasad.dammalapati@sjsu.edu', 'msadi.finalproject@gmail.com'] start_window='2026-04-10T09:00:00-07:00' end_window='2026-04-10T17:00:00-07:00' time_confidence='medium'
RAW OUTPUT .is_meeting           = True
RAW OUTPUT .title                = Discuss roadmap
RAW OUTPUT .attendees            = ['leelaprasad.dammalapati@sjsu.edu', 'msadi.finalproject@gmail.com']
RAW OUTPUT .start_window         = 2026-04-10T09:00:00-07:00
RAW OUTPUT .end_window           = 2026-04-10T17:00:00-07:00
RAW OUTPUT .time_confidence      = medium
```

**Parsed result:**
| Field | Raw value |
|---|---|
| `time_confidence` | medium |
| `start_window` | 2026-04-10T09:00:00-07:00 |
| `end_window` | 2026-04-10T17:00:00-07:00 |

**Should start_window be null?** YES
**Is start_window actually null?** NO
**Model hallucinated a date?** YES — assigned tomorrow (Fri Apr 10) with `medium` confidence for "connect soon" with no date reference.

**Notes:** The prompt rule says `No date reference at all -> null` and `"none" = no time or date information`. The base model ignores this and guesses tomorrow. The confidence should be `none`, not `medium`.

---

### 6b. CoD (classify returned is_meeting=True)

**Proposer raw output:**
```
RAW OUTPUT type  : SlotProposal
RAW OUTPUT str   : proposed_slot='2026-04-10T10:00:00-07:00' argument='The preferred meeting time is not specified beyond the general request for a connection soon. Given this, we should aim for a time that maximizes productivity, which aligns with the rule to prefer mid-morning slots when no specific time preference is mentioned. Therefore, the optimal slot is 10:00 AM to 11:00 AM on April 10th, 2026.'
RAW OUTPUT .proposed_slot        = 2026-04-10T10:00:00-07:00
RAW OUTPUT .argument             = The preferred meeting time is not specified beyond the general request for a connection soon. Given this, we should aim for a time that maximizes productivity, which aligns with the rule to prefer mid-morning slots when no specific time preference is mentioned. Therefore, the optimal slot is 10:00 AM to 11:00 AM on April 10th, 2026.
```

**Challenger raw output:**
```
RAW OUTPUT type  : SlotChallenge
RAW OUTPUT str   : agrees=True counter_slot='' argument="The proposer's choice of 10:00 AM to 11:00 AM aligns well with the email intent which did not specify a particular time preference. The chosen time falls within the available free slots and follows the guideline to prefer mid-morning slots."
RAW OUTPUT .agrees               = True
RAW OUTPUT .counter_slot         = 
RAW OUTPUT .argument             = The proposer's choice of 10:00 AM to 11:00 AM aligns well with the email intent which did not specify a particular time preference. The chosen time falls within the available free slots and follows the guideline to prefer mid-morning slots.
```

**Judge raw output:**
```
RAW OUTPUT type  : SlotVerdict
RAW OUTPUT str   : final_slot_start='2026-04-10T10:00:00-07:00' final_slot_end='2026-04-10T11:00:00-07:00' reason='This slot is preferred as it aligns with the guideline to choose mid-morning times when no specific time preference is mentioned.'
RAW OUTPUT .final_slot_start     = 2026-04-10T10:00:00-07:00
RAW OUTPUT .final_slot_end       = 2026-04-10T11:00:00-07:00
RAW OUTPUT .reason               = This slot is preferred as it aligns with the guideline to choose mid-morning times when no specific time preference is mentioned.
```

**Final verdict:** 2026-04-10T10:00:00-07:00 to 2026-04-10T11:00:00-07:00 (Fri Apr 10 10am)

---

### 6c. Timezone Analysis (from log)

```
TIMEZONE ANALYSIS after classify:
  meeting_start = 2026-04-10T09:00:00-07:00
    OK: -07:00 PDT offset present
    parsed hour (as-is)  : 9
    local hour           : 9
    local datetime       : 2026-04-10T09:00:00-07:00
  meeting_end = 2026-04-10T17:00:00-07:00
    OK: -07:00 PDT offset present
    parsed hour (as-is)  : 17
    local hour           : 17
    local datetime       : 2026-04-10T17:00:00-07:00

TIMEZONE ANALYSIS after CoD:
  meeting_start = 2026-04-10T10:00:00-07:00
    OK: -07:00 PDT
    local time: Fri Apr 10 10:00 AM PDT
  meeting_end = 2026-04-10T11:00:00-07:00
    OK: -07:00 PDT
    local time: Fri Apr 10 11:00 AM PDT
```

---

## 7. Bugs Found

| # | Test | Step | What log shows | Expected | Impact |
|---|---|---|---|---|---|
| 1 | Test 2 | System prompt (line 85/73-74) | `"next week" -> end="2026-04-10T17:00:00-07:00"` and `"next Friday" -> "2026-04-10"` | end should be `2026-04-17` (next Friday of next week) | **Prompt teaches the LLM to produce inverted date range for "next week".** The model is correctly following wrong instructions. This is a `_build_email_system_prompt()` bug, not a model bug. |
| 2 | Test 2 | `_determine_search_days` | Only Mon Apr 13 searched (1 day, 8 slots) | Should search Mon–Fri of next week (5 days) | Week-level emails only search 1 day instead of 5 |
| 3 | Test 3 | email_classify output | `time_confidence=medium`, `start_window=2026-04-10`, `end_window=2026-04-10` | `time_confidence=none`, `start_window=null`, `end_window=null` | Base model ignores the `null` / `none` rules in the prompt for vague "soon" emails — guesses tomorrow with medium confidence |
| 4 | All | `_normalize_iso_dt` (not exercised) | PDT_TZ hardcoded to -7h | Should use LOCAL_TZ | Events will be 1 hour off during PST (Nov–Mar) |

**NEW FINDING (not in previous reports):** Bug #1 above — the "next week" `end_window` bug is **in the system prompt itself**, not in the model. The `_build_email_system_prompt()` function computes "next Friday" as the nearest upcoming Friday (2026-04-10, tomorrow) instead of the Friday of next week (2026-04-17). The model is faithfully following the prompt's own example. This is a code bug in the weekday calculation logic, not a model hallucination.

---

## 8. LLM Call Count

| Test | Classify | Proposer | Challenger | Judge | Total |
|---|---|---|---|---|---|
| Test 1 | 1 | 1 | 1 | 1 | 4 |
| Test 2 | 1 | 1 | 1 | 1 | 4 |
| Test 3 | 1 | 1 | 1 | 1 | 4 |
| **Total** | **3** | **3** | **3** | **3** | **12** |

---

## 9. Full Log File

Saved at: `TIMEZONE_LLM_FULL_LOG.txt` (1045 lines)
