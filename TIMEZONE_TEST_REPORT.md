# Timezone Test Report
**Date:** 2026-04-09
**System timezone:** PDT
**UTC offset:** -0700 (-07:00)
**Model:** Qwen/Qwen2.5-14B-Instruct-AWQ (base model, no LoRA)
**Note:** vLLM had LoRA adapters loaded (slack, email, meeting) but all API calls target the base model by name.

---

## 1. Timezone Code Audit

### PDT_TZ definition
```
calendar_agent.py:38:PDT_TZ = timezone(timedelta(hours=-7), name="PDT")
```
Hardcoded to UTC-7. Does NOT account for PST (UTC-8) in winter.

### _build_email_system_prompt injection
```
calendar_agent.py:89:    _now        = datetime.now().astimezone()
calendar_agent.py:90:    _tz_name    = _now.strftime("%Z")                          # e.g. "IST", "PDT", "EST"
calendar_agent.py:91:    _raw_offset = _now.strftime("%z")                          # e.g. "+0530", "-0700"
calendar_agent.py:92:    _tz_offset  = _raw_offset[:3] + ":" + _raw_offset[3:]     # e.g. "+05:30", "-07:00"
```
The **system prompt** correctly derives the local timezone dynamically from `datetime.now().astimezone()`. This tells the LLM to use `PDT (-07:00)` in its output.

### _normalize_iso_dt function
```python
# calendar_agent.py:554-569
def _normalize_iso_dt(raw_value: str | None) -> str | None:
    """Normalize model datetime strings to valid RFC3339 with timezone."""
    if not raw_value:
        return None
    text = str(raw_value).strip()

    # Fix common malformed offsets like "2026-04-09T10:00:00 00:00" -> "+00:00".
    text = re.sub(r"([0-9])\s([+-]\d{2}:\d{2})$", r"\1\2", text)
    text = re.sub(r"([0-9])\s(\d{2}:\d{2})$", r"\1+\2", text)

    dt = dp(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=PDT_TZ)          # <-- assumes PDT if naive
    else:
        dt = dt.astimezone(PDT_TZ)              # <-- converts to PDT always
    return dt.isoformat()
```
**Bug:** Forces all datetimes to `PDT_TZ` (hardcoded UTC-7). Should use `LOCAL_TZ` (dynamic). Currently masks the issue because local IS PDT.

### _determine_search_days hour extraction
```python
# calendar_cod.py:519-549
start_hour = datetime.fromisoformat(str(start_str)).hour   # line 543
end_hour   = datetime.fromisoformat(str(end_str)).hour     # line 547
```
Uses `.hour` on the parsed datetime WITHOUT converting to local timezone first. Works when the offset in the string matches the local timezone (currently true). Would break if the input had a Z suffix or different offset — `.hour` would return the UTC hour, not the local hour.

### LOCAL_TZ in calendar_cod.py
```
calendar_cod.py:236:LOCAL_TZ = datetime.now().astimezone().tzinfo
```
Correctly derived from system time. Used for slot windows and event conversion. **Good.**

---

## 2. Mismatch Analysis (from Step 6)

```
=== CURRENT SYSTEM TIMEZONE ===
  datetime.now().astimezone() : 2026-04-09 11:13:03.102206-07:00
  tzinfo                      : PDT
  UTC offset                  : -0700
  TZ name                     : PDT

=== PDT_TZ AS HARDCODED IN calendar_agent.py ===
  PDT_TZ                      : PDT
  Current time in PDT_TZ      : 2026-04-09 11:13:03.102237-07:00

=== WHAT _build_email_system_prompt() INJECTS ===
  _tz_name   : PDT
  _tz_offset : -07:00
  Example output field: "2026-04-13T14:00:00-07:00"

=== MISMATCH CHECK ===
  OK: local offset matches PDT_TZ (-7.0h) — no error right now
  WARNING: This will break in winter (PST = -8h)

=== SIMULATE _normalize_iso_dt ON BASE MODEL OUTPUT ===
  Base model output           : 2026-04-13T14:00:00-07:00
  After astimezone(PDT_TZ)    : 2026-04-13T14:00:00-07:00  <- what calendar_agent.py does
  After astimezone(LOCAL_TZ)  : 2026-04-13T14:00:00-07:00  <- what it should do
  Same result? True
```

**Is there a mismatch right now?** NO — both PDT_TZ and LOCAL_TZ are -07:00.
**When will it break?** In winter (November–March) when the system switches to PST (UTC-8). The prompt will tell the LLM to use `-08:00`, but `_normalize_iso_dt` will force-convert to PDT (`-07:00`), shifting every event by 1 hour.

### Step 7 — _determine_search_days Hour Extraction

```
start_window from state : 2026-04-13T14:00:00-07:00
end_window from state   : 2026-04-13T17:00:00-07:00

=== CURRENT CODE (fromisoformat) ===
  start_hour : 14  (OK)
  end_hour   : 17  (OK)

=== IF START_WINDOW HAD Z SUFFIX (2026-04-13T21:00:00Z) ===
  Raw hour from Z string      : 21  (UTC hour — WRONG for slot scanning)
  Hour after local conversion : 14  (OK)
```

Hour extraction works correctly today because the LLM outputs the local offset. If input ever arrived in UTC (Z suffix), the code would use the UTC hour (21) instead of the local hour (14), scanning the wrong time window.

---

## 3. Test 1 — Specific Day + Time

**Email:** "Quick sync next Monday at 2pm?"
**Expected:** window = Monday 14:00 → 17:00, correct local offset

| Field | Value | Correct? |
|---|---|---|
| `time_confidence` | high | YES |
| `start_window` from classify | 2026-04-13T14:00:00-07:00 | YES |
| `end_window` from classify | 2026-04-13T17:00:00-07:00 | YES |
| Days scanned | Mon Apr 13 (1 day) | YES |
| `start_hour` used | 14 | YES |
| `end_hour` used | 17 | YES |
| CoD proposer | 2026-04-13T14:00:00-07:00 | YES |
| CoD challenger | agrees=True | YES |
| CoD verdict start | 2026-04-13T14:00:00-07:00 | YES |
| CoD verdict end | 2026-04-13T15:00:00-07:00 | YES |
| Timezone offset in verdict | -07:00 (PDT) | YES |

**Notes:** Pipeline worked correctly end-to-end. Base model correctly identified Monday at 2pm, produced the right offset, CoD chose the exact slot. Failed only at Slack notification (SSL cert issue — unrelated to timezone).

---

## 4. Test 2 — Week-Level Reference

**Email:** "Sync next week?"
**Expected:** window = Monday 09:00 → Friday 17:00 of next week (Apr 13–17)

| Field | Value | Correct? |
|---|---|---|
| `time_confidence` | low | YES |
| `start_window` from classify | 2026-04-13T09:00:00-07:00 | YES (Monday) |
| `end_window` from classify | 2026-04-10T17:00:00-07:00 | **NO — should be 2026-04-17** |
| Days scanned | Mon Apr 13 only (1 day) | **NO — should be Mon–Fri (5 days)** |
| `start_hour` used | 9 | YES |
| `end_hour` used | 17 | YES |
| CoD proposer | 2026-04-13T09:00:00-07:00 | Correct for searched day |
| CoD verdict start | 2026-04-13T09:00:00-07:00 | Partial — only Mon was searched |
| Timezone offset in verdict | -07:00 (PDT) | YES |

**Notes:**
- **BASE MODEL BUG:** The LLM set `end_window` to `2026-04-10` (this Friday, April 10) instead of `2026-04-17` (next Friday). The email says "next week" but the model picked this week's Friday.
- **CODE LOGIC BUG:** Because `end_date (Apr 10) < start_date (Apr 13)`, `_determine_search_days` finds no weekdays in the range [Apr 13 → Apr 10] (empty), then falls back to just `start_date` alone (Apr 13). The full-week scan never happens.
- The code at `calendar_cod.py:538-559` does not validate that `end_date >= start_date`. It silently degrades to single-day search.
- This is a **combined model + code** issue: the model gave a wrong `end_window`, AND the code has no guard for `end < start`.

---

## 5. Test 3 — No Date

**Email:** "Lets connect soon"
**Expected:** window = None → full next-week fallback scan, confidence = none

| Field | Value | Correct? |
|---|---|---|
| `time_confidence` | medium | **NO — expected none/low** |
| `start_window` from classify | 2026-04-10T09:00:00-07:00 | **NO — should be null** |
| `end_window` from classify | 2026-04-10T17:00:00-07:00 | **NO — should be null** |
| Days scanned | Fri Apr 10 only (1 day) | **NO — should fall back to next week** |
| CoD proposer | 2026-04-10T10:00:00-07:00 | Correct for searched day |
| CoD verdict start | 2026-04-10T09:00:00-07:00 | Wrong day — should be next week |
| Timezone offset in verdict | -07:00 (PDT) | YES |

**Notes:**
- **BASE MODEL BUG:** The LLM classified "connect soon" as `confidence: medium` with a specific date (tomorrow), when it should have returned `null` windows and `none`/`low` confidence. The email has no date reference at all.
- The code's next-week fallback in `_determine_search_days` never triggers because the LLM always provides a window. The base model (no fine-tuning) is too eager to guess dates.
- This means the confidence-based fallback logic in the code is effectively dead code with the base model.

---

## 6. Identified Bugs

List every timezone issue found, in order of severity:

| # | File | Line | Bug | Impact |
|---|---|---|---|---|
| 1 | calendar_agent.py | 38, 566, 568 | `PDT_TZ` hardcoded to UTC-7. `_normalize_iso_dt` forces all datetimes to PDT. | **Events created 1 hour off during PST (Nov–Mar).** Currently masked because system is in PDT. |
| 2 | calendar_cod.py | 543, 547 | `_determine_search_days` extracts `.hour` without converting to local timezone first. | **Wrong hour bounds if input arrives in UTC (Z suffix).** Currently works because LLM outputs local offset. |
| 3 | calendar_cod.py | 538-559 | No validation that `end_date >= start_date`. Silently degrades to single-day when end < start. | **Week-level searches collapse to 1 day when LLM returns inverted dates** (observed in Test 2). |
| 4 | calendar_agent.py | 254 / calendar_cod.py | Base model over-classifies: gives `medium` confidence + specific date for emails with no date reference. | **Next-week fallback never triggers** — code path is dead with base model. |
| 5 | calendar_agent.py | 254 | Base model returns wrong `end_window` date for "next week" (picks current week's Friday instead of next week's). | **Week range scanned is wrong** — only 1 day searched instead of 5. |

---

## 7. Recommended Fixes

### Fix 1 — Replace hardcoded `PDT_TZ` with dynamic `LOCAL_TZ` (calendar_agent.py)

```python
# Line 38: Replace
PDT_TZ = timezone(timedelta(hours=-7), name="PDT")

# With
LOCAL_TZ = datetime.now().astimezone().tzinfo
```

Update all references to `PDT_TZ` → `LOCAL_TZ` on lines 566, 568, 575.

### Fix 2 — Convert to local timezone before extracting hour (calendar_cod.py)

```python
# Lines 543-547: Replace
start_hour = datetime.fromisoformat(str(start_str)).hour

# With
start_hour = datetime.fromisoformat(str(start_str)).astimezone(LOCAL_TZ).hour
```

Same for `end_hour`.

### Fix 3 — Add end >= start validation in `_determine_search_days` (calendar_cod.py)

```python
# After line 539, add:
if end_date < start_date:
    end_date = start_date + timedelta(days=4)  # default to 5 weekdays
```

### Fix 4 — Base model confidence calibration (requires fine-tuning or prompt engineering)

The base model assigns `medium` confidence to emails with no date. Options:
- Add stricter examples in the system prompt for the `none` confidence case
- Add a post-classification validation: if `time_confidence` is `medium`/`high` but the email body contains no date-like tokens, override to `low`
- Fine-tune (LoRA) the email classifier to better calibrate confidence

### Fix 5 — Base model date-range accuracy (requires fine-tuning or prompt engineering)

The base model misidentifies "next week" end date. Options:
- Add more explicit examples for week-level ranges in the system prompt
- Add validation: if start_window is next Monday, end_window should be >= start_window
- Fine-tune (LoRA) for better date-range extraction
