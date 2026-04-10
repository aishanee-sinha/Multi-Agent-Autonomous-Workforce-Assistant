# Changes Made From Original Version

**Date:** 2026-04-09
**Branch:** aishanee-codes-cod

---

## Code Changes

### 1. Fix: Wrong "next weekday" date calculation in system prompt

**File:** `calendar_agent.py` (lines 74-78)
**Function:** `_build_email_system_prompt()`

**Problem:** The original code computed "next \<weekday\>" dates by finding the nearest upcoming occurrence of that weekday. For weekdays earlier in the week than today, this produced dates in the *current* week instead of *next* week. For example, on Thursday April 9:

- "next Friday" resolved to April 10 (tomorrow) instead of April 17 (next week's Friday)
- "next Saturday" resolved to April 11 instead of April 18

This caused the "next week" example in the LLM prompt to have `end_window` (April 10) *before* `start_window` (April 13), which the model faithfully reproduced. The downstream `_determine_search_days` function then silently degraded week-level searches to a single-day scan.

**Original code:**
```python
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
next_weekdays = {}
for target_wd, name in enumerate(weekday_names):
    days_ahead = (target_wd - today.weekday()) % 7 or 7
    next_weekdays[name] = today + timedelta(days=days_ahead)
```

**Fixed code:**
```python
weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
next_monday = today + timedelta(days=(7 - today.weekday()))
next_weekdays = {}
for target_wd, name in enumerate(weekday_names):
    next_weekdays[name] = next_monday + timedelta(days=target_wd)
```

**How it works:** Anchors on next Monday (always 1-7 days ahead), then offsets each weekday from that Monday. All "next \<weekday\>" dates now land in the same Mon-Sun week.

**Verified result (when today = Thursday April 9):**
```
"next Monday"    -> 2026-04-13  (was 2026-04-13, unchanged)
"next Tuesday"   -> 2026-04-14  (was 2026-04-14, unchanged)
"next Wednesday" -> 2026-04-15  (was 2026-04-15, unchanged)
"next Thursday"  -> 2026-04-16  (was 2026-04-16, unchanged)
"next Friday"    -> 2026-04-17  (was 2026-04-10, FIXED)
"next Saturday"  -> 2026-04-18  (was 2026-04-11, FIXED)
"next Sunday"    -> 2026-04-19  (was 2026-04-12, FIXED)

"next week" example -> start=2026-04-13, end=2026-04-17  (was end=2026-04-10, FIXED)
```

**Impact:** "Sync next week?" emails now correctly scan Mon-Fri (5 days) instead of collapsing to Monday only.

---

## Git Diff

```diff
diff --git a/calendar_agent.py b/calendar_agent.py
--- a/calendar_agent.py
+++ b/calendar_agent.py
@@ -72,10 +72,10 @@ def _build_email_system_prompt() -> str:
 
     # Compute "next <weekday>" for all 7 days dynamically — no hardcoding
     weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
+    next_monday = today + timedelta(days=(7 - today.weekday()))
     next_weekdays = {}
     for target_wd, name in enumerate(weekday_names):
-        days_ahead = (target_wd - today.weekday()) % 7 or 7
-        next_weekdays[name] = today + timedelta(days=days_ahead)
+        next_weekdays[name] = next_monday + timedelta(days=target_wd)
```

---

## Test Reports Generated (no code changes, reference only)

| File | Description |
|---|---|
| `TIMEZONE_TEST_REPORT.md` | Base model timezone handling audit across 3 synthetic email scenarios |
| `REAL_EMAIL_TEST_REPORT.md` | End-to-end test with real Gmail emails fetched via API |
| `TIMEZONE_LLM_TEST_REPORT.md` | Full raw LLM call capture for all 12 LLM calls (3 tests x 4 calls each) |
| `TIMEZONE_LLM_FULL_LOG.txt` | Complete 1045-line debug log from the LLM capture test |

---

## Known Issues Not Yet Fixed

These were identified during testing but no code changes have been made for them:

| # | File | Bug | Status |
|---|---|---|---|
| 1 | `calendar_agent.py:38,566,568` | `PDT_TZ` hardcoded to UTC-7; `_normalize_iso_dt` forces all datetimes to PDT. Will shift events 1 hour during PST (Nov-Mar). | Not fixed |
| 2 | `calendar_cod.py:543,547` | `_determine_search_days` extracts `.hour` without converting to local timezone first. Breaks if input has Z suffix. | Not fixed |
| 3 | `calendar_cod.py:538-559` | No validation that `end_date >= start_date`. Silently degrades to single-day search on inverted ranges. | Not fixed |
| 4 | `calendar_agent.py:254` | Base model assigns `medium` confidence + specific date for emails with no date reference ("connect soon"). Next-week fallback never triggers. | Not fixed (model behavior) |
