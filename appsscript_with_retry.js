/**
 * meeting-summarizer Apps Script
 * ================================
 * Watches a Google Drive folder for new .txt transcripts and triggers
 * the meeting-agent Lambda via HTTP POST.
 *
 * Retry logic (replaces S3 lock dependency):
 *   - Files are tracked in Script Properties with status:
 *       pending   → sent to Lambda, awaiting Slack confirm/cancel
 *       confirmed → user clicked Confirm in Slack
 *       cancelled → user clicked Cancel in Slack
 *       failed    → Lambda returned error
 *   - A file in "pending" state for > RETRY_AFTER_MS is re-triggered.
 *   - A file in "confirmed" or "cancelled" state is never re-triggered.
 *   - Files older than FORGET_AFTER_MS are removed from tracking.
 *
 * This mirrors how SQS visibility timeout works:
 *   message invisible while being processed → reappears if not deleted.
 */

// ── Configuration ─────────────────────────────────────────────────────────────
const LAMBDA_URL        = "https://mpyfhaq9kf.execute-api.us-east-1.amazonaws.com/prod/meeting-agent-handler";
const WEBHOOK_SECRET    = "meeting-summarizer-secret-2026";
const DRIVE_FOLDER_ID   = "1sC7OSWfkRSoldvMk8PLwX5SOL739HUS";

// Retry a pending job after 15 minutes (Lambda should finish in ~8 min)
const RETRY_AFTER_MS    = 15 * 60 * 1000;

// Stop tracking a file after 2 hours (covers all retries)
const FORGET_AFTER_MS   = 2 * 60 * 60 * 1000;

// Max retry attempts before giving up
const MAX_RETRIES       = 3;


// ── Main trigger (runs every minute via time-driven trigger) ──────────────────
function checkForNewTranscripts() {
  const props    = PropertiesService.getScriptProperties();
  const tracking = _loadTracking(props);
  const now      = Date.now();

  // ── Step 1: Clean up old finished/expired entries ─────────────────────────
  for (const fileId of Object.keys(tracking)) {
    const entry = tracking[fileId];
    const age   = now - entry.firstSeenAt;
    if (age > FORGET_AFTER_MS) {
      delete tracking[fileId];
      Logger.log("Forgot old entry: %s (%s)", fileId, entry.fileName);
    }
  }

  // ── Step 2: Scan Drive folder for .txt files ──────────────────────────────
  const folder = DriveApp.getFolderById(DRIVE_FOLDER_ID);
  const files  = folder.getFilesByType(MimeType.PLAIN_TEXT);
  const newFiles = [];

  while (files.hasNext()) {
    const file   = files.next();
    const fileId = file.getId();
    const entry  = tracking[fileId];

    if (!entry) {
      // Brand new file — trigger immediately
      newFiles.push({ id: fileId, name: file.getName() });
    } else if (
      entry.status === "pending" &&
      (now - entry.lastSentAt) > RETRY_AFTER_MS &&
      entry.retries < MAX_RETRIES
    ) {
      // Pending too long — retry
      Logger.log("Retrying stale pending job: %s (attempt %d)", entry.fileName, entry.retries + 1);
      newFiles.push({ id: fileId, name: file.getName(), isRetry: true });
    }
    // confirmed / cancelled / max retries reached → skip
  }

  if (newFiles.length === 0) {
    Logger.log("No new or retryable files found.");
    _saveTracking(props, tracking);
    return;
  }

  Logger.log("Found %d file(s) to process.", newFiles.length);

  // ── Step 3: Trigger Lambda for each file ──────────────────────────────────
  for (const file of newFiles) {
    const success = _triggerLambda(file.id, file.name);

    if (success) {
      const existing = tracking[file.id] || {
        fileName:    file.name,
        firstSeenAt: now,
        retries:     0,
      };
      tracking[file.id] = {
        ...existing,
        status:    "pending",
        lastSentAt: now,
        retries:   (existing.retries || 0) + (file.isRetry ? 1 : 0),
      };
      Logger.log("Triggered Lambda for: %s (retries=%d)", file.name, tracking[file.id].retries);
    } else {
      // Lambda call itself failed (network error, bad URL, etc.)
      const existing = tracking[file.id] || {
        fileName:    file.name,
        firstSeenAt: now,
        retries:     0,
      };
      tracking[file.id] = {
        ...existing,
        status:    "pending",  // keep pending so it retries
        lastSentAt: now - (RETRY_AFTER_MS - 60000), // retry sooner (1 min)
        retries:   (existing.retries || 0) + 1,
      };
      Logger.log("Lambda trigger failed for: %s — will retry next poll", file.name);
    }
  }

  _saveTracking(props, tracking);
}


/**
 * Call this from the Lambda (via a separate webhook endpoint) when
 * the user confirms or cancels in Slack. This marks the file as done
 * so it never gets retried.
 *
 * OR: set up a doPost() endpoint in this Apps Script that Lambda calls
 * after meeting_send_email / meeting_post_cancel completes.
 */
function markFileConfirmed(fileId) {
  const props    = PropertiesService.getScriptProperties();
  const tracking = _loadTracking(props);
  if (tracking[fileId]) {
    tracking[fileId].status = "confirmed";
    Logger.log("Marked confirmed: %s", fileId);
    _saveTracking(props, tracking);
  }
}

function markFileCancelled(fileId) {
  const props    = PropertiesService.getScriptProperties();
  const tracking = _loadTracking(props);
  if (tracking[fileId]) {
    tracking[fileId].status = "cancelled";
    Logger.log("Marked cancelled: %s", fileId);
    _saveTracking(props, tracking);
  }
}


/**
 * HTTP POST endpoint — Lambda can call this to mark a file done.
 * Deploy this Apps Script as a Web App (Execute as: Me, Who has access: Anyone)
 * then Lambda can POST: { action: "confirmed"|"cancelled", fileId: "..." }
 */
function doPost(e) {
  try {
    const body = JSON.parse(e.postData.contents);
    if (body.action === "confirmed") markFileConfirmed(body.fileId);
    if (body.action === "cancelled") markFileCancelled(body.fileId);
    return ContentService
      .createTextOutput(JSON.stringify({ ok: true }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: err.message }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}


// ── Helper: trigger Lambda ────────────────────────────────────────────────────
function _triggerLambda(fileId, fileName) {
  const payload = JSON.stringify({
    type:      "new_transcript",
    secret:    WEBHOOK_SECRET,
    file_id:   fileId,
    file_name: fileName,
  });

  const options = {
    method:             "post",
    contentType:        "application/json",
    payload:            payload,
    muteHttpExceptions: true,
  };

  try {
    const response = UrlFetchApp.fetch(LAMBDA_URL, options);
    const code     = response.getResponseCode();
    Logger.log("Lambda response: %d for %s", code, fileName);
    return code >= 200 && code < 300;
  } catch (err) {
    Logger.log("HTTP error calling Lambda: %s", err.message);
    return false;
  }
}


// ── Helper: load/save tracking state from Script Properties ──────────────────
function _loadTracking(props) {
  try {
    const raw = props.getProperty("FILE_TRACKING");
    return raw ? JSON.parse(raw) : {};
  } catch (e) {
    return {};
  }
}

function _saveTracking(props, tracking) {
  props.setProperty("FILE_TRACKING", JSON.stringify(tracking));
}


// ── Helper: view current tracking state (run manually to debug) ──────────────
function viewTracking() {
  const props    = PropertiesService.getScriptProperties();
  const tracking = _loadTracking(props);
  const now      = Date.now();
  for (const [fileId, entry] of Object.entries(tracking)) {
    const ageMins   = Math.round((now - entry.firstSeenAt) / 60000);
    const staleMins = Math.round((now - entry.lastSentAt)  / 60000);
    Logger.log(
      "%s | %s | status=%s retries=%d age=%dm stale=%dm",
      entry.fileName, fileId, entry.status,
      entry.retries, ageMins, staleMins
    );
  }
}

// ── Helper: clear all tracking (run manually to reset) ───────────────────────
function clearTracking() {
  PropertiesService.getScriptProperties().deleteProperty("FILE_TRACKING");
  Logger.log("Tracking cleared.");
}


// ── Setup: create the time-driven trigger (run once manually) ────────────────
function createTrigger() {
  // Delete existing triggers first
  ScriptApp.getProjectTriggers().forEach(t => ScriptApp.deleteTrigger(t));
  // Create new 1-minute polling trigger
  ScriptApp.newTrigger("checkForNewTranscripts")
    .timeBased()
    .everyMinutes(1)
    .create();
  Logger.log("Trigger created: checkForNewTranscripts every 1 minute.");
}
