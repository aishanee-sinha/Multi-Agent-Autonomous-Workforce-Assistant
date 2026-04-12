/**
 * google_apps_script.js
 * Meeting Summarizer - Google Drive Monitor
 *
 * Monitors the transcript_summarizer folder in Google Drive.
 * When a new .txt file appears, calls the API Gateway endpoint
 * which triggers the Lambda function.
 *
 * HOW TO DEPLOY:
 *   1. Go to script.google.com
 *   2. Create a new project named "MeetingSummarizerTrigger"
 *   3. Paste this entire file into Code.gs
 *   4. Edit the CONFIG section below with your values
 *   5. Run setupTrigger() once manually to install the time-based trigger
 *   6. Authorize the script when prompted
 */

// ─────────────────────────────────────────────
// CONFIG - edit these values before deploying
// ─────────────────────────────────────────────
const CONFIG = {
  // The ID from the URL of your transcript_summarizer Drive folder
  // URL looks like: drive.google.com/drive/folders/THIS_PART_IS_THE_ID
  DRIVE_FOLDER_ID: "YOUR_TRANSCRIPT_SUMMARIZER_FOLDER_ID",

  // API Gateway endpoint URL for your Lambda trigger function
  API_GATEWAY_URL: "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod/trigger",

  // A secret token to verify requests came from this script (set same value in Lambda env)
  WEBHOOK_SECRET: "YOUR_SHARED_SECRET_TOKEN",

  // How often to poll in minutes (1, 5, 10, 15, 30)
  POLL_INTERVAL_MINUTES: 2,

  // Script property key used to track processed file IDs
  PROCESSED_KEY: "processed_file_ids",

  // Max entries to keep in processed list (prevents unbounded growth)
  MAX_PROCESSED: 500,
};

// ─────────────────────────────────────────────
// MAIN POLL FUNCTION
// Called automatically by time-based trigger
// ─────────────────────────────────────────────
function checkForNewTranscripts() {
  const props        = PropertiesService.getScriptProperties();
  const processedRaw = props.getProperty(CONFIG.PROCESSED_KEY) || "[]";
  let processed      = JSON.parse(processedRaw);

  // Get the transcript_summarizer folder
  let folder;
  try {
    folder = DriveApp.getFolderById(CONFIG.DRIVE_FOLDER_ID);
  } catch (e) {
    Logger.log("ERROR: Could not access Drive folder: " + e.message);
    Logger.log("Check DRIVE_FOLDER_ID in CONFIG.");
    return;
  }

  // List all .txt files in the folder
  const files = folder.getFilesByType(MimeType.PLAIN_TEXT);
  const newFiles = [];

  while (files.hasNext()) {
    const file   = files.next();
    const fileId = file.getId();

    // Skip already processed files
    if (processed.includes(fileId)) {
      continue;
    }

    // Only process files modified in the last 24 hours
    // (safety net in case the script was down for a while)
    const modTime = file.getLastUpdated();
    const ageMs   = Date.now() - modTime.getTime();
    const ageHrs  = ageMs / (1000 * 60 * 60);

    if (ageHrs > 24) {
      // Mark old unprocessed files as processed so we don't retry forever
      processed.push(fileId);
      continue;
    }

    newFiles.push({ id: fileId, name: file.getName() });
  }

  Logger.log("New files found: " + newFiles.length);

  // Process each new file
  for (const fileInfo of newFiles) {
    Logger.log("Processing: " + fileInfo.name + " (id=" + fileInfo.id + ")");

    const success = triggerLambda(fileInfo.id, fileInfo.name);

    if (success) {
      processed.push(fileInfo.id);
      Logger.log("Triggered Lambda for: " + fileInfo.name);
    } else {
      Logger.log("Failed to trigger Lambda for: " + fileInfo.name + " - will retry next poll");
    }
  }

  // Trim processed list and save
  if (processed.length > CONFIG.MAX_PROCESSED) {
    processed = processed.slice(processed.length - CONFIG.MAX_PROCESSED);
  }
  props.setProperty(CONFIG.PROCESSED_KEY, JSON.stringify(processed));
}

// ─────────────────────────────────────────────
// LAMBDA TRIGGER
// ─────────────────────────────────────────────
function triggerLambda(fileId, fileName) {
  const payload = {
    type:      "new_transcript",
    file_id:   fileId,
    file_name: fileName,
    secret:    CONFIG.WEBHOOK_SECRET,
    timestamp: new Date().toISOString(),
  };

  const options = {
    method:             "POST",
    contentType:        "application/json",
    payload:            JSON.stringify(payload),
    muteHttpExceptions: true,
    followRedirects:    true,
  };

  try {
    const response = UrlFetchApp.fetch(CONFIG.API_GATEWAY_URL, options);
    const code     = response.getResponseCode();

    if (code === 200) {
      return true;
    } else {
      Logger.log("Lambda returned status " + code + ": " + response.getContentText());
      return false;
    }
  } catch (e) {
    Logger.log("HTTP error calling Lambda: " + e.message);
    return false;
  }
}

// ─────────────────────────────────────────────
// SETUP TRIGGER
// Run this once manually after deploying the script
// ─────────────────────────────────────────────
function setupTrigger() {
  // Remove any existing triggers for this function
  const existingTriggers = ScriptApp.getProjectTriggers();
  for (const trigger of existingTriggers) {
    if (trigger.getHandlerFunction() === "checkForNewTranscripts") {
      ScriptApp.deleteTrigger(trigger);
      Logger.log("Removed existing trigger.");
    }
  }

  // Create new time-based trigger
  ScriptApp.newTrigger("checkForNewTranscripts")
    .timeBased()
    .everyMinutes(CONFIG.POLL_INTERVAL_MINUTES)
    .create();

  Logger.log(
    "Trigger created: checkForNewTranscripts runs every " +
    CONFIG.POLL_INTERVAL_MINUTES + " minutes."
  );
}

// ─────────────────────────────────────────────
// RESET
// Clears the processed file list (run if you want to reprocess all files)
// ─────────────────────────────────────────────
function resetProcessedFiles() {
  PropertiesService.getScriptProperties().deleteProperty(CONFIG.PROCESSED_KEY);
  Logger.log("Processed file list cleared.");
}

// ─────────────────────────────────────────────
// TEST HELPER
// Run this manually to test the full pipeline with a specific file
// ─────────────────────────────────────────────
function testWithFile() {
  // Paste a specific file ID here to test manually
  const TEST_FILE_ID   = "YOUR_TEST_FILE_ID_HERE";
  const TEST_FILE_NAME = "test_transcript.txt";

  Logger.log("Manual test: triggering Lambda for " + TEST_FILE_NAME);
  const success = triggerLambda(TEST_FILE_ID, TEST_FILE_NAME);
  Logger.log("Result: " + (success ? "SUCCESS" : "FAILED"));
}

// ─────────────────────────────────────────────
// STATUS CHECK
// Run manually to see current state
// ─────────────────────────────────────────────
function showStatus() {
  const props        = PropertiesService.getScriptProperties();
  const processedRaw = props.getProperty(CONFIG.PROCESSED_KEY) || "[]";
  const processed    = JSON.parse(processedRaw);

  Logger.log("=== Meeting Summarizer Drive Monitor ===");
  Logger.log("Folder ID    : " + CONFIG.DRIVE_FOLDER_ID);
  Logger.log("API Gateway  : " + CONFIG.API_GATEWAY_URL);
  Logger.log("Poll interval: every " + CONFIG.POLL_INTERVAL_MINUTES + " minutes");
  Logger.log("Files tracked: " + processed.length);

  // Show active triggers
  const triggers = ScriptApp.getProjectTriggers();
  Logger.log("Active triggers: " + triggers.length);
  for (const t of triggers) {
    Logger.log("  - " + t.getHandlerFunction() + " | " + t.getEventType());
  }
}
