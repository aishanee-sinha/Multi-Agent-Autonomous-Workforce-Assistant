
## 1. Elevator Pitch

"I built an event-driven workplace automation agent using LangGraph and a self-hosted LLM. It listens to two real-world event streams — Slack messages and incoming Gmail — and takes autonomous action on them. When someone mentions a task in Slack, it extracts a Jira ticket and asks for human approval before creating it. When an email arrives that looks like a meeting request, it parses the details and proposes a Google Calendar event, again waiting for human confirmation via Slack. The whole system runs as a single AWS Lambda function, uses a LangGraph orchestrator to route between two specialized subgraphs, and is designed around a human-in-the-loop pattern so no action is ever taken without explicit approval."

---

## 2. Detailed Component Walkthrough

**Infrastructure & Entry Point**

The system runs entirely inside a single AWS Lambda function. All incoming events — Slack messages, Slack button clicks, and Gmail Pub/Sub push notifications — arrive as HTTP POST requests through API Gateway. This is a deliberate design choice: one endpoint, one handler, one deployed artifact.

**State — `OrchestratorState`**

Everything flows through a single Pydantic model called `OrchestratorState`. It holds fields for both the Slack/Jira flow and the email/calendar flow simultaneously. This shared state is what allows a single LangGraph graph to serve both pipelines without duplication. LangGraph passes this state object from node to node, and each node returns a shallow copy with only the fields it modified, using Pydantic's `model_copy(update={...})`.

**`parse_input` — Event Classification**

This is the first node every event hits. It does no LLM calls — it's pure deterministic parsing. It inspects the raw HTTP body and headers to figure out what kind of event arrived:
- A `payload=` URL-encoded body means a Slack button press (interactivity)
- A `x-slack-retry-num` header means Slack is retrying a timed-out request — these are immediately short-circuited to avoid double-processing
- A body with `message.data` base64 field means a Gmail Pub/Sub push
- A body with `event` key means a Slack message event
- A body with `from_email` or `subject` means a direct test invocation

It sets `intent` to `slack`, `email`, or `none` — or leaves it as `unknown` for raw Slack messages that need the LLM router to decide.

**`router_agent` — LLM Intent Routing**

Only invoked when `intent` is still `unknown`, which happens exclusively for raw Slack message events. It calls the self-hosted Qwen model with a structured output schema (`RouterDecision`) to classify whether the message is a Jira task request, a meeting-related message, or neither. All other event types have their intent set deterministically by `parse_input`, so the LLM is never called unnecessarily.

**Slack Subgraph — Jira Ticket Flow**

This runs in two separate Lambda invocations:

First invocation (new Slack message):
- `slack_extract_ticket` — calls the LLM with a structured output schema (`JiraTicket`) to extract a task summary and assignee from the message text. Sets `no_action=True` if no actionable task is found.
- `slack_post_preview` — posts a Slack card with the proposed ticket details and two buttons: "Create Ticket" and "Cancel". The ticket summary and assignee are embedded in the button's `value` JSON payload so no database is needed to remember state between invocations.

Second invocation (button press):
- `slack_resolve_assignee` — maps the raw assignee name or Slack user ID to a Jira account ID using a `TEAM_MAP` dictionary loaded from an environment variable.
- `slack_create_jira` — calls the Jira REST API v3 to create the issue, with full error logging including permission diagnostics.
- `slack_post_result` — updates the original Slack preview card with the Jira ticket link or an error message.

**Calendar Subgraph — Email to Meeting Flow**

Also runs in two separate Lambda invocations:

First invocation (Gmail Pub/Sub push):
- `email_fetch_and_parse` — uses the Gmail API with the history ID from the Pub/Sub message to fetch the actual new email. Parses headers, extracts sender, recipients, CC, and body.
- `email_classify` — calls the LLM with a detailed system prompt to extract structured meeting details: title, start time, end time, location, attendees, and a confidence score. Filters by a `GROUP_EMAILS` allowlist so only emails from known senders trigger the flow.
- `email_post_slack_preview` — posts a Slack card to a configured notify channel with all meeting details and Create/Cancel buttons. The full pending meeting data is serialized into the button's `value` field — again avoiding any database dependency.

Second invocation (button press):
- `email_create_calendar` — reads `pending_meeting` from state (which `parse_input` deserialized from the button value), calls the Google Calendar API to create the event, sends invites to all attendees, and updates the Slack card with the calendar link.

**Self-hosted LLM**

The LLM used throughout is Qwen, running on an EC2 instance behind a vLLM-compatible OpenAI API endpoint. The `_llm()` factory in `state.py` points `ChatOpenAI` at `http://{EC2_IP}:8000/v1`. Structured outputs use LangChain's `.with_structured_output()` wrapper so node code receives typed Pydantic objects rather than raw strings.

**Stateless Design — No Database**

A key architectural decision is that there is no persistent storage between the two Lambda invocations. All state needed for the second invocation (the button press) is embedded inside the Slack button's `value` JSON field. Slack stores it, and sends it back when the button is clicked. This keeps the system fully stateless and eliminates a DynamoDB or Redis dependency entirely.

---

## 3. Challenges Faced

**Slack's 3-second timeout.** Slack requires an HTTP 200 response within 3 seconds of sending an event, but LLM inference takes longer than that. The solution is to return 200 immediately from Lambda, let the graph run asynchronously, and use Slack's `chat_postMessage` / `chat_update` to communicate results after the fact. Slack retry handling (`x-slack-retry-num`) had to be added explicitly to prevent the same message being processed multiple times.

**Stateless two-phase flows.** The human-in-the-loop pattern means the flow is split across two completely separate Lambda invocations with no shared memory. All intermediate state — the proposed ticket summary, the parsed meeting details — had to be serialized into the Slack button value. This works but has a size limit (Slack button values are capped at 2000 characters), which required careful trimming of email bodies passed through.

**LLM structured output reliability.** Getting the LLM to consistently return valid JSON in the exact schema required — especially for the email classifier — required careful prompt engineering and defensive parsing (stripping markdown code fences, catching JSON decode errors gracefully).

**Jira API permission debugging.** The Jira REST API returns vague error messages for permission failures. Logging had to be built out specifically to detect permission-related error strings and surface them clearly, since the same credentials work for some projects but not others depending on how the Jira project is configured.

**Gmail Pub/Sub indirection.** Gmail doesn't push the email content — it pushes a history ID. A separate Gmail API call is then needed to fetch the actual new messages from that history ID. This introduces a race condition where the history could include multiple new messages, requiring the code to handle lists and pick the most recent.

---

## 4. Limitations & Future Improvements

**Limitations**

- The button value size cap (2000 chars) means very long emails can have their body truncated before being passed to the calendar creation step, potentially losing context.
- The `TEAM_MAP` for Jira assignee resolution is a static dictionary in an environment variable — it doesn't automatically stay in sync if team members change.
- The system has no retry or dead-letter mechanism if a Lambda invocation fails after Slack has already received the 200 OK. A failed Jira creation or calendar event is silently lost unless the user notices the error message in Slack.
- Time zone handling in the calendar flow defaults to UTC, which will create incorrectly-timed events for users in other time zones if the email doesn't specify one.
- There is no authentication check on the Slack interactivity payloads beyond what API Gateway provides — Slack request signing verification is not implemented.

**Future Improvements**

- Add Slack request signature verification to prevent spoofed button payloads.
- Replace the static `TEAM_MAP` with a live lookup against the Jira user search API or a Slack-to-Jira user mapping stored in DynamoDB.
- Add a DynamoDB-backed dead-letter queue for failed second-phase actions, with a background retry mechanism.
- Extend the router to handle more intent types — for example, PagerDuty alerts, expense approvals, or PR review reminders — each as its own subgraph.
- Move the self-hosted LLM behind an auto-scaling group so it doesn't become a single point of failure.
- Add streaming responses so the Slack card updates progressively as the LLM generates output, rather than appearing all at once after full inference completes.