import json
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


from langchain_core.messages import HumanMessage, SystemMessage
from calendar_agent import EMAIL_SYSTEM_PROMPT, EmailMeetingDetails
from state import _llm


def _build_email_text(email_data: dict) -> str:
	return (
		f"Subject: {email_data.get('subject', '')}\n"
		f"From: {email_data.get('from_email', '')}\n"
		f"To: {', '.join(email_data.get('to_emails', []))}\n"
		f"Cc: {', '.join(email_data.get('cc_emails', []))}\n\n"
		f"{email_data.get('body', '')[:2000]}"
	)


def _invoke_email_llm(email_data: dict) -> tuple[str, EmailMeetingDetails, dict]:
	llm = _llm(structured_output=EmailMeetingDetails, model_name="Qwen/Qwen2.5-14B-Instruct-AWQ")
	email_text = _build_email_text(email_data)
	model = llm.invoke([
		SystemMessage(content=EMAIL_SYSTEM_PROMPT),
		HumanMessage(content=email_text),
	])
	raw = model.model_dump_json(indent=2)
	parsed = model.model_dump(mode="json")
	return raw, model, parsed


def _fake_email_cases() -> list[tuple[str, dict]]:
	return [
		(
			"meeting_email",
			{
				"subject": "Q2 Planning Meeting",
				"from_email": "manager@example.com",
				"to_emails": ["alice@example.com", "bob@example.com"],
				"cc_emails": ["pm@example.com"],
				"body": (
					"Hi team, let's schedule a planning meeting tomorrow at 3:00 PM UTC "
					"for 45 minutes on Google Meet."
				),
			},
		),
		(
			"non_meeting_email",
			{
				"subject": "Weekly Report",
				"from_email": "ops@example.com",
				"to_emails": ["team@example.com"],
				"cc_emails": [],
				"body": "Please find the weekly KPI report attached. No action needed.",
			},
		),
	]


def run_local_email_llm_demo() -> None:
	if not os.environ.get("EC2_IP"):
		print("EC2_IP is not set. Export EC2_IP before running this script.")
		return

	for name, email_data in _fake_email_cases():
		print(f"\n--- {name} ---")
		try:
			raw, model, parsed = _invoke_email_llm(email_data)
			print("Raw LLM response:")
			print(raw)
			print("Parsed JSON:")
			print(json.dumps(parsed, indent=2, default=str))
			print("Model validation: OK")
			print(f"is_meeting={model.is_meeting}, time_confidence={model.time_confidence}")
		except Exception as exc:
			print(f"Failed to process case '{name}': {exc}")


if __name__ == "__main__":
	run_local_email_llm_demo()
