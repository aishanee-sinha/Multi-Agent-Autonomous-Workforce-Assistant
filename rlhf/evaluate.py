"""
evaluate.py — Before/after comparison for DPO-trained adapters
===============================================================
Runs the same test prompts through old and new LoRA adapters,
compares outputs, and produces a report. Blocks deployment if
regression exceeds a threshold.

Usage:
  python rlhf/evaluate.py --flow slack --old-adapter lora_adapters/slack --new-adapter rlhf/adapters/slack_dpo
"""

import argparse
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Test prompts per flow
# ─────────────────────────────────────────────────────────────────────────────
EVAL_PROMPTS = {
    "slack": [
        "Set up CI/CD pipeline for the new repo, assign to Soham",
        "Fix the login bug on the dashboard, assign to Aishanee",
        "hey, what's for lunch today?",
        "Can someone review PR #42 for the auth service? @ketki should take a look",
        "Deploy v2.1 to staging environment before Friday, assigning this to Soham",
    ],
    "email": [
        "Subject: Team sync next Monday\nFrom: soham@example.com\nTo: team@example.com\n\nCan we schedule a sync for next Monday morning?",
        "Subject: Q3 Budget Report\nFrom: finance@example.com\nTo: team@example.com\n\nPlease find attached the Q3 budget report.",
        "Subject: Design review Thursday\nFrom: aishanee@example.com\nTo: soham@example.com\n\nCan we meet Thursday afternoon around 2pm?",
    ],
    "meeting": [
        json.dumps({"transcript_preview": "Soham: Deploy hotfix by tonight. Aishanee: I'll handle rollback."}),
        json.dumps({"transcript_preview": "Ketki: Let's review sprint backlog. 12 stories left."}),
    ],
}


def evaluate_adapter(
    flow: str,
    base_model: str,
    old_adapter_path: str,
    new_adapter_path: str,
    max_regression_pct: float = 20.0,
) -> dict:
    """
    Compare old vs new adapter outputs on test prompts.

    Returns a report dict with:
      - per-prompt comparison
      - overall scores
      - pass/fail decision
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    prompts = EVAL_PROMPTS.get(flow, [])
    if not prompts:
        logger.error("No eval prompts for flow '%s'", flow)
        return {"error": f"No eval prompts for flow {flow}"}

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = {"flow": flow, "prompts": [], "old_adapter": old_adapter_path, "new_adapter": new_adapter_path}

    for adapter_label, adapter_path in [("old", old_adapter_path), ("new", new_adapter_path)]:
        logger.info("Loading %s adapter from %s", adapter_label, adapter_path)

        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )

        if adapter_path and os.path.exists(adapter_path):
            model = PeftModel.from_pretrained(base, adapter_path)
        else:
            model = base
            logger.warning("Adapter path not found: %s — using base model", adapter_path)

        model.eval()

        for i, prompt_text in enumerate(prompts):
            if len(report["prompts"]) <= i:
                report["prompts"].append({"prompt": prompt_text, "old_output": "", "new_output": ""})

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=256, temperature=0.1,
                    do_sample=True, pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            report["prompts"][i][f"{adapter_label}_output"] = response.strip()

        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Score outputs
    improved = same = regressed = 0
    for entry in report["prompts"]:
        old_score = _score_output(flow, entry["prompt"], entry["old_output"])
        new_score = _score_output(flow, entry["prompt"], entry["new_output"])
        entry["old_score"] = old_score
        entry["new_score"] = new_score

        if new_score > old_score:
            improved += 1
            entry["verdict"] = "improved"
        elif new_score < old_score:
            regressed += 1
            entry["verdict"] = "regressed"
        else:
            same += 1
            entry["verdict"] = "same"

    total = len(report["prompts"])
    regression_pct = (regressed / total * 100) if total > 0 else 0

    report["summary"] = {
        "improved": improved,
        "same": same,
        "regressed": regressed,
        "total": total,
        "regression_pct": round(regression_pct, 1),
        "pass": regression_pct <= max_regression_pct,
    }

    return report


def _score_output(flow: str, prompt: str, output: str) -> int:
    """
    Heuristic scoring of an output (0-10 scale).
    Higher = better quality.
    """
    score = 0

    # 1. Valid JSON? (+3)
    try:
        parsed = json.loads(output)
        score += 3
    except (json.JSONDecodeError, TypeError):
        return score  # If not valid JSON, score stays 0

    if flow == "slack":
        # Has task_summary? (+2)
        if parsed.get("task_summary") and len(parsed["task_summary"]) > 10:
            score += 2
        # Has assignee (not "Unassigned")? (+2)
        if parsed.get("assignee") and parsed["assignee"] != "Unassigned":
            score += 2
        # Correct no_action for non-tasks? (+3)
        if "lunch" in prompt.lower() or "morning" in prompt.lower():
            if parsed.get("no_action") is True:
                score += 3
        else:
            if parsed.get("no_action") is False:
                score += 3

    elif flow == "email":
        # Correct is_meeting? (+3)
        has_meeting_words = any(w in prompt.lower() for w in ["sync", "meet", "schedule", "call"])
        if parsed.get("is_meeting") == has_meeting_words:
            score += 3
        # Has title when meeting? (+2)
        if parsed.get("is_meeting") and parsed.get("title") and len(parsed["title"]) > 5:
            score += 2
        # Has attendees when meeting? (+2)
        if parsed.get("is_meeting") and parsed.get("attendees"):
            score += 2

    elif flow == "meeting":
        # Has abstract? (+3)
        if parsed.get("abstract") and len(parsed["abstract"]) > 20:
            score += 3
        # Has actions? (+2)
        if parsed.get("n_actions", 0) > 0:
            score += 2
        # Has decisions? (+2)
        if parsed.get("decisions") and len(parsed["decisions"]) > 10:
            score += 2

    return score


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate DPO-trained adapters")
    parser.add_argument("--flow", choices=["slack", "email", "meeting"], required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    parser.add_argument("--old-adapter", required=True, help="Path to old LoRA adapter")
    parser.add_argument("--new-adapter", required=True, help="Path to new DPO adapter")
    parser.add_argument("--max-regression", type=float, default=20.0, help="Max regression % to pass")
    parser.add_argument("--output", default=None, help="Save report JSON to this path")
    args = parser.parse_args()

    report = evaluate_adapter(
        flow=args.flow,
        base_model=args.base_model,
        old_adapter_path=args.old_adapter,
        new_adapter_path=args.new_adapter,
        max_regression_pct=args.max_regression,
    )

    # Print summary
    s = report.get("summary", {})
    print("\n" + "=" * 50)
    print(f"Evaluation Report: {args.flow}")
    print("=" * 50)
    print(f"  Improved:  {s.get('improved',  0)}/{s.get('total', 0)}")
    print(f"  Same:      {s.get('same',      0)}/{s.get('total', 0)}")
    print(f"  Regressed: {s.get('regressed', 0)}/{s.get('total', 0)}")
    print(f"  Regression: {s.get('regression_pct', 0)}%")
    print(f"  PASS: {'✅ YES' if s.get('pass') else '❌ NO — DEPLOYMENT BLOCKED'}")
    print("=" * 50)

    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")

    sys.exit(0 if s.get("pass") else 1)


if __name__ == "__main__":
    main()
