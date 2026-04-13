# RLHF / DPO Handover

## Current State of Work
As of April 12, the Multi-Agent autonomous assistant architecture has been fully refactored, containerized, and deployed end-to-end.

**Victories Achieved:**
1. **Structural Refactor**: All core logic previously hidden in `modular/` and `AWSOrchestration/` has been safely relocated to `src/`.
2. **Stateless Scale**: Slack's payload limitations have been circumvented completely using an Upstash Redis database (`redis_store.py`) to manage transient state.
3. **Telemetry & RLHF Foundations**:
   - `ChromaDB` was installed locally alongside your EC2 `vLLM` engine.
   - The Lambda function flawlessly records all human interactivity triggers (Slack approves/cancels) to `ChromaDB` synchronously.
   - AWS Lambda Read-Only File System constraints evaluating `onnxruntime` bindings are permanently resolved via Dockerfile cache mounts.

## Next Steps: DPO Fine-tuning Pipeline
The telemetry is now actively streaming into ChromaDB. Once you have a sufficient volume of real human interaction (e.g. 50+ recorded decisions), the final capstone goal is to execute the **Direct Preference Optimization (DPO)** pipeline.

### Step 1: Generate Preference Pairs
We need to group the recorded "Cancelled" workflows with similar "Approved" ones from ChromaDB to create DPO triplets `(Prompt, Chosen, Rejected)`.
- **Command:** `python3 rlhf/build_preference_dataset.py --flow slack`
- **Output:** This script scans ChromaDB, isolates `<rejected>` interactions, calculates the closest semantic `<approved>` matches, and merges them into `rlhf/datasets/slack_train.jsonl`.

### Step 2: DPO Training (HuggingFace TRL)
Start the LoRA fine-tuning run on your EC2 instance GPU using the generated dataset.
- **Command:** `python3 rlhf/train_dpo.py --config rlhf/train_config.yaml`
- **Action:** This will pull the base Qwen 14B model, load your existing adapters, apply DPO loss using the preference pairs, and save a specialized adapter to `rlhf/adapters/slack_dpo/`.

### Step 3: Evaluation & Hot-Swap Deployment
Before routing traffic to the new model, ensure it didn't catastrophically forget task instructions.
- **Eval:** `python3 rlhf/evaluate.py --old lora_adapters/slack --new rlhf/adapters/slack_dpo`
- **Deploy:** `bash rlhf/deploy_adapters.sh slack rlhf/adapters/slack_dpo/` (This script hot-swaps the vLLM LoRA adapter without restarting the massive model).
