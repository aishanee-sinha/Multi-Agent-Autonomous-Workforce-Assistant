"""
train_dpo.py — DPO fine-tuning for LoRA adapters
==================================================
Uses Hugging Face trl.DPOTrainer + peft LoRA to retrain the
slack/email/meeting adapters on preference pairs.

Usage:
  python rlhf/train_dpo.py --flow slack
  python rlhf/train_dpo.py --flow email --dataset rlhf/datasets/email_pairs.jsonl
  python rlhf/train_dpo.py --flow all

Requirements (install via rlhf/requirements_train.txt):
  trl, peft, transformers, datasets, accelerate, bitsandbytes
"""

import argparse
import json
import logging
import os
import sys

import yaml
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default config (overridden by train_config.yaml)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_CONFIG = {
    "beta": 0.1,
    "lr": 5e-7,
    "epochs": 2,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_length": 512,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def load_config(config_path: str = "rlhf/train_config.yaml") -> dict:
    """Load training config from YAML file."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_preference_dataset(dataset_path: str) -> Dataset:
    """Load JSONL preference pairs into a HuggingFace Dataset."""
    records = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append({
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            })

    if not records:
        raise ValueError(f"No records found in {dataset_path}")

    logger.info("Loaded %d preference pairs from %s", len(records), dataset_path)
    return Dataset.from_list(records)


def train_dpo(
    flow: str,
    dataset_path: str,
    output_dir: str = None,
    base_model: str = DEFAULT_BASE_MODEL,
    existing_lora_path: str = None,
    config: dict = None,
) -> str:
    """
    Run DPO fine-tuning for a given flow's LoRA adapter.

    Parameters
    ----------
    flow : str
        One of "slack", "email", "meeting".
    dataset_path : str
        Path to JSONL file with {prompt, chosen, rejected} records.
    output_dir : str
        Where to save the new adapter. Defaults to rlhf/adapters/{flow}_dpo.
    base_model : str
        HuggingFace model ID for the base model.
    existing_lora_path : str
        Path to existing LoRA adapter to continue training from.
    config : dict
        Training hyperparameters.

    Returns
    -------
    str
        Path to the saved adapter.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    output_dir = output_dir or f"rlhf/adapters/{flow}_dpo"

    logger.info("=" * 60)
    logger.info("DPO Training: flow=%s", flow)
    logger.info("  Base model: %s", base_model)
    logger.info("  Dataset:    %s", dataset_path)
    logger.info("  Output:     %s", output_dir)
    logger.info("  Config:     %s", json.dumps(cfg, indent=2))
    logger.info("=" * 60)

    # 1. Load dataset
    dataset = load_preference_dataset(dataset_path)

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load base model
    logger.info("Loading base model: %s", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 4. If existing LoRA adapter exists, load it as starting point
    if existing_lora_path and os.path.exists(existing_lora_path):
        logger.info("Loading existing LoRA adapter from: %s", existing_lora_path)
        model = PeftModel.from_pretrained(model, existing_lora_path)
        model = model.merge_and_unload()  # Merge into base for DPO re-training
        logger.info("Merged existing adapter into base model")

    # 5. Configure new LoRA for DPO
    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg["lora_target_modules"],
    )

    # 6. Configure DPO training
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=cfg["beta"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        max_length=cfg["max_length"],
        remove_unused_columns=False,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )

    # 7. Initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # With LoRA, ref_model is implicit (base weights)
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=dpo_config,
        peft_config=peft_config,
    )

    # 8. Train
    logger.info("Starting DPO training...")
    trainer.train()

    # 9. Save adapter
    logger.info("Saving adapter to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("DPO training complete for flow='%s'!", flow)
    return output_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="DPO fine-tuning for LoRA adapters")
    parser.add_argument("--flow", choices=["slack", "email", "meeting", "all"], required=True)
    parser.add_argument("--dataset", help="Path to preference pairs JSONL")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--existing-lora", help="Path to existing LoRA adapter")
    parser.add_argument("--output-dir", help="Output directory for new adapter")
    parser.add_argument("--config", default="rlhf/train_config.yaml", help="Training config YAML")
    args = parser.parse_args()

    all_config = load_config(args.config)
    flows = ["slack", "email", "meeting"] if args.flow == "all" else [args.flow]

    for flow in flows:
        flow_config = {**DEFAULT_CONFIG, **all_config.get(flow, {})}
        dataset_path = args.dataset or flow_config.get(
            "dataset_path", f"rlhf/datasets/{flow}_seed.jsonl"
        )
        existing_lora = args.existing_lora or flow_config.get("lora_path")
        output_dir = args.output_dir or f"rlhf/adapters/{flow}_dpo"

        if not os.path.exists(dataset_path):
            logger.error("Dataset not found: %s — skipping flow '%s'", dataset_path, flow)
            continue

        train_dpo(
            flow=flow,
            dataset_path=dataset_path,
            output_dir=output_dir,
            base_model=args.base_model,
            existing_lora_path=existing_lora,
            config=flow_config,
        )


if __name__ == "__main__":
    main()
