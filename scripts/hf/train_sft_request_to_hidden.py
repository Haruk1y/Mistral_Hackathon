#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=3.0.0",
#   "peft>=0.13.0",
#   "transformers>=4.46.0",
#   "trl>=0.12.0",
#   "wandb>=0.18.0",
#   "trackio>=0.1.0",
# ]
# ///

"""SFT for request_text -> hidden_params on Ministral 3B using HF Jobs.

Expected dataset row:
{
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"request_text=..."},
    {"role":"assistant","content":"{\"energy\":...}"}
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=env_str("HF_BASE_MODEL_ID", "mistralai/Ministral-3-3B-Instruct-2512"))
    parser.add_argument("--dataset-repo", default=os.getenv("HF_FT_DATASET_REPO_ID"))
    parser.add_argument("--dataset-config", default=os.getenv("HF_FT_DATASET_CONFIG"))
    parser.add_argument("--train-split", default=env_str("HF_FT_TRAIN_SPLIT", "train"))
    parser.add_argument("--valid-split", default=env_str("HF_FT_VALID_SPLIT", "validation"))
    parser.add_argument("--local-train", default=os.getenv("HF_FT_LOCAL_TRAIN"))
    parser.add_argument("--local-valid", default=os.getenv("HF_FT_LOCAL_VALID"))
    parser.add_argument(
        "--hub-model-id",
        default=env_str("HF_FT_OUTPUT_MODEL_ID", "mistral-hackaton-2026/atelier-kotone-ministral3b-ft"),
    )
    parser.add_argument("--output-dir", default=env_str("HF_FT_OUTPUT_DIR", "outputs/ministral3b-request-hidden"))

    parser.add_argument("--epochs", type=int, default=env_int("HF_FT_EPOCHS", 2))
    parser.add_argument("--learning-rate", type=float, default=env_float("HF_FT_LR", 2e-5))
    parser.add_argument("--batch-size", type=int, default=env_int("HF_FT_BATCH_SIZE", 2))
    parser.add_argument("--grad-accum", type=int, default=env_int("HF_FT_GRAD_ACCUM", 8))
    parser.add_argument("--warmup-ratio", type=float, default=env_float("HF_FT_WARMUP_RATIO", 0.1))
    parser.add_argument("--max-length", type=int, default=env_int("HF_FT_MAX_LENGTH", 768))

    parser.add_argument("--lora-r", type=int, default=env_int("HF_FT_LORA_R", 16))
    parser.add_argument("--lora-alpha", type=int, default=env_int("HF_FT_LORA_ALPHA", 32))
    parser.add_argument("--lora-dropout", type=float, default=env_float("HF_FT_LORA_DROPOUT", 0.05))

    parser.add_argument("--run-name", default=env_str("HF_FT_RUN_NAME", "ministral3b-request-hidden-sft"))
    parser.add_argument("--wandb-project", default=env_str("WANDB_PROJECT", "atelier-kotone-ft"))
    parser.add_argument("--trackio-project", default=env_str("TRACKIO_PROJECT", "atelier-kotone-ft"))
    parser.add_argument("--trackio-space-id", default=os.getenv("TRACKIO_SPACE_ID"))
    parser.add_argument("--enable-trackio", action="store_true", default=env_bool("ENABLE_TRACKIO", False))

    return parser.parse_args()


def load_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.dataset_repo:
        dataset = load_dataset(args.dataset_repo, args.dataset_config) if args.dataset_config else load_dataset(args.dataset_repo)
        if args.train_split not in dataset:
            raise ValueError(f"train split '{args.train_split}' not found in dataset repo")
        if args.valid_split not in dataset:
            # fallback: split train internally
            split = dataset[args.train_split].train_test_split(test_size=0.1, seed=42)
            return split["train"], split["test"]
        return dataset[args.train_split], dataset[args.valid_split]

    if args.local_train:
        data_files: dict[str, str] = {"train": args.local_train}
        if args.local_valid:
            data_files["validation"] = args.local_valid
        ds = load_dataset("json", data_files=data_files)
        if "validation" not in ds:
            split = ds["train"].train_test_split(test_size=0.1, seed=42)
            return split["train"], split["test"]
        return ds["train"], ds["validation"]

    raise ValueError("Specify --dataset-repo or --local-train")


def normalize_messages(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if isinstance(messages, list):
        return {"messages": messages}

    request_text = str(row.get("request_text", "")).strip()
    target = row.get("target_hidden_params", {}).get("vector", {})
    target_json = json.dumps(target, ensure_ascii=False)
    return {
        "messages": [
            {
                "role": "system",
                "content": "Return strict JSON only with keys: energy,warmth,brightness,acousticness,complexity,nostalgia",
            },
            {"role": "user", "content": f"request_text={request_text}"},
            {"role": "assistant", "content": target_json},
        ]
    }


def build_text_dataset(dataset: Dataset, tokenizer: Any, split_name: str) -> Dataset:
    normalized = dataset.map(normalize_messages, remove_columns=dataset.column_names, desc=f"normalize:{split_name}")

    def to_text(sample: dict[str, Any]) -> dict[str, Any]:
        rendered = tokenizer.apply_chat_template(
            sample["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": rendered}

    return normalized.map(to_text, remove_columns=normalized.column_names, desc=f"chat_template:{split_name}")


def setup_wandb(args: argparse.Namespace) -> bool:
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        return False

    import wandb

    wandb.login(key=api_key, relogin=True)
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "base_model": args.base_model,
            "hub_model_id": args.hub_model_id,
            "dataset_repo": args.dataset_repo,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        tags=["hackathon", "hf-jobs", "ministral-3b", "request-to-hidden"],
    )
    return True


def maybe_setup_trackio(args: argparse.Namespace) -> bool:
    if not args.enable_trackio:
        return False

    import trackio

    trackio.init(
        project=args.trackio_project,
        name=args.run_name,
        space_id=args.trackio_space_id,
        config={
            "base_model": args.base_model,
            "hub_model_id": args.hub_model_id,
            "dataset_repo": args.dataset_repo,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        },
    )
    return True


def main() -> None:
    args = parse_args()

    train_ds, valid_ds = load_splits(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_text = build_text_dataset(train_ds, tokenizer, "train")
    valid_text = build_text_dataset(valid_ds, tokenizer, "valid")

    has_wandb = setup_wandb(args)
    has_trackio = maybe_setup_trackio(args)

    report_to: list[str] = []
    if has_wandb:
        report_to.append("wandb")
    if has_trackio:
        report_to.append("trackio")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_strategy="end",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        max_length=args.max_length,
        bf16=True,
        report_to=report_to,
    )

    trainer = SFTTrainer(
        model=args.base_model,
        args=config,
        peft_config=peft_config,
        train_dataset=train_text,
        eval_dataset=valid_text,
        dataset_text_field="text",
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    trainer.push_to_hub()

    metrics_dir = Path(args.output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "final_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train": train_result.metrics,
                "eval": eval_result,
                "train_examples": len(train_text),
                "valid_examples": len(valid_text),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if has_wandb:
        import wandb

        wandb.log(
            {
                "eval/final_loss": eval_result.get("eval_loss"),
                "dataset/train_examples": len(train_text),
                "dataset/valid_examples": len(valid_text),
            }
        )
        artifact = wandb.Artifact("final_metrics", type="metrics")
        artifact.add_file(str(metrics_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    if has_trackio:
        import trackio

        trackio.finish()

    print(f"Training complete. Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
