#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "accelerate>=0.34.0",
#   "datasets>=3.0.0",
#   "peft>=0.13.0",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "mistral-common>=1.8.6",
#   "huggingface-hub>=0.34.0",
#   "wandb>=0.18.0",
#   "trackio>=0.1.0",
#   "weave>=0.51.0",
# ]
# ///

"""Headless next-token SFT for request_text + hidden_params pair JSON generation.

This script trains a causal LM objective (no regression head) so inference can
directly return strict JSON pair:
{
  "request_text": str,
  "hidden_params": {
    "energy": int, "warmth": int, "brightness": int,
    "acousticness": int, "complexity": int, "nostalgia": int
  }
}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")
WEATHER_VALUES = {"sunny", "cloudy", "rainy"}


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


def clamp_int(value: float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(round(float(value)))))


def convert_scale_int(value: float, source_scale: int, target_scale: int) -> int:
    src = max(1, int(source_scale))
    dst = max(1, int(target_scale))
    bounded = clamp_int(value, lo=0, hi=src)
    return clamp_int((bounded / src) * dst, lo=0, hi=dst)


def default_vector_for_scale(target_scale: int) -> dict[str, int]:
    hi = max(1, int(target_scale))
    midpoint = clamp_int(target_scale / 2.0, lo=0, hi=hi)
    return {key: midpoint for key in KEYS}


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
    parser.add_argument("--logging-steps", type=int, default=env_int("HF_FT_LOGGING_STEPS", 1))
    parser.add_argument("--eval-steps", type=int, default=env_int("HF_FT_EVAL_STEPS", 25))
    parser.add_argument("--detailed-eval-steps", type=int, default=env_int("HF_FT_DETAILED_EVAL_STEPS", 1))
    parser.add_argument("--max-steps", type=int, default=env_int("HF_FT_MAX_STEPS", -1))
    parser.add_argument("--hard-cases-top-k", type=int, default=env_int("HF_FT_HARD_CASE_TOP_K", 80))
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=env_int("HF_FT_GENERATION_MAX_NEW_TOKENS", 96),
    )
    parser.add_argument("--target-scale", type=int, default=env_int("HF_FT_TARGET_SCALE", env_int("FT_TARGET_SCALE", 10)))

    parser.add_argument("--lora-r", type=int, default=env_int("HF_FT_LORA_R", 16))
    parser.add_argument("--lora-alpha", type=int, default=env_int("HF_FT_LORA_ALPHA", 32))
    parser.add_argument("--lora-dropout", type=float, default=env_float("HF_FT_LORA_DROPOUT", 0.05))

    parser.add_argument("--run-name", default=env_str("HF_FT_RUN_NAME", "ministral3b-request-hidden-nexttoken"))
    parser.add_argument("--wandb-project", default=env_str("WANDB_PROJECT", "atelier-kotone-ft"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-group", default=os.getenv("WANDB_RUN_GROUP"))
    parser.add_argument("--cycle-id", default=env_str("LOOP_CYCLE_ID", "cycle_0"))
    parser.add_argument("--dataset-version", default=env_str("FT_DATASET_VERSION", "unknown"))
    parser.add_argument("--source-type-mix", default=env_str("FT_SOURCE_TYPE_MIX", "request_text+rule_prompt"))
    parser.add_argument("--weave-project", default=env_str("WEAVE_PROJECT", "atelier-kotone-weave"))
    parser.add_argument("--trackio-project", default=env_str("TRACKIO_PROJECT", "atelier-kotone-ft"))
    parser.add_argument("--trackio-space-id", default=os.getenv("TRACKIO_SPACE_ID"))
    parser.add_argument("--enable-trackio", action="store_true", default=env_bool("ENABLE_TRACKIO", False))
    parser.add_argument("--enable-weave-trace", action="store_true", default=env_bool("ENABLE_WEAVE_TRACE", True))
    parser.add_argument("--push-to-hub", action="store_true", default=env_bool("HF_FT_PUSH_TO_HUB", True))

    args = parser.parse_args()
    if args.target_scale != 10:
        raise ValueError(f"HF_FT_TARGET_SCALE/--target-scale must be 10 for this pipeline, got: {args.target_scale}")
    return args


def load_splits(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.dataset_repo:
        dataset = load_dataset(args.dataset_repo, args.dataset_config) if args.dataset_config else load_dataset(args.dataset_repo)
        if args.train_split in dataset:
            if args.valid_split not in dataset:
                split = dataset[args.train_split].train_test_split(test_size=0.1, seed=42)
                return split["train"], split["test"]
            return dataset[args.train_split], dataset[args.valid_split]

        data_files: dict[str, str] = {"train": f"hf://datasets/{args.dataset_repo}/{args.train_split}.jsonl"}
        data_files["validation"] = f"hf://datasets/{args.dataset_repo}/{args.valid_split}.jsonl"
        explicit = load_dataset("json", data_files=data_files)
        if "validation" not in explicit:
            split = explicit["train"].train_test_split(test_size=0.1, seed=42)
            return split["train"], split["test"]
        return explicit["train"], explicit["validation"]

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


def _unwrap_token_field(field: Any) -> list[int]:
    if isinstance(field, list) and field and isinstance(field[0], list):
        return [int(v) for v in field[0]]
    if isinstance(field, list):
        return [int(v) for v in field]
    return []


def _extract_request_text(row: dict[str, Any]) -> str:
    direct = str(row.get("request_text", "")).strip()
    if direct:
        return direct

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            if content.startswith("request_text="):
                tail = content.split("request_text=", 1)[1].strip()
                return tail.splitlines()[0].strip()
            if content:
                return content

    raise ValueError("request_text not found")


def _extract_source_type(row: dict[str, Any]) -> str:
    value = str(row.get("source_type", "")).strip().lower()
    if value:
        return value

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            if "source_type=" not in content:
                continue
            source_type = content.split("source_type=", 1)[1].splitlines()[0].strip().lower()
            if source_type:
                return source_type
    return "request_text"


def _extract_weather(row: dict[str, Any]) -> str:
    direct = str(row.get("weather", "")).strip().lower()
    if direct in WEATHER_VALUES:
        return direct

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") != "user":
                continue
            content = str(message.get("content", "")).strip()
            if "weather=" not in content:
                continue
            weather = content.split("weather=", 1)[1].split()[0].strip().lower().strip(",.")
            if weather in WEATHER_VALUES:
                return weather

    lower = str(row.get("request_text", "")).lower()
    if any(token in lower for token in ("rain", "storm", "drizzle", "wet")):
        return "rainy"
    if any(token in lower for token in ("night", "evening", "dusk", "cloud", "fog")):
        return "cloudy"
    if any(token in lower for token in ("sun", "morning", "bright", "daylight")):
        return "sunny"
    return "cloudy"


def _extract_target_vector(row: dict[str, Any]) -> dict[str, float]:
    vector = row.get("target_hidden_params", {}).get("vector")
    if isinstance(vector, dict) and all(key in vector for key in KEYS):
        return {key: float(vector[key]) for key in KEYS}

    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = str(message.get("content", "")).strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and all(key in parsed for key in KEYS):
                return {key: float(parsed[key]) for key in KEYS}

    raise ValueError("target hidden params not found")


def _infer_vector_scale(vector: dict[str, Any]) -> int:
    values: list[float] = []
    for key in KEYS:
        value = vector.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    max_value = max(values) if values else 0.0
    if max_value <= 10:
        return 10
    if max_value <= 100:
        return 100
    return 10


def _extract_target_scale(row: dict[str, Any], default_scale: int) -> int:
    candidates = [row.get("target_scale"), row.get("target_hidden_params", {}).get("target_scale")]
    for value in candidates:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    try:
        vector = _extract_target_vector(row)
        return _infer_vector_scale(vector)
    except Exception:  # noqa: BLE001
        return default_scale


def normalize_row(row: dict[str, Any], default_scale: int) -> dict[str, Any]:
    request_text = _extract_request_text(row)
    source_type = _extract_source_type(row)
    weather = _extract_weather(row)
    target = _extract_target_vector(row)
    target_scale = _extract_target_scale(row, default_scale)
    return {
        "request_text": request_text,
        "source_type": source_type,
        "weather": weather,
        "target": target,
        "target_scale": target_scale,
    }


def build_inference_prompt(request_text: str, weather: str, source_type: str, target_scale: int) -> str:
    return "\n".join(
        [
            "You estimate 6 hidden music parameters.",
            "Return strict JSON only with this schema:",
            '{"request_text":"...", "hidden_params":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
            "Copy request_text exactly from input.",
            f"Each hidden_params value must be integer between 0 and {target_scale}.",
            f"source_type={source_type}",
            f"weather={weather}",
            f"request_text={request_text}",
        ]
    )


def build_sft_dataset(dataset: Dataset, tokenizer: Any, split_name: str, max_length: int, target_scale: int) -> Dataset:
    normalized = dataset.map(
        lambda row: normalize_row(row, target_scale),
        remove_columns=dataset.column_names,
        desc=f"normalize:{split_name}",
    )

    def to_features(sample: dict[str, Any]) -> dict[str, Any]:
        sample_scale = max(1, int(sample.get("target_scale", target_scale)))
        target_vector = {
            key: convert_scale_int(float(sample["target"][key]), source_scale=sample_scale, target_scale=target_scale)
            for key in KEYS
        }
        prompt = build_inference_prompt(
            request_text=str(sample["request_text"]),
            weather=str(sample["weather"]),
            source_type=str(sample["source_type"]),
            target_scale=target_scale,
        )
        assistant_json = json.dumps(
            {
                "request_text": str(sample["request_text"]),
                "hidden_params": target_vector,
            },
            ensure_ascii=False,
        )
        eos = tokenizer.eos_token or ""
        full_text = f"{prompt}\n{assistant_json}{eos}"

        tokenized_full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )

        input_ids = _unwrap_token_field(tokenized_full["input_ids"])
        attention_mask = _unwrap_token_field(tokenized_full["attention_mask"])
        prompt_ids = _unwrap_token_field(tokenized_prompt["input_ids"])
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "request_text": sample["request_text"],
            "source_type": sample["source_type"],
            "weather": sample["weather"],
            "target_scale": target_scale,
            "target_raw": [float(target_vector[key]) for key in KEYS],
            "prompt_text": prompt,
        }

    return normalized.map(to_features, remove_columns=["target"], desc=f"tokenize:{split_name}")


class SFTDataCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in features]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        max_len = int(batch["input_ids"].shape[1])

        labels = []
        for item in features:
            row = list(item["labels"])
            if len(row) < max_len:
                row = row + [-100] * (max_len - len(row))
            labels.append(row[:max_len])

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class EvalGenerationCollator:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "prompts": [str(item["prompt_text"]) for item in features],
            "request_text": [str(item["request_text"]) for item in features],
            "source_type": [str(item.get("source_type", "request_text")) for item in features],
            "weather": [str(item["weather"]) for item in features],
            "target_raw": [list(item["target_raw"]) for item in features],
        }


class HiddenParamSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model.model.language_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            use_cache=False,
            return_dict=True,
        )
        loss = outputs.loss
        if return_outputs:
            return loss, outputs
        return loss


def _format_regression_metrics(preds_raw: torch.Tensor, labels_raw: torch.Tensor, target_scale: int) -> dict[str, float]:
    denom = float(max(1, target_scale))
    preds_norm = preds_raw / denom
    labels_norm = labels_raw / denom

    abs_norm = torch.abs(preds_norm - labels_norm)
    sq_norm = (preds_norm - labels_norm) ** 2
    abs_raw = torch.abs(preds_raw - labels_raw)
    sq_raw = (preds_raw - labels_raw) ** 2

    metrics: dict[str, float] = {
        "mse_norm": torch.mean(sq_norm).item(),
        "mae_norm": torch.mean(abs_norm).item(),
        "mse_raw": torch.mean(sq_raw).item(),
        "mae_raw": torch.mean(abs_raw).item(),
    }

    for idx, key in enumerate(KEYS):
        metrics[f"mae_norm_{key}"] = torch.mean(abs_norm[:, idx]).item()
        metrics[f"mse_norm_{key}"] = torch.mean(sq_norm[:, idx]).item()
        metrics[f"mae_raw_{key}"] = torch.mean(abs_raw[:, idx]).item()
        metrics[f"mse_raw_{key}"] = torch.mean(sq_raw[:, idx]).item()

    return metrics


def extract_json_blob(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return stripped
    except json.JSONDecodeError:
        pass

    fenced = stripped.split("```json")
    if len(fenced) > 1:
        tail = fenced[1].split("```", 1)[0].strip()
        if tail:
            return tail

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        return stripped[first : last + 1]
    return None


def _extract_vector_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("parsed_payload_not_object")

    candidates: list[Any] = [payload]
    candidates.extend(
        [
            payload.get("hidden_params"),
            payload.get("target_hidden_params"),
            payload.get("targetHiddenParams"),
            payload.get("target_vector"),
            payload.get("vector"),
        ]
    )

    for candidate in candidates:
        if isinstance(candidate, dict):
            if all(key in candidate for key in KEYS):
                return candidate
            nested_vector = candidate.get("vector")
            if isinstance(nested_vector, dict) and all(key in nested_vector for key in KEYS):
                return nested_vector

    raise ValueError("vector_payload_not_found")


def sanitize_vector(payload: Any, target_scale: int) -> dict[str, int]:
    vector_payload = _extract_vector_payload(payload)
    out: dict[str, int] = {}
    for key in KEYS:
        value = float(vector_payload[key])
        if not math.isfinite(value):
            raise ValueError(f"non_finite:{key}")
        out[key] = clamp_int(value, lo=0, hi=target_scale)
    return out


def parse_generated_vector(text: str, target_scale: int) -> tuple[dict[str, int] | None, str | None]:
    blob = extract_json_blob(text)
    if not blob:
        return None, "json_block_not_found"

    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as error:
        return None, f"json_decode_error:{error}"

    try:
        return sanitize_vector(parsed, target_scale=target_scale), None
    except Exception as error:  # noqa: BLE001
        return None, f"invalid_vector:{error}"


def _extract_hard_cases(
    dataset: Dataset,
    preds_raw: torch.Tensor,
    labels_raw: torch.Tensor,
    parse_errors: list[str | None],
    generated_texts: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or len(dataset) == 0:
        return []

    abs_raw = torch.abs(preds_raw - labels_raw)
    mae_by_sample = torch.mean(abs_raw, dim=1)
    take = min(top_k, mae_by_sample.shape[0])
    top_indices = torch.topk(mae_by_sample, k=take).indices.tolist()

    hard_cases: list[dict[str, Any]] = []
    for rank, sample_idx in enumerate(top_indices):
        row = dataset[int(sample_idx)]
        request_text = str(row.get("request_text", ""))
        source_type = str(row.get("source_type", "request_text"))
        weather = str(row.get("weather", "sunny"))
        target = {key: float(labels_raw[sample_idx, idx].item()) for idx, key in enumerate(KEYS)}
        predicted = {key: float(preds_raw[sample_idx, idx].item()) for idx, key in enumerate(KEYS)}
        abs_error_raw = {key: float(abs_raw[sample_idx, idx].item()) for idx, key in enumerate(KEYS)}
        hard_cases.append(
            {
                "rank": rank + 1,
                "sample_index": int(sample_idx),
                "request_text": request_text,
                "source_type": source_type,
                "weather": weather,
                "target_vector": target,
                "predicted_vector": predicted,
                "abs_error_raw": abs_error_raw,
                "mae_raw": float(mae_by_sample[sample_idx].item()),
                "parse_error": parse_errors[sample_idx],
                "generated_text": generated_texts[sample_idx],
            }
        )
    return hard_cases


def evaluate_generation(
    model: torch.nn.Module,
    dataset: Dataset,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    target_scale: int,
    hard_cases_top_k: int = 0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=EvalGenerationCollator())

    preds_raw_rows: list[list[float]] = []
    labels_raw_rows: list[list[float]] = []
    parse_errors: list[str | None] = []
    generated_texts: list[str] = []
    was_training = model.training

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prompts = list(batch["prompts"])
            tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            generated_ids = model.model.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for idx in range(generated_ids.shape[0]):
                prompt_len = int(attention_mask[idx].sum().item())
                continuation_ids = generated_ids[idx][prompt_len:].tolist()
                generated_text = str(tokenizer.decode(continuation_ids, skip_special_tokens=True)).strip()
                vector, parse_error = parse_generated_vector(generated_text, target_scale=target_scale)
                if vector is None:
                    vector = default_vector_for_scale(target_scale)

                preds_raw_rows.append([float(vector[key]) for key in KEYS])
                labels_raw_rows.append([float(v) for v in batch["target_raw"][idx]])
                parse_errors.append(parse_error)
                generated_texts.append(generated_text[:600])

    if was_training:
        model.train()

    if not preds_raw_rows:
        return {
            "mse_norm": 0.0,
            "mae_norm": 0.0,
            "mse_raw": 0.0,
            "mae_raw": 0.0,
            "json_valid_rate": 0.0,
            "parse_error_rate": 1.0,
        }, []

    preds_raw = torch.tensor(preds_raw_rows, dtype=torch.float32)
    labels_raw = torch.tensor(labels_raw_rows, dtype=torch.float32)
    metrics = _format_regression_metrics(preds_raw, labels_raw, target_scale=target_scale)
    valid_count = sum(1 for error in parse_errors if not error)
    total = len(parse_errors)
    metrics["json_valid_rate"] = valid_count / total if total > 0 else 0.0
    metrics["parse_error_rate"] = 1.0 - metrics["json_valid_rate"]
    metrics["target_scale"] = float(target_scale)

    hard_cases = _extract_hard_cases(
        dataset=dataset,
        preds_raw=preds_raw,
        labels_raw=labels_raw,
        parse_errors=parse_errors,
        generated_texts=generated_texts,
        top_k=hard_cases_top_k,
    )
    return metrics, hard_cases


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")


class DetailedEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_fn: Any,
        output_path: Path,
        detailed_eval_steps: int,
        has_wandb: bool,
        has_trackio: bool,
    ) -> None:
        self.eval_fn = eval_fn
        self.output_path = output_path
        self.detailed_eval_steps = max(0, detailed_eval_steps)
        self.has_wandb = has_wandb
        self.has_trackio = has_trackio
        self.last_logged_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if self.detailed_eval_steps <= 0:
            return control

        step = int(state.global_step)
        if step <= 0 or step == self.last_logged_step:
            return control
        if step % self.detailed_eval_steps != 0:
            return control

        metrics, hard_cases = self.eval_fn()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "metrics": metrics,
            "hard_cases_count": len(hard_cases),
            "top_hard_case": hard_cases[0] if hard_cases else None,
        }
        append_jsonl(self.output_path, payload)

        log_payload = {f"iter_eval/{key}": value for key, value in metrics.items()}
        log_payload["iter_eval/step"] = step
        log_payload["iter_eval/hard_cases_count"] = len(hard_cases)

        if self.has_wandb:
            import wandb

            wandb.log(log_payload)

        if self.has_trackio:
            import trackio

            trackio.log(log_payload)

        self.last_logged_step = step
        return control


def setup_wandb(args: argparse.Namespace) -> bool:
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        return False

    import wandb

    wandb.login(key=api_key, relogin=True)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_run_group,
        job_type="train",
        name=args.run_name,
        config={
            "objective": "next_token_json_sft",
            "base_model": args.base_model,
            "hub_model_id": args.hub_model_id,
            "dataset_repo": args.dataset_repo,
            "dataset_version": args.dataset_version,
            "source_type_mix": args.source_type_mix,
            "target_scale": args.target_scale,
            "cycle_id": args.cycle_id,
            "weave_project": args.weave_project,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_length": args.max_length,
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "detailed_eval_steps": args.detailed_eval_steps,
            "max_steps": args.max_steps,
            "hard_cases_top_k": args.hard_cases_top_k,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_keys": list(KEYS),
        },
        tags=[
            "hackathon",
            "hf-jobs",
            "ministral-3b",
            "request-to-hidden",
            "sft",
            "next-token",
            "json-generation",
            args.cycle_id,
        ],
    )

    wandb.define_metric("iter_eval/step")
    wandb.define_metric("iter_eval/*", step_metric="iter_eval/step")
    return True


def log_dataset_artifact_to_wandb(args: argparse.Namespace) -> None:
    import wandb

    artifact = wandb.Artifact(
        f"dataset-{args.run_name}",
        type="dataset",
        metadata={
            "dataset_repo": args.dataset_repo,
            "dataset_version": args.dataset_version,
            "source_type_mix": args.source_type_mix,
            "target_scale": args.target_scale,
            "train_split": args.train_split,
            "valid_split": args.valid_split,
        },
    )
    if args.dataset_repo:
        artifact.add_reference(f"https://huggingface.co/datasets/{args.dataset_repo}")
    if args.local_train and Path(args.local_train).exists():
        artifact.add_file(args.local_train, name="train.jsonl")
    if args.local_valid and Path(args.local_valid).exists():
        artifact.add_file(args.local_valid, name="validation.jsonl")
    wandb.log_artifact(artifact, aliases=["input", args.cycle_id])


def maybe_setup_trackio(args: argparse.Namespace) -> bool:
    if not args.enable_trackio:
        return False

    import trackio

    trackio.init(
        project=args.trackio_project,
        name=args.run_name,
        space_id=args.trackio_space_id,
        config={
            "objective": "next_token_json_sft",
            "base_model": args.base_model,
            "hub_model_id": args.hub_model_id,
            "dataset_repo": args.dataset_repo,
            "target_scale": args.target_scale,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "eval_steps": args.eval_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        },
    )
    return True


def save_adapter_model(
    model: torch.nn.Module,
    tokenizer: Any,
    output_dir: str,
    base_model: str,
    target_scale: int,
) -> Path:
    adapter_dir = Path(output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.model.language_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    readme = [
        "# Atelier Kotone JSON SFT Adapter",
        "",
        f"Base model: `{base_model}`",
        "",
        "This adapter is trained with next-token SFT and outputs strict JSON",
        "for hidden parameter prediction with keys:",
        "energy, warmth, brightness, acousticness, complexity, nostalgia.",
    ]
    (adapter_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    metadata = {
        "task": "request_text_to_hidden_params_json_generation",
        "target_keys": list(KEYS),
        "target_scale": f"integer 0-{target_scale}",
        "output_format": "strict_json_object",
        "objective": "next_token_prediction",
    }
    (adapter_dir / "sft_output_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return adapter_dir


def push_adapter_to_hub(adapter_dir: Path, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required to push model artifacts")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(adapter_dir))


def maybe_log_eval_to_weave(
    args: argparse.Namespace,
    hard_cases: list[dict[str, Any]],
) -> Path | None:
    if not args.enable_weave_trace:
        return None

    try:
        import weave
    except Exception:  # noqa: BLE001
        return None

    try:
        weave.init(args.weave_project)
    except Exception:  # noqa: BLE001
        return None

    @weave.op()
    def log_case(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def build_trace_url(trace_id: str) -> str:
        entity = str(args.wandb_entity or "").strip()
        project = str(args.wandb_project or "").strip()
        project_name = project
        if "/" in project:
            left, right = project.split("/", 1)
            if not entity:
                entity = left
            project_name = right
        if entity:
            return f"https://wandb.ai/{entity}/{project_name}/weave/traces?query={trace_id}"
        return f"https://wandb.ai/{project_name}/weave/traces?query={trace_id}"

    traces: list[dict[str, Any]] = []
    limit = min(20, len(hard_cases))
    for index in range(limit):
        case = hard_cases[index]
        trace_id = f"ft_eval_{uuid.uuid4().hex[:16]}"
        payload = {
            "trace_id": trace_id,
            "run_name": args.run_name,
            "cycle_id": args.cycle_id,
            "dataset_version": args.dataset_version,
            "model_id": args.hub_model_id,
            "rank": case.get("rank"),
            "request_text": case.get("request_text"),
            "target_vector": case.get("target_vector"),
            "predicted_vector": case.get("predicted_vector"),
            "abs_error_raw": case.get("abs_error_raw"),
            "mae_raw": case.get("mae_raw"),
            "parse_error": case.get("parse_error"),
        }
        log_case(payload)
        traces.append(
            {
                "trace_id": trace_id,
                "trace_url_hint": build_trace_url(trace_id),
                "rank": case.get("rank"),
                "mae_raw": case.get("mae_raw"),
                "parse_error": case.get("parse_error"),
            }
        )

    out_path = Path(args.output_dir) / "weave_eval_traces.json"
    out_path.write_text(json.dumps({"run_name": args.run_name, "rows": traces}, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()

    train_ds, valid_ds = load_splits(args)
    tokenizer = MistralCommonBackend.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_lm = build_sft_dataset(train_ds, tokenizer, "train", args.max_length, args.target_scale)
    valid_lm = build_sft_dataset(valid_ds, tokenizer, "valid", args.max_length, args.target_scale)

    model = Mistral3ForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        attn_implementation="eager",
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.model.language_model = get_peft_model(model.model.language_model, peft_config)

    has_wandb = setup_wandb(args)
    has_trackio = maybe_setup_trackio(args)
    if has_wandb:
        log_dataset_artifact_to_wandb(args)

    report_to: list[str] = []
    if has_wandb:
        report_to.append("wandb")
    if has_trackio:
        report_to.append("trackio")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=max(1, args.logging_steps),
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        max_steps=args.max_steps,
        bf16=True,
        report_to=report_to,
        remove_unused_columns=False,
    )

    collator = SFTDataCollator(tokenizer)
    metrics_dir = Path(args.output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    iter_eval_metrics_path = metrics_dir / "iter_eval_metrics.jsonl"
    if iter_eval_metrics_path.exists():
        iter_eval_metrics_path.unlink()

    trainer = HiddenParamSFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_lm,
        eval_dataset=valid_lm,
        data_collator=collator,
    )

    def eval_fn() -> tuple[dict[str, float], list[dict[str, Any]]]:
        return evaluate_generation(
            model=model,
            dataset=valid_lm,
            tokenizer=tokenizer,
            batch_size=max(1, args.batch_size),
            max_length=args.max_length,
            max_new_tokens=args.generation_max_new_tokens,
            target_scale=args.target_scale,
            hard_cases_top_k=args.hard_cases_top_k,
        )

    trainer.add_callback(
        DetailedEvalCallback(
            eval_fn=eval_fn,
            output_path=iter_eval_metrics_path,
            detailed_eval_steps=args.detailed_eval_steps,
            has_wandb=has_wandb,
            has_trackio=has_trackio,
        )
    )

    train_result = trainer.train()
    eval_result, hard_cases = eval_fn()
    hard_cases_path = metrics_dir / "hard_cases.valid.jsonl"
    hard_cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in hard_cases),
        encoding="utf-8",
    )

    adapter_dir = save_adapter_model(model, tokenizer, args.output_dir, args.base_model, args.target_scale)
    if args.push_to_hub:
        push_adapter_to_hub(adapter_dir, args.hub_model_id)

    weave_trace_path = maybe_log_eval_to_weave(args, hard_cases)

    metrics_path = metrics_dir / "final_metrics.json"
    metrics_payload = {
        "objective": "next_token_json_sft",
        "target_keys": list(KEYS),
        "cycle_id": args.cycle_id,
        "dataset_version": args.dataset_version,
        "source_type_mix": args.source_type_mix,
        "target_scale": args.target_scale,
        "train": train_result.metrics,
        "eval": eval_result,
        "hard_cases_top_k": args.hard_cases_top_k,
        "hard_cases_path": str(hard_cases_path),
        "iter_eval_metrics_path": str(iter_eval_metrics_path),
        "weave_trace_path": str(weave_trace_path) if weave_trace_path else None,
        "train_examples": len(train_lm),
        "valid_examples": len(valid_lm),
        "adapter_dir": str(adapter_dir),
        "hub_model_id": args.hub_model_id if args.push_to_hub else None,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if has_wandb:
        import wandb

        final_payload = {
            "objective/train_loss": train_result.metrics.get("train_loss"),
            "dataset/train_examples": len(train_lm),
            "dataset/valid_examples": len(valid_lm),
            "dataset/hard_cases_count": len(hard_cases),
            "dataset/version": args.dataset_version,
            "dataset/source_type_mix": args.source_type_mix,
            "dataset/target_scale": args.target_scale,
            "run/cycle_id": args.cycle_id,
        }
        final_payload.update({f"eval/{key}": value for key, value in eval_result.items()})
        wandb.log(final_payload)

        metrics_artifact = wandb.Artifact(f"final-metrics-{args.run_name}", type="metrics")
        metrics_artifact.add_file(str(metrics_path))
        metrics_artifact.add_file(str(hard_cases_path))
        if iter_eval_metrics_path.exists():
            metrics_artifact.add_file(str(iter_eval_metrics_path))
        if weave_trace_path and weave_trace_path.exists():
            metrics_artifact.add_file(str(weave_trace_path))
        wandb.log_artifact(metrics_artifact, aliases=["latest", args.cycle_id])

        model_artifact = wandb.Artifact(
            f"model-adapter-{args.run_name}",
            type="model",
            metadata={
                "hub_model_id": args.hub_model_id,
                "base_model": args.base_model,
                "cycle_id": args.cycle_id,
                "dataset_version": args.dataset_version,
                "objective": "next_token_json_sft",
                "target_scale": args.target_scale,
            },
        )
        model_artifact.add_dir(str(adapter_dir))
        wandb.log_artifact(model_artifact, aliases=["latest", args.cycle_id])
        wandb.finish()

    if has_trackio:
        import trackio

        trackio_payload = {f"eval/{key}": value for key, value in eval_result.items()}
        trackio_payload["dataset/hard_cases_count"] = len(hard_cases)
        trackio.log(trackio_payload)
        trackio.finish()

    print(
        "Training complete (next-token JSON SFT). "
        f"Metrics: {metrics_path} | Hard cases: {hard_cases_path} | Iter eval: {iter_eval_metrics_path}"
    )


if __name__ == "__main__":
    main()
