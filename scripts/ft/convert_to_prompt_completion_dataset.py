#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0.0",
#   "huggingface-hub>=0.34.0",
# ]
# ///

"""Convert request-hidden dataset to explicit prompt/completion format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")
WEATHER_VALUES = {"sunny", "cloudy", "rainy"}


def clamp_int(value: float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(round(float(value)))))


def convert_scale_int(value: float, source_scale: int, target_scale: int) -> int:
    src = max(1, int(source_scale))
    dst = max(1, int(target_scale))
    bounded = clamp_int(value, lo=0, hi=src)
    return clamp_int((bounded / src) * dst, lo=0, hi=dst)


def build_inference_prompt(request_text: str, source_type: str) -> str:
    return "\n".join(
        [
            "You estimate 6 hidden music parameters.",
            "Each value must be integer between 0 and 10.",
            "",
            "- energy: silent(0) ↔ intense(10)",
            "- warmth: mechanical(0) ↔ warm(10)",
            "- brightness: dark(0) ↔ bright(10)",
            "- acousticness: electronic(0) ↔ acoustic(10)",
            "- complexity: simple(0) ↔ complex(10)",
            "- nostalgia: futuristic(0) ↔ nostalgic(10)",
            "",
            "Return JSON only with exactly these keys:",
            "energy, warmth, brightness, acousticness, complexity, nostalgia",
            f"source_type={source_type}",
            f"request_text={request_text}",
        ]
    )


def extract_request_text(row: dict[str, Any]) -> str:
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
    raise ValueError("request_text_not_found")


def extract_source_type(row: dict[str, Any]) -> str:
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


def extract_weather(row: dict[str, Any]) -> str:
    direct = str(row.get("weather", "")).strip().lower()
    if direct in WEATHER_VALUES:
        return direct

    request = str(row.get("request_text", "")).lower()
    if any(token in request for token in ("rain", "storm", "drizzle", "wet")):
        return "rainy"
    if any(token in request for token in ("night", "evening", "dusk", "cloud", "fog")):
        return "cloudy"
    if any(token in request for token in ("sun", "morning", "bright", "daylight")):
        return "sunny"
    return "cloudy"


def extract_target_vector(row: dict[str, Any]) -> dict[str, float]:
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
            nested_vector = parsed.get("vector") if isinstance(parsed, dict) else None
            if isinstance(nested_vector, dict) and all(key in nested_vector for key in KEYS):
                return {key: float(nested_vector[key]) for key in KEYS}

    raise ValueError("target_hidden_params_not_found")


def infer_vector_scale(vector: dict[str, Any]) -> int:
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


def extract_target_scale(row: dict[str, Any], default_scale: int) -> int:
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
        return infer_vector_scale(extract_target_vector(row))
    except Exception:  # noqa: BLE001
        return default_scale


def convert_split(split: Dataset, split_name: str, target_scale: int) -> Dataset:
    records: list[dict[str, Any]] = []
    for row in split:
        request_text = extract_request_text(row)
        source_type = extract_source_type(row)
        weather = extract_weather(row)
        target = extract_target_vector(row)
        source_scale = extract_target_scale(row, target_scale)

        vector = {
            key: convert_scale_int(float(target[key]), source_scale=source_scale, target_scale=target_scale)
            for key in KEYS
        }
        prompt = build_inference_prompt(request_text=request_text, source_type=source_type)
        completion = json.dumps(vector, ensure_ascii=False)

        records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "request_text": request_text,
                "source_type": source_type,
                "weather": weather,
                "target_scale": int(target_scale),
                "target_hidden_params": {"vector": vector},
            }
        )

    print(f"[convert] split={split_name} rows={len(records)}")
    return Dataset.from_list(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", default=os.getenv("HF_FT_DATASET_REPO_ID", "Haruk1y/atelier-kotone-ft-request-hidden"))
    parser.add_argument("--source-config", default=os.getenv("HF_FT_DATASET_CONFIG", ""))
    parser.add_argument("--train-split", default=os.getenv("HF_FT_TRAIN_SPLIT", "train"))
    parser.add_argument("--valid-split", default=os.getenv("HF_FT_VALID_SPLIT", "validation"))
    parser.add_argument("--test-split", default=os.getenv("HF_FT_TEST_SPLIT", "test"))
    parser.add_argument(
        "--output-repo",
        default=os.getenv("HF_FT_PROMPT_COMPLETION_DATASET_REPO_ID", "Haruk1y/atelier-kotone-ft-request-hidden-prompt-completion-20260301"),
    )
    parser.add_argument("--target-scale", type=int, default=10)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--local-out-dir", default="artifacts/ft_prompt_completion_dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source_config:
        src = load_dataset(args.source_repo, args.source_config)
    else:
        src = load_dataset(args.source_repo)

    splits: dict[str, Dataset] = {}
    for src_name, out_name in (
        (args.train_split, "train"),
        (args.valid_split, "validation"),
        (args.test_split, "test"),
    ):
        if src_name in src:
            splits[out_name] = convert_split(src[src_name], out_name, target_scale=args.target_scale)

    if not splits:
        raise ValueError("no source splits found")

    out = DatasetDict(splits)
    out_dir = Path(args.local_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, dataset in out.items():
        path = out_dir / f"{split_name}.jsonl"
        dataset.to_json(str(path), force_ascii=False)
        print(f"[save] {path} rows={len(dataset)}")

    if args.no_push:
        print("[done] no push requested")
        return

    token = os.getenv("HF_TOKEN")
    out.push_to_hub(args.output_repo, private=args.private, token=token)
    print(f"[done] pushed: https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
