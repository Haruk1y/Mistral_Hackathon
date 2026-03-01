#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6.0",
#   "accelerate>=1.10.0",
#   "datasets>=3.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "mistral-common>=1.8.6",
# ]
# ///

"""Compare prompt variants for strict JSON output quality."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralCommonBackend

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")


def build_prompt(mode: str, request_text: str, source_type: str) -> str:
    if mode == "baseline":
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

    if mode == "strict_v1":
        return "\n".join(
            [
                "Task: estimate 6 hidden music parameters from the request.",
                "Output EXACTLY one JSON object and nothing else.",
                "No markdown, no explanation, no code block, no extra tokens.",
                "Start with '{' and end with '}'.",
                "Use this exact schema and key names:",
                '{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}',
                "All values must be integers in [0, 10].",
                f"source_type={source_type}",
                f"request_text={request_text}",
            ]
        )

    if mode == "strict_v2":
        return "\n".join(
            [
                "Respond with a single-line valid JSON object only.",
                "Do not output any characters before '{' or after '}'.",
                "Do not repeat JSON.",
                "Use this exact format:",
                '{"energy":E,"warmth":W,"brightness":B,"acousticness":A,"complexity":C,"nostalgia":N}',
                "Replace E,W,B,A,C,N with integers from 0 to 10.",
                "No null, no string numbers, no trailing comma.",
                f"source_type={source_type}",
                f"request_text={request_text}",
            ]
        )

    if mode == "strict_v3":
        return "\n".join(
            [
                "Return exactly one JSON object and stop immediately.",
                "Forbidden in output: </s>, markdown, code fences, comments, explanations, repeated JSON.",
                "Output must match this schema exactly:",
                '{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}',
                "Constraints:",
                "- all six values are integers 0..10",
                "- output starts with '{' and ends with '}'",
                "- after the final '}' output nothing",
                f"source_type={source_type}",
                f"request_text={request_text}",
            ]
        )

    raise ValueError(f"unknown prompt mode: {mode}")


def extract_request_text(row: dict[str, Any]) -> str:
    value = str(row.get("request_text", "")).strip()
    if value:
        return value
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") == "user":
                content = str(message.get("content", "")).strip()
                if content:
                    return content
    return ""


def extract_source_type(row: dict[str, Any]) -> str:
    value = str(row.get("source_type", "")).strip().lower()
    return value or "request_text"


def load_cases(dataset_repo: str, split: str, max_cases: int, seed: int) -> list[tuple[str, str]]:
    ds = load_dataset(dataset_repo, split=split)
    ds = ds.shuffle(seed=seed)
    take = min(max_cases, len(ds))
    cases: list[tuple[str, str]] = []
    for row in ds.select(range(take)):
        request_text = extract_request_text(row)
        source_type = extract_source_type(row)
        if not request_text:
            continue
        cases.append((source_type, request_text))
    return cases


def validate_schema(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = [key for key in KEYS if key not in payload]
    extra = [key for key in payload.keys() if key not in KEYS]
    if missing:
        errors.append(f"missing={','.join(missing)}")
    if extra:
        errors.append(f"extra={','.join(extra)}")

    for key in KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, int):
            errors.append(f"{key}_not_int")
            continue
        if value < 0 or value > 10:
            errors.append(f"{key}_out_of_range")
    return len(errors) == 0, errors


def extract_first_json(text: str) -> tuple[dict[str, Any] | None, int, int, str | None]:
    start = text.find("{")
    if start < 0:
        return None, -1, -1, "no_json_start"
    depth = 0
    end = -1
    for idx, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end < 0:
        return None, start, -1, "unterminated_json"
    blob = text[start : end + 1]
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, start, end, f"first_json_decode_error:{exc.msg}"
    if not isinstance(parsed, dict):
        return None, start, end, "first_json_not_object"
    return parsed, start, end, None


def classify_failure(completion: str, strict_error: str, first_json_error: str | None) -> str:
    if "</s>" in completion:
        return "eos_token_leak"
    if strict_error.startswith("strict_decode_error:Extra data"):
        return "extra_trailing_text"
    if strict_error.startswith("strict_decode_error:"):
        return "strict_json_decode_error"
    if first_json_error is not None:
        return first_json_error
    return "schema_or_unknown"


def evaluate_mode(
    model: Any,
    tokenizer: Any,
    mode: str,
    cases: list[tuple[str, str]],
    max_new_tokens: int,
) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    strict_ok = 0
    first_json_ok = 0
    categories = Counter()
    failures: list[dict[str, Any]] = []

    for idx, (source_type, request_text) in enumerate(cases, start=1):
        prompt = build_prompt(mode, request_text=request_text, source_type=source_type)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(model_device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = generated[0][encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        strict_parsed: dict[str, Any] | None = None
        strict_error: str | None = None
        try:
            maybe = json.loads(completion)
            if isinstance(maybe, dict):
                strict_parsed = maybe
            else:
                strict_error = "strict_not_object"
        except json.JSONDecodeError as exc:
            strict_error = f"strict_decode_error:{exc.msg}"

        strict_schema_errors: list[str] = []
        if strict_parsed is not None:
            strict_valid, strict_schema_errors = validate_schema(strict_parsed)
            if strict_valid:
                strict_ok += 1
            else:
                strict_error = "strict_schema_mismatch"

        first_parsed, first_start, first_end, first_err = extract_first_json(completion)
        if first_parsed is not None:
            first_valid, _ = validate_schema(first_parsed)
            if first_valid:
                first_json_ok += 1

        if strict_error is not None:
            category = classify_failure(completion, strict_error, first_err)
            categories[category] += 1
            if len(failures) < 5:
                failures.append(
                    {
                        "case_index": idx,
                        "source_type": source_type,
                        "request_text": request_text[:220],
                        "strict_error": strict_error,
                        "strict_schema_errors": strict_schema_errors,
                        "first_json_error": first_err,
                        "first_json_span": [first_start, first_end],
                        "completion_raw": completion[:600],
                    }
                )

    total = len(cases)
    return {
        "mode": mode,
        "total_cases": total,
        "strict_valid_count": strict_ok,
        "strict_valid_rate": (strict_ok / total) if total else 0.0,
        "first_json_valid_count": first_json_ok,
        "first_json_valid_rate": (first_json_ok / total) if total else 0.0,
        "failure_categories": dict(categories),
        "failure_examples": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Haruk1y/atelier-kotone-ministral3b-ft-resplit-vector-20260301-full",
    )
    parser.add_argument(
        "--dataset-repo",
        default="Haruk1y/atelier-kotone-ft-request-hidden",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--tokenizer-backend",
        choices=("auto", "mistral_common"),
        default="mistral_common",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("baseline", "strict_v1", "strict_v2", "strict_v3"),
        default=[],
        help="Prompt mode(s) to evaluate. Default: all modes.",
    )
    args = parser.parse_args()

    modes = args.mode or ["baseline", "strict_v1", "strict_v2", "strict_v3"]
    cases = load_cases(args.dataset_repo, args.dataset_split, args.max_cases, args.seed)

    if not cases:
        raise RuntimeError("no cases loaded")

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map: str | None = "auto" if use_cuda else None

    print(f"[debug] model_id={args.model_id}")
    print(f"[debug] dataset_repo={args.dataset_repo} split={args.dataset_split}")
    print(f"[debug] tokenizer_backend={args.tokenizer_backend}")
    print(f"[debug] use_cuda={use_cuda} dtype={dtype}")
    print(f"[debug] total_cases={len(cases)}")
    print(f"[debug] modes={modes}")

    if args.tokenizer_backend == "mistral_common":
        tokenizer = MistralCommonBackend.from_pretrained(args.model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    report: dict[str, Any] = {"model_id": args.model_id, "modes": []}
    for mode in modes:
        result = evaluate_mode(
            model=model,
            tokenizer=tokenizer,
            mode=mode,
            cases=cases,
            max_new_tokens=args.max_new_tokens,
        )
        report["modes"].append(result)

    print("\n=== REPORT_JSON ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
