#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6.0",
#   "accelerate>=1.10.0",
#   "peft>=0.13.0",
#   "transformers @ git+https://github.com/huggingface/transformers",
#   "mistral-common>=1.8.6",
# ]
# ///

"""Debug inference job for merged full model JSON output validation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mistral3ForConditionalGeneration, MistralCommonBackend

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")


def build_inference_prompt(request_text: str, source_type: str) -> str:
    # Keep this exactly aligned with train_sft_request_to_hidden_lm.py.
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


def extract_first_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    start = text.find("{")
    if start < 0:
        return None, "no_json_start"

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
        return None, "unterminated_json"

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"
    if not isinstance(obj, dict):
        return None, "json_not_object"
    return obj, None


def validate_payload(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = [key for key in KEYS if key not in payload]
    extra = [key for key in payload.keys() if key not in KEYS]
    if missing:
        errors.append(f"missing_keys={missing}")
    if extra:
        errors.append(f"extra_keys={extra}")

    for key in KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if not isinstance(value, int):
            errors.append(f"{key}_not_int={value!r}")
            continue
        if value < 0 or value > 10:
            errors.append(f"{key}_out_of_range={value}")

    return len(errors) == 0, errors


def parse_case(text: str) -> tuple[str, str]:
    if "|" not in text:
        raise ValueError(f"invalid --case format: {text!r} (expected source_type|request_text)")
    source_type, request_text = text.split("|", 1)
    source_type = source_type.strip()
    request_text = request_text.strip()
    if not source_type or not request_text:
        raise ValueError(f"invalid --case values: {text!r}")
    return source_type, request_text


def ensure_peft_generation_compat(model: torch.nn.Module) -> None:
    if hasattr(model, "prepare_inputs_for_generation"):
        return

    def _prepare_inputs_for_generation(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prepared: dict[str, Any] = {"input_ids": input_ids}
        if attention_mask is not None:
            prepared["attention_mask"] = attention_mask
        prepared.update(kwargs)
        return prepared

    setattr(model, "prepare_inputs_for_generation", _prepare_inputs_for_generation)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Haruk1y/atelier-kotone-ministral3b-ft-resplit-vector-20260301-full",
        help="Merged full model repo id.",
    )
    parser.add_argument(
        "--base-model-id",
        default="mistralai/Ministral-3-3B-Instruct-2512",
        help="Base model id for adapter mode.",
    )
    parser.add_argument(
        "--adapter-model-id",
        default="",
        help="Optional LoRA adapter repo id. When set, loads base+adapter instead of --model-id.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--tokenizer-backend",
        choices=("auto", "mistral_common"),
        default="auto",
        help="Tokenizer backend for prompt encode/decode.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Test case in format source_type|request_text. Can be specified multiple times.",
    )
    args = parser.parse_args()

    raw_cases = args.case or [
        "request_text|明るくて少しノスタルジックな春の朝の曲にして",
        "request_text|夜のドライブ向けに、エネルギー高めでクールな感じ",
        "rule_prompt|acoustic guitar 중심で温かく、複雑さは低め",
    ]
    cases = [parse_case(case) for case in raw_cases]
    adapter_model_id = str(args.adapter_model_id or "").strip()
    infer_mode = "adapter" if adapter_model_id else "full"

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map: str | None = "auto" if use_cuda else None

    if infer_mode == "adapter":
        print(f"[debug] infer_mode=adapter")
        print(f"[debug] base_model_id={args.base_model_id}")
        print(f"[debug] adapter_model_id={adapter_model_id}")
    else:
        print(f"[debug] infer_mode=full")
        print(f"[debug] model_id={args.model_id}")
    print(f"[debug] use_cuda={use_cuda}")
    print(f"[debug] dtype={dtype}")
    print(f"[debug] cases={len(cases)}")

    tokenizer_source = adapter_model_id if infer_mode == "adapter" else args.model_id
    fallback_tokenizer_source = args.base_model_id if infer_mode == "adapter" else args.model_id
    if args.tokenizer_backend == "mistral_common":
        try:
            tokenizer = MistralCommonBackend.from_pretrained(tokenizer_source)
        except Exception:
            tokenizer = MistralCommonBackend.from_pretrained(fallback_tokenizer_source)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if infer_mode == "adapter":
        base_model = Mistral3ForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
        ensure_peft_generation_compat(base_model.model.language_model)
        base_model.model.language_model = PeftModel.from_pretrained(
            base_model.model.language_model,
            adapter_model_id,
            is_trainable=False,
        )
        model = base_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    model.eval()

    invalid_cases = 0
    for index, (source_type, request_text) in enumerate(cases, start=1):
        prompt = build_inference_prompt(request_text=request_text, source_type=source_type)
        encoded = tokenizer(prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        encoded = {k: v.to(model_device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        completion_ids = generated[0][encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        parsed, parse_error = extract_first_json_object(completion)
        is_valid = False
        validation_errors: list[str] = []
        if parsed is not None:
            is_valid, validation_errors = validate_payload(parsed)
        else:
            validation_errors = [parse_error or "unknown_parse_error"]

        if not is_valid:
            invalid_cases += 1

        print(f"\n=== CASE {index} ===")
        print(f"source_type={source_type}")
        print(f"request_text={request_text}")
        print("prompt:")
        print(prompt)
        print("completion_raw:")
        print(completion)
        print(f"json_valid={is_valid}")
        if parsed is not None:
            print("parsed_json:")
            print(json.dumps(parsed, ensure_ascii=False))
        if validation_errors:
            print(f"errors={validation_errors}")

    valid_cases = len(cases) - invalid_cases
    print("\n=== SUMMARY ===")
    print(f"valid_cases={valid_cases}/{len(cases)}")

    if invalid_cases > 0:
        print("[result] FAILED")
        return 1

    print("[result] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
