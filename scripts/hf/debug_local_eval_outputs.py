#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.6.0",
#   "transformers>=4.52.0",
#   "datasets>=3.2.0",
#   "accelerate>=1.2.0",
# ]
# ///

from __future__ import annotations

import json
import os
import re
from typing import Any

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def extract_json_blob(text: str) -> str | None:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return stripped
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return None


def try_parse_vector(text: str, target_scale: int = 10) -> tuple[dict[str, int] | None, str | None]:
    blob = extract_json_blob(text)
    if not blob:
        return None, "json_block_not_found"
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as error:
        return None, f"json_decode_error:{error}"
    if not isinstance(parsed, dict):
        return None, "parsed_payload_not_object"

    payload = parsed.get("hidden_params") if isinstance(parsed.get("hidden_params"), dict) else parsed
    if not isinstance(payload, dict):
        return None, "hidden_params_missing_or_not_object"

    out: dict[str, int] = {}
    for key in KEYS:
        if key not in payload:
            return None, f"missing_key:{key}"
        try:
            value = int(round(float(payload[key])))
        except (TypeError, ValueError):
            return None, f"invalid_value:{key}"
        if value < 0 or value > target_scale:
            return None, f"out_of_range:{key}={value}"
        out[key] = value
    return out, None


def build_prompt(request_text: str) -> str:
    return "\n".join(
        [
            "You estimate 6 hidden music parameters.",
            "Return strict JSON only with schema:",
            '{"request_text":"...", "hidden_params":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
            "Copy request_text exactly from input.",
            "Values in hidden_params must be integers between 0 and 10.",
            f"request_text={request_text}",
        ]
    )


def main() -> None:
    model_id = env_str("DEBUG_MODEL_ID", "Haruk1y/atelier-kotone-ministral3b-ft-selfimprove1-full")
    dataset_repo = env_str("EVAL_DATASET_REPO_ID", "Haruk1y/atelier-kotone-ft-request-hidden")
    dataset_config = env_str("EVAL_DATASET_CONFIG", "")
    dataset_split = env_str("EVAL_DATASET_SPLIT", "test")
    max_samples = max(1, env_int("DEBUG_MAX_SAMPLES", 5))
    target_scale = env_int("EVAL_TARGET_SCALE", 10)
    trust_remote_code = env_str("EVAL_LOCAL_TRUST_REMOTE_CODE", "false").lower() in {"1", "true", "yes", "on"}
    hf_token = env_str("HF_TOKEN", "")
    auth_token = hf_token.strip() or None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=auth_token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=auth_token,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    dataset = load_dataset(dataset_repo, dataset_config or None, split=dataset_split)

    emitted = 0
    for row in dataset:
        request_text = str(row.get("request_text") or row.get("request") or row.get("input") or "").strip()
        if not request_text:
            continue
        prompt = build_prompt(request_text=request_text)
        messages = [
            {"role": "system", "content": "You output only strict JSON with six integer fields."},
            {"role": "user", "content": prompt},
        ]

        input_text = prompt
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                input_text = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            except TypeError:
                input_text = str(tokenizer.apply_chat_template(messages, tokenize=False))
            except Exception:
                input_text = prompt

        model_inputs = tokenizer(input_text, return_tensors="pt")
        try:
            first_param = next(model.parameters())
            model_inputs = {key: value.to(first_param.device) for key, value in model_inputs.items()}
        except Exception:
            pass

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": 96,
            "do_sample": False,
        }
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = int(tokenizer.eos_token_id)

        import torch  # local import for uv script runtime

        with torch.no_grad():
            output_ids = model.generate(**model_inputs, **generation_kwargs)

        input_len = int(model_inputs["input_ids"].shape[-1])
        generated_ids = output_ids[0][input_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        vector, parse_error = try_parse_vector(generated_text, target_scale=target_scale)

        print(
            json.dumps(
                {
                    "idx": emitted,
                    "request_text": request_text,
                    "generated_text": generated_text,
                    "parse_error": parse_error,
                    "parsed_vector": vector,
                },
                ensure_ascii=False,
            )
        )

        emitted += 1
        if emitted >= max_samples:
            break

    print(json.dumps({"model_id": model_id, "emitted": emitted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
