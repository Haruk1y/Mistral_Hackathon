#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "requests>=2.32.0",
#   "wandb>=0.18.0",
#   "weave>=0.51.0",
# ]
# ///

"""Measured eval runner with W&B Models + Weave traces.

Evaluates frozen eval set in 3 modes:
- rule_baseline
- prompt_baseline
- fine_tuned

Outputs:
- artifacts/eval/runs/<run_id>.json
- artifacts/eval/samples/<run_id>.json
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")
SLOTS = ("style", "instrument", "mood", "gimmick")

DEFAULT_COST_PER_REQUEST_USD = {
    "rule_baseline": 0.0002,
    "prompt_baseline": 0.0028,
    "fine_tuned": 0.0019,
    "large_baseline": 0.0045,
}


def _read_target_scale() -> int:
    raw = os.getenv("EVAL_TARGET_SCALE") or os.getenv("HF_FT_TARGET_SCALE") or os.getenv("FT_TARGET_SCALE") or "10"
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = 10
    if parsed != 10:
        raise ValueError(f"EVAL_TARGET_SCALE must be 10 for this pipeline, got: {parsed}")
    return 10


EVAL_TARGET_SCALE = _read_target_scale()
THRESHOLD_SCALE = 100.0 / float(EVAL_TARGET_SCALE)


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


def normalize_backend_mode(value: str) -> str:
    lowered = str(value or "auto").strip().lower().replace("_", "-")
    if lowered in {"auto", "text-generation", "chat-completions", "local-transformers"}:
        return lowered
    return "auto"


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


def scale_from_100(value: float) -> int:
    return int(round(clamp((float(value) / 100.0) * float(EVAL_TARGET_SCALE), 0.0, float(EVAL_TARGET_SCALE))))


def threshold(value_100: float) -> float:
    return (float(value_100) / 100.0) * float(EVAL_TARGET_SCALE)


def infer_scale(vector: dict[str, Any]) -> int:
    values: list[float] = []
    for key in KEYS:
        value = vector.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    max_value = max(values) if values else float(EVAL_TARGET_SCALE)
    if max_value <= 10:
        return 10
    if max_value <= 100:
        return 100
    return EVAL_TARGET_SCALE


def normalize_to_eval_scale(vector: dict[str, Any], source_scale: int) -> dict[str, int]:
    src = max(1, int(source_scale))
    out: dict[str, int] = {}
    for key in KEYS:
        try:
            value = float(vector.get(key, 0))
        except (TypeError, ValueError):
            value = 0.0
        bounded = clamp(value, 0.0, float(src))
        out[key] = int(round(clamp((bounded / float(src)) * float(EVAL_TARGET_SCALE), 0.0, float(EVAL_TARGET_SCALE))))
    return out


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = max(0, min(len(sorted_values) - 1, int((p / 100.0) * len(sorted_values) + 0.9999) - 1))
    return float(sorted_values[idx])


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


def extract_vector_payload(payload: dict[str, Any]) -> dict[str, Any]:
    candidates: list[Any] = [
        payload,
        payload.get("hidden_params"),
        payload.get("target_hidden_params"),
        payload.get("targetHiddenParams"),
        payload.get("target_vector"),
        payload.get("vector"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if all(key in candidate for key in KEYS):
            return candidate
        nested = candidate.get("vector")
        if isinstance(nested, dict) and all(key in nested for key in KEYS):
            return nested
    raise ValueError("vector_payload_not_found")


def sanitize_vector(payload: dict[str, Any]) -> dict[str, int]:
    vector_payload = extract_vector_payload(payload)
    vector: dict[str, int] = {}
    for key in KEYS:
        value = vector_payload.get(key)
        if value is None:
            raise ValueError(f"missing key: {key}")
        vector[key] = int(round(clamp(float(value), 0, EVAL_TARGET_SCALE)))
    return vector


def derive_constraints(vector: dict[str, int]) -> dict[str, Any]:
    return {
        "preferredStyleTags": ["citypop_80s"] if vector["nostalgia"] > threshold(60) else ["pop_2000s"] if vector["brightness"] > threshold(65) else ["hiphop_90s"],
        "preferredGimmickTags": ["filter_rise"] if vector["energy"] > threshold(60) else ["beat_mute"],
        "avoidPartIds": ["style_2000s_pop"] if vector["brightness"] < threshold(22) else [],
    }


def derive_slot_top1(vector: dict[str, int]) -> dict[str, str]:
    style = "style_2000s_pop" if vector["brightness"] > threshold(68) else "style_80s_citypop" if vector["nostalgia"] > threshold(65) else "style_90s_hiphop"
    instrument = "inst_piano_upright" if vector["acousticness"] > threshold(70) else "inst_soft_strings" if vector["warmth"] > threshold(64) else "inst_analog_synth"
    mood = "mood_rain_ambience" if vector["brightness"] < threshold(35) else "mood_sun_glow" if vector["energy"] > threshold(62) else "mood_night_drive"
    gimmick = "gimmick_harmony_stack" if vector["complexity"] > threshold(55) else "gimmick_filter_rise" if vector["energy"] > threshold(62) else "gimmick_beat_mute"
    return {"style": style, "instrument": instrument, "mood": mood, "gimmick": gimmick}


def score_output_sanity(json_valid: bool, parse_error: str | None, vector: dict[str, int] | None) -> int:
    if not json_valid:
        return 12 if parse_error else 20
    if not vector:
        return 30
    spread = statistics.pstdev([vector[k] for k in KEYS]) if len(KEYS) > 1 else 0
    return int(round(clamp(72 + (spread * THRESHOLD_SCALE) * 0.6, 0, 100)))


def score_intent_from_error(abs_error: dict[str, float]) -> float:
    sample_mae = mean([abs_error[k] for k in KEYS])
    return float(clamp(100.0 - sample_mae * THRESHOLD_SCALE * 2.1, 0, 100))


def rule_based_predict(request_text: str) -> dict[str, int]:
    lower = request_text.lower()
    default = {
        "energy": scale_from_100(45),
        "warmth": scale_from_100(55),
        "brightness": scale_from_100(50),
        "acousticness": scale_from_100(65),
        "complexity": scale_from_100(38),
        "nostalgia": scale_from_100(60),
    }
    profiles = [
        (
            ["rain", "quiet", "evening", "night"],
            {
                "energy": scale_from_100(22),
                "warmth": scale_from_100(60),
                "brightness": scale_from_100(24),
                "acousticness": scale_from_100(72),
                "complexity": scale_from_100(32),
                "nostalgia": scale_from_100(72),
            },
        ),
        (
            ["smile", "bright", "market", "sun"],
            {
                "energy": scale_from_100(75),
                "warmth": scale_from_100(58),
                "brightness": scale_from_100(82),
                "acousticness": scale_from_100(38),
                "complexity": scale_from_100(48),
                "nostalgia": scale_from_100(42),
            },
        ),
        (
            ["focus", "study", "reading", "cafe"],
            {
                "energy": scale_from_100(34),
                "warmth": scale_from_100(54),
                "brightness": scale_from_100(44),
                "acousticness": scale_from_100(68),
                "complexity": scale_from_100(28),
                "nostalgia": scale_from_100(58),
            },
        ),
        (
            ["memory", "old", "nostalgia", "retro"],
            {
                "energy": scale_from_100(40),
                "warmth": scale_from_100(74),
                "brightness": scale_from_100(50),
                "acousticness": scale_from_100(76),
                "complexity": scale_from_100(40),
                "nostalgia": scale_from_100(88),
            },
        ),
    ]
    for keywords, vector in profiles:
        if any(keyword in lower for keyword in keywords):
            return vector
    return default


@dataclass(frozen=True)
class LocalTransformersSpec:
    base_model_id: str
    adapter_model_id: str
    device: str
    dtype: str
    max_new_tokens: int
    trust_remote_code: bool
    cache_dir: str | None


_LOCAL_TRANSFORMERS_CACHE: dict[str, tuple[Any, Any]] = {}


def _local_transformers_cache_key(spec: LocalTransformersSpec) -> str:
    return "|".join(
        [
            spec.base_model_id,
            spec.adapter_model_id,
            spec.device,
            spec.dtype,
            "1" if spec.trust_remote_code else "0",
            spec.cache_dir or "",
        ]
    )


def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalized = (dtype_name or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    mapping = {
        "bfloat16": getattr(torch_module, "bfloat16", None),
        "bf16": getattr(torch_module, "bfloat16", None),
        "float16": getattr(torch_module, "float16", None),
        "fp16": getattr(torch_module, "float16", None),
        "float32": getattr(torch_module, "float32", None),
        "fp32": getattr(torch_module, "float32", None),
    }
    resolved = mapping.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported_local_dtype:{dtype_name}")
    return resolved


def _load_local_transformers_bundle(
    spec: LocalTransformersSpec,
    hf_token: str,
) -> tuple[Any, Any]:
    cache_key = _local_transformers_cache_key(spec)
    cached = _LOCAL_TRANSFORMERS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"missing_local_transformers_dependency:{error}") from error

    auth_token = hf_token.strip() or None
    common_kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
    }
    if auth_token:
        common_kwargs["token"] = auth_token
    if spec.cache_dir:
        common_kwargs["cache_dir"] = spec.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(spec.base_model_id, **common_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model_kwargs: dict[str, Any] = dict(common_kwargs)
    model_kwargs["torch_dtype"] = _resolve_torch_dtype(torch, spec.dtype)
    if spec.device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(spec.base_model_id, **model_kwargs)

    adapter_model_id = spec.adapter_model_id.strip()
    if adapter_model_id:
        try:
            from peft import PeftModel  # type: ignore
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"missing_local_peft_dependency:{error}") from error

        adapter_kwargs: dict[str, Any] = {}
        if auth_token:
            adapter_kwargs["token"] = auth_token
        if spec.cache_dir:
            adapter_kwargs["cache_dir"] = spec.cache_dir
        if spec.trust_remote_code:
            adapter_kwargs["trust_remote_code"] = True

        model = PeftModel.from_pretrained(model, adapter_model_id, **adapter_kwargs)

    if spec.device != "auto":
        try:
            model.to(spec.device)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"local_model_move_failed:{error}") from error
    model.eval()

    _LOCAL_TRANSFORMERS_CACHE[cache_key] = (tokenizer, model)
    return tokenizer, model


def local_transformers_predict_vector(
    spec: LocalTransformersSpec,
    request_text: str,
    weather: str,
    hf_token: str,
) -> tuple[dict[str, int] | None, str | None]:
    if not spec.base_model_id.strip():
        return None, "local_base_model_missing"

    def parse_generated_text(text: str) -> tuple[dict[str, int] | None, str | None]:
        if not text:
            return None, "empty_generated_text"
        blob = extract_json_blob(text)
        if not blob:
            return None, "json_block_not_found"
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as error:
            return None, f"json_decode_error:{error}"
        if not isinstance(parsed, dict):
            return None, "parsed_payload_not_object"
        try:
            vector = sanitize_vector(parsed)
        except Exception as error:  # noqa: BLE001
            return None, f"invalid_vector:{error}"
        return vector, None

    prompt = "\n".join(
        [
            "You estimate 6 hidden music parameters.",
            "Return strict JSON only with schema:",
            '{"vector":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
            "Each vector value must be integer between 0 and 10.",
            f"request_text={request_text}",
        ]
    )

    try:
        tokenizer, model = _load_local_transformers_bundle(spec, hf_token)
    except Exception as error:  # noqa: BLE001
        return None, f"local_backend_init_error:{error}"

    try:
        import torch  # type: ignore
    except Exception as error:  # noqa: BLE001
        return None, f"missing_local_transformers_dependency:{error}"

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

    try:
        model_inputs = tokenizer(input_text, return_tensors="pt")
    except Exception as error:  # noqa: BLE001
        return None, f"local_tokenize_error:{error}"

    target_device: Any = None
    if spec.device != "auto":
        target_device = spec.device
    else:
        try:
            first_param = next(model.parameters())
            if str(first_param.device) != "meta":
                target_device = first_param.device
        except Exception:
            target_device = None

    if target_device is not None:
        try:
            model_inputs = {key: value.to(target_device) for key, value in model_inputs.items()}
        except Exception as error:  # noqa: BLE001
            return None, f"local_input_move_error:{error}"

    try:
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, int(spec.max_new_tokens)),
            "do_sample": False,
        }
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = int(tokenizer.eos_token_id)

        with torch.no_grad():
            output_ids = model.generate(**model_inputs, **generation_kwargs)

        input_len = int(model_inputs["input_ids"].shape[-1])
        if hasattr(output_ids, "__getitem__"):
            generated_ids = output_ids[0][input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        else:
            text = str(output_ids)
    except Exception as error:  # noqa: BLE001
        return None, f"local_generate_error:{error}"

    return parse_generated_text(text)


def hf_predict_vector(
    model_id: str,
    request_text: str,
    weather: str,
    token: str,
    backend_mode: str,
    text_generation_base_url: str,
    chat_completions_base_url: str,
    chat_completions_model_id: str,
    chat_completions_model_suffix: str,
    timeout_s: int = 90,
) -> tuple[dict[str, int] | None, str | None, str]:
    if not token:
        return None, "HF_TOKEN missing", "hf_not_configured"

    def build_prompt() -> str:
        return "\n".join(
            [
                "You estimate 6 hidden music parameters.",
                "Return strict JSON only with schema:",
                '{"vector":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
                "Each vector value must be integer between 0 and 10.",
                f"request_text={request_text}",
            ]
        )

    def parse_generated_text(text: str) -> tuple[dict[str, int] | None, str | None]:
        if not text:
            return None, "empty_generated_text"
        blob = extract_json_blob(text)
        if not blob:
            return None, "json_block_not_found"
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as error:
            return None, f"json_decode_error:{error}"
        if not isinstance(parsed, dict):
            return None, "parsed_payload_not_object"
        try:
            vector = sanitize_vector(parsed)
        except Exception as error:  # noqa: BLE001
            return None, f"invalid_vector:{error}"
        return vector, None

    def call_text_generation(prompt: str) -> tuple[dict[str, int] | None, str | None]:
        url = f"{text_generation_base_url.rstrip('/')}/{model_id}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 96,
                "temperature": 0.1,
                "return_full_text": False,
            },
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        except requests.RequestException as error:
            return None, f"inference_error:{error}"
        if response.status_code >= 400:
            return None, f"http_{response.status_code}:{response.text[:240]}"
        try:
            data = response.json()
        except json.JSONDecodeError as error:
            return None, f"non_json_response:{error}"

        text = ""
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                text = str(first.get("generated_text", "")).strip()
        elif isinstance(data, dict):
            text = str(data.get("generated_text", "")).strip()
        return parse_generated_text(text)

    def resolve_chat_model_id() -> str:
        base_model = chat_completions_model_id.strip() if chat_completions_model_id.strip() else model_id
        suffix = chat_completions_model_suffix.strip()
        if suffix and not base_model.endswith(suffix):
            return f"{base_model}{suffix}"
        return base_model

    def call_chat_completions(prompt: str) -> tuple[dict[str, int] | None, str | None]:
        url = f"{chat_completions_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": resolve_chat_model_id(),
            "temperature": 0.1,
            "max_tokens": 96,
            "messages": [
                {
                    "role": "system",
                    "content": "You output only strict JSON with six integer fields.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
        except requests.RequestException as error:
            return None, f"chat_request_error:{error}"
        if response.status_code >= 400:
            return None, f"chat_http_{response.status_code}:{response.text[:240]}"
        try:
            data = response.json()
        except json.JSONDecodeError as error:
            return None, f"chat_non_json_response:{error}"

        text = ""
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, list):
                        text_parts: list[str] = []
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") in {"text", "output_text"}:
                                    text_parts.append(str(part.get("text", "")))
                                elif "content" in part:
                                    text_parts.append(str(part.get("content", "")))
                        text = "".join(text_parts).strip()
                    else:
                        text = str(content).strip()
        return parse_generated_text(text)

    backend_normalized = backend_mode.strip().lower().replace("_", "-")
    if backend_normalized not in {"auto", "text-generation", "chat-completions"}:
        backend_normalized = "auto"

    prompt = build_prompt()
    attempt_errors: list[str] = []

    text_backend_label = (
        "hf_router_hf_inference"
        if "router.huggingface.co" in text_generation_base_url
        else "hf_text_generation"
    )
    chat_backend_label = (
        "hf_router_chat_completions"
        if "router.huggingface.co" in chat_completions_base_url
        else "hf_endpoint_chat_completions"
    )

    if backend_normalized in {"auto", "text-generation"}:
        vector, error = call_text_generation(prompt)
        if vector is not None:
            return vector, None, text_backend_label
        if error:
            attempt_errors.append(f"text_generation:{error}")
        if backend_normalized == "text-generation":
            joined = ";".join(attempt_errors) if attempt_errors else "text_generation_failed"
            return None, joined, text_backend_label

    if backend_normalized in {"auto", "chat-completions"}:
        vector, error = call_chat_completions(prompt)
        if vector is not None:
            return vector, None, chat_backend_label
        if error:
            attempt_errors.append(f"chat_completions:{error}")

    joined = ";".join(attempt_errors) if attempt_errors else "hf_predict_failed"
    default_backend = chat_backend_label if backend_normalized == "chat-completions" else text_backend_label
    return None, joined, default_backend


def should_try_mistral_fallback(parse_error: str | None) -> bool:
    if not parse_error:
        return False
    lowered = parse_error.lower()
    return (
        "http_404" in lowered
        or "http_410" in lowered
        or "no longer supported" in lowered
        or "model_not_supported" in lowered
    )


def mistral_predict_vector(
    model_id: str,
    request_text: str,
    weather: str,
    api_key: str,
    base_url: str,
    timeout_s: int = 90,
) -> tuple[dict[str, int] | None, str | None]:
    if not api_key:
        return None, "MISTRAL_API_KEY missing"
    if not model_id:
        return None, "MISTRAL model_id missing"

    prompt = "\n".join(
        [
            "You estimate 6 hidden music parameters.",
            "Return strict JSON only with schema:",
            '{"vector":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
            "Each vector value must be integer between 0 and 10.",
            f"request_text={request_text}",
        ]
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model_id,
        "temperature": 0.1,
        "max_tokens": 96,
        "messages": [
            {
                "role": "system",
                "content": "You output only strict JSON with six integer fields.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    except requests.RequestException as error:
        return None, f"mistral_request_error:{error}"

    if response.status_code >= 400:
        return None, f"mistral_http_{response.status_code}:{response.text[:240]}"

    try:
        data = response.json()
    except json.JSONDecodeError as error:
        return None, f"mistral_non_json_response:{error}"

    content: Any = ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content", "")

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in {"text", "output_text"}:
                    text_parts.append(str(part.get("text", "")))
                elif "content" in part:
                    text_parts.append(str(part.get("content", "")))
        text = "".join(text_parts).strip()
    else:
        text = str(content).strip()

    if not text:
        return None, "mistral_empty_content"

    blob = extract_json_blob(text)
    if not blob:
        return None, "mistral_json_block_not_found"

    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as error:
        return None, f"mistral_json_decode_error:{error}"

    if not isinstance(parsed, dict):
        return None, "mistral_parsed_payload_not_object"

    try:
        vector = sanitize_vector(parsed)
    except Exception as error:  # noqa: BLE001
        return None, f"mistral_invalid_vector:{error}"
    return vector, None


def weave_trace_url(entity: str | None, project: str, trace_id: str) -> str | None:
    if not project:
        return None
    if entity:
        return f"https://wandb.ai/{entity}/{project}/weave/traces?query={trace_id}"
    return f"https://wandb.ai/{project}/weave/traces?query={trace_id}"


@dataclass
class EvalConfig:
    mode: str
    dataset_path: Path
    dataset_repo_id: str
    dataset_config: str
    dataset_split: str
    dataset_max_samples: int
    target_scale: int
    prompt_model_id: str
    fine_tuned_model_id: str
    large_model_id: str
    hf_token: str
    hf_inference_backend: str
    hf_inference_base_url: str
    hf_openai_base_url: str
    hf_openai_model_id: str
    hf_openai_model_suffix: str
    local_base_model_id: str
    local_adapter_model_id: str
    local_fine_tuned_is_adapter: bool
    local_device: str
    local_dtype: str
    local_max_new_tokens: int
    local_trust_remote_code: bool
    local_cache_dir: str
    require_hf_direct: bool
    mistral_api_key: str
    mistral_base_url: str
    mistral_prompt_model_id: str
    mistral_fine_tuned_model_id: str
    mistral_large_model_id: str
    mistral_fallback_enabled: bool
    top_failures: int
    weave_project: str
    weave_enabled: bool
    wandb_enabled: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_run_group: str | None


def load_config(root: Path) -> EvalConfig:
    return EvalConfig(
        mode=env_str("EVAL_MODE", "prompt_baseline"),
        dataset_path=Path(env_str("EVAL_DATASET_PATH", str(root / "data/eval/frozen_eval_set.v1.json"))),
        dataset_repo_id=env_str("EVAL_DATASET_REPO_ID", env_str("HF_FT_DATASET_REPO_ID", "")),
        dataset_config=env_str("EVAL_DATASET_CONFIG", env_str("HF_FT_DATASET_CONFIG", "")),
        dataset_split=env_str("EVAL_DATASET_SPLIT", "test"),
        dataset_max_samples=env_int("EVAL_DATASET_MAX_SAMPLES", 0),
        target_scale=EVAL_TARGET_SCALE,
        prompt_model_id=env_str("EVAL_PROMPT_BASELINE_MODEL_ID", env_str("HF_BASE_MODEL_ID", "mistralai/Ministral-3-3B-Instruct-2512")),
        fine_tuned_model_id=env_str("EVAL_FINE_TUNED_MODEL_ID", env_str("HF_FT_OUTPUT_MODEL_ID", "")),
        large_model_id=env_str(
            "EVAL_LARGE_BASELINE_MODEL_ID",
            env_str("MISTRAL_LARGE_MODEL_ID", "mistral-large-latest"),
        ),
        hf_token=env_str("HF_TOKEN", env_str("HF_API_TOKEN", "")),
        hf_inference_backend=env_str("HF_INFERENCE_BACKEND", "auto"),
        hf_inference_base_url=env_str("HF_INFERENCE_BASE_URL", "https://router.huggingface.co/hf-inference/models"),
        hf_openai_base_url=env_str("HF_OPENAI_BASE_URL", "https://router.huggingface.co/v1"),
        hf_openai_model_id=env_str("HF_OPENAI_MODEL_ID", ""),
        hf_openai_model_suffix=env_str("HF_OPENAI_MODEL_SUFFIX", ""),
        local_base_model_id=env_str("EVAL_LOCAL_BASE_MODEL_ID", ""),
        local_adapter_model_id=env_str("EVAL_LOCAL_ADAPTER_MODEL_ID", ""),
        local_fine_tuned_is_adapter=env_bool("EVAL_LOCAL_FINE_TUNED_IS_ADAPTER", True),
        local_device=env_str("EVAL_LOCAL_DEVICE", "auto"),
        local_dtype=env_str("EVAL_LOCAL_DTYPE", "auto"),
        local_max_new_tokens=env_int("EVAL_LOCAL_MAX_NEW_TOKENS", 96),
        local_trust_remote_code=env_bool("EVAL_LOCAL_TRUST_REMOTE_CODE", False),
        local_cache_dir=env_str("EVAL_LOCAL_CACHE_DIR", ""),
        require_hf_direct=env_bool("EVAL_REQUIRE_HF_DIRECT", False),
        mistral_api_key=env_str("MISTRAL_API_KEY", ""),
        mistral_base_url=env_str("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
        mistral_prompt_model_id=env_str("EVAL_MISTRAL_PROMPT_MODEL_ID", env_str("MISTRAL_BASE_MODEL", "mistral-small-latest")),
        mistral_fine_tuned_model_id=env_str(
            "EVAL_MISTRAL_FINE_TUNED_MODEL_ID",
            env_str("MISTRAL_FINE_TUNED_MODEL_ID", env_str("MISTRAL_FT_MODEL_ID", "")),
        ),
        mistral_large_model_id=env_str(
            "EVAL_MISTRAL_LARGE_MODEL_ID",
            env_str("MISTRAL_LARGE_MODEL_ID", "mistral-large-latest"),
        ),
        mistral_fallback_enabled=env_bool("EVAL_MISTRAL_FALLBACK_ENABLED", True),
        top_failures=env_int("EVAL_TOP_FAILURES", 20),
        weave_project=env_str("WEAVE_PROJECT", "atelier-kotone-weave"),
        weave_enabled=env_bool("EVAL_WEAVE_ENABLED", True),
        wandb_enabled=env_bool("EVAL_WANDB_ENABLED", True),
        wandb_project=env_str("WANDB_PROJECT", "atelier-kotone-ft"),
        wandb_entity=os.getenv("WANDB_ENTITY") or None,
        wandb_run_group=os.getenv("WANDB_RUN_GROUP") or None,
    )


def ensure_mode(mode: str) -> str:
    allowed = {"rule_baseline", "prompt_baseline", "fine_tuned", "large_baseline"}
    if mode not in allowed:
        return "prompt_baseline"
    return mode


def cost_per_request(mode: str) -> float:
    env_key = f"EVAL_COST_PER_REQUEST_{mode.upper()}"
    return env_float(env_key, DEFAULT_COST_PER_REQUEST_USD.get(mode, 0.0))


def resolve_local_transformers_spec(mode: str, model_id: str, cfg: EvalConfig) -> LocalTransformersSpec:
    normalized_mode = ensure_mode(mode)
    base_model_id = model_id.strip()
    adapter_model_id = cfg.local_adapter_model_id.strip()

    if normalized_mode == "fine_tuned" and cfg.local_fine_tuned_is_adapter:
        adapter_model_id = cfg.local_adapter_model_id.strip() or cfg.fine_tuned_model_id.strip()
        base_model_id = cfg.local_base_model_id.strip() or cfg.prompt_model_id.strip() or base_model_id

    return LocalTransformersSpec(
        base_model_id=base_model_id,
        adapter_model_id=adapter_model_id,
        device=cfg.local_device.strip() or "auto",
        dtype=cfg.local_dtype.strip() or "auto",
        max_new_tokens=max(1, int(cfg.local_max_new_tokens)),
        trust_remote_code=cfg.local_trust_remote_code,
        cache_dir=cfg.local_cache_dir.strip() or None,
    )


def compute_r2(target_values: list[float], sq_errors: list[float]) -> float:
    if not target_values:
        return 0.0
    target_mean = mean(target_values)
    ss_tot = sum((value - target_mean) ** 2 for value in target_values)
    ss_res = sum(sq_errors)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def maybe_init_weave(enabled: bool, project: str):
    if not enabled:
        return None
    try:
        import weave

        weave.init(project)
        return weave
    except Exception:  # noqa: BLE001
        return None


def maybe_init_wandb(enabled: bool, project: str, entity: str | None, group: str | None, mode: str):
    if not enabled:
        return None
    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb

        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        run = wandb.init(
            project=project,
            entity=entity,
            group=group,
            job_type="eval",
            tags=["hackathon", "eval", mode, "weave"],
            config={
                "mode": mode,
                "dataset_path": env_str("EVAL_DATASET_PATH", ""),
                "dataset_repo_id": env_str("EVAL_DATASET_REPO_ID", env_str("HF_FT_DATASET_REPO_ID", "")),
                "dataset_split": env_str("EVAL_DATASET_SPLIT", "test"),
                "prompt_model_id": env_str("EVAL_PROMPT_BASELINE_MODEL_ID", ""),
                "fine_tuned_model_id": env_str("EVAL_FINE_TUNED_MODEL_ID", ""),
                "large_model_id": env_str("EVAL_LARGE_BASELINE_MODEL_ID", ""),
                "hf_inference_backend": normalize_backend_mode(env_str("HF_INFERENCE_BACKEND", "auto")),
                "eval_local_fine_tuned_is_adapter": env_bool("EVAL_LOCAL_FINE_TUNED_IS_ADAPTER", True),
                "target_scale": EVAL_TARGET_SCALE,
            },
        )
        return run
    except Exception:  # noqa: BLE001
        return None

def maybe_run_weave_evaluation(
    weave_module: Any,
    sample_rows: list[dict[str, Any]],
    mode: str,
    run_id: str,
) -> dict[str, Any] | None:
    if weave_module is None:
        return None

    dataset_rows: list[dict[str, Any]] = []
    predictions_by_request: dict[str, dict[str, Any]] = {}
    for row in sample_rows:
        request_text = str(row.get("request_text", "")).strip()
        target = row.get("target_vector")
        if not request_text or not isinstance(target, dict):
            continue
        dataset_rows.append(
            {
                "request_text": request_text,
                "target": target,
            }
        )
        predictions_by_request[request_text] = {
            "vector": row.get("predicted_vector") if isinstance(row.get("predicted_vector"), dict) else {},
            "json_valid": bool(row.get("json_valid", False)),
            "parse_error": row.get("parse_error"),
        }

    if not dataset_rows:
        return None

    @weave_module.op()
    def mae_scorer(output: dict, target: dict) -> dict:
        vector = output.get("vector", {}) if isinstance(output, dict) else {}
        abs_errors = {
            axis: abs(float(vector.get(axis, 0.0)) - float(target.get(axis, 0.0)))
            for axis in KEYS
        }
        return {
            "mae": mean(list(abs_errors.values())),
            "abs_errors": abs_errors,
        }

    @weave_module.op()
    def json_valid_scorer(output: dict) -> dict:
        is_valid = bool(output.get("json_valid")) if isinstance(output, dict) else False
        return {"json_valid": is_valid}

    class PrecomputedEvalModel(weave_module.Model):
        predictions: dict[str, dict[str, Any]]

        @weave_module.op()
        def predict(self, request_text: str) -> dict[str, Any]:
            return self.predictions.get(
                request_text,
                {"vector": {}, "json_valid": False, "parse_error": "missing_precomputed_prediction"},
            )

    try:
        evaluation = weave_module.Evaluation(
            dataset=dataset_rows,
            scorers=[mae_scorer, json_valid_scorer],
        )
        model = PrecomputedEvalModel(predictions=predictions_by_request)
        eval_result = evaluation.evaluate(model)
        if inspect.isawaitable(eval_result):
            eval_result = asyncio.run(eval_result)

        return {
            "mode": mode,
            "run_id": run_id,
            "dataset_size": len(dataset_rows),
            "result": eval_result,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "mode": mode,
            "run_id": run_id,
            "dataset_size": len(dataset_rows),
            "error": str(error),
        }


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    stripped = raw.strip()
    if not stripped:
        return []

    parsed: Any = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
        return [item for item in parsed["items"] if isinstance(item, dict)]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    items: list[dict[str, Any]] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        if isinstance(row, dict):
            items.append(row)
    return items


def resolve_runtime_root() -> Path:
    explicit = env_str("EVAL_PROJECT_ROOT", "")
    if explicit:
        return Path(explicit).expanduser().resolve()

    script_path = Path(__file__).resolve()
    if len(script_path.parents) >= 3:
        candidate = script_path.parents[2]
        if (candidate / "scripts").exists() and (candidate / "artifacts").exists():
            return candidate

    return Path.cwd().resolve()


def _as_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_target_hidden_params(raw_row: dict[str, Any]) -> dict[str, Any] | None:
    payload = _as_dict(raw_row.get("target_hidden_params"))
    if payload is None:
        payload = _as_dict(raw_row.get("hidden_params"))
    if payload is None:
        payload = _as_dict(raw_row.get("target_vector"))
    if payload is None:
        payload = _as_dict(raw_row.get("vector"))

    if payload is None:
        return None

    vector = payload.get("vector") if isinstance(payload.get("vector"), dict) else payload
    if not isinstance(vector, dict):
        return None
    if not all(key in vector for key in KEYS):
        return None

    cleaned: dict[str, int] = {}
    for key in KEYS:
        try:
            cleaned[key] = int(round(float(vector[key])))
        except (TypeError, ValueError):
            return None

    target_scale = raw_row.get("target_scale") or payload.get("target_scale") or EVAL_TARGET_SCALE
    try:
        normalized_target_scale = int(target_scale)
    except (TypeError, ValueError):
        normalized_target_scale = EVAL_TARGET_SCALE

    return {
        "vector": cleaned,
        "target_scale": normalized_target_scale,
    }


def load_eval_items_from_hf_dataset(
    repo_id: str,
    split: str,
    config_name: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    if not repo_id.strip():
        raise ValueError("EVAL_DATASET_REPO_ID is required when local dataset file is missing")

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"missing_hf_datasets_dependency:{error}") from error

    dataset = load_dataset(repo_id, config_name or None, split=split)
    items: list[dict[str, Any]] = []
    limit = max(0, int(max_samples))
    for idx, row_any in enumerate(dataset):
        if limit > 0 and len(items) >= limit:
            break
        if not isinstance(row_any, dict):
            continue
        row = row_any
        target_hidden_params = _build_target_hidden_params(row)
        if target_hidden_params is None:
            continue
        request_text = str(
            row.get("request_text")
            or row.get("request")
            or row.get("input")
            or ""
        ).strip()
        if not request_text:
            continue

        item: dict[str, Any] = {
            "id": row.get("id") or f"{split}_{idx:04d}",
            "request_text": request_text,
            "weather": str(row.get("weather") or "sunny"),
            "source_type": row.get("source_type") or "hf_dataset",
            "target_hidden_params": target_hidden_params,
            "target_scale": target_hidden_params.get("target_scale", EVAL_TARGET_SCALE),
        }
        items.append(item)

    return items


def main() -> None:
    root = resolve_runtime_root()
    cfg = load_config(root)
    mode = ensure_mode(cfg.mode)
    backend_mode = normalize_backend_mode(cfg.hf_inference_backend)

    if cfg.dataset_path.exists():
        items = load_eval_items(cfg.dataset_path)
        default_dataset_version = cfg.dataset_path.stem
    else:
        items = load_eval_items_from_hf_dataset(
            repo_id=cfg.dataset_repo_id,
            split=cfg.dataset_split,
            config_name=cfg.dataset_config,
            max_samples=cfg.dataset_max_samples,
        )
        default_dataset_version = f"{cfg.dataset_repo_id}:{cfg.dataset_split}"
    dataset_version = env_str("EVAL_DATASET_VERSION", default_dataset_version)

    if not isinstance(items, list) or len(items) == 0:
        raise ValueError(
            f"No eval items. dataset_path={cfg.dataset_path}, dataset_repo_id={cfg.dataset_repo_id}, split={cfg.dataset_split}"
        )

    model_source = mode
    if mode == "rule_baseline":
        model_id = "rule_based_keyword_v1"
    elif mode == "fine_tuned":
        model_id = cfg.fine_tuned_model_id or cfg.prompt_model_id
    elif mode == "large_baseline":
        model_id = cfg.large_model_id or cfg.prompt_model_id
    else:
        model_id = cfg.prompt_model_id

    weave = maybe_init_weave(cfg.weave_enabled, cfg.weave_project)
    wandb_run = maybe_init_wandb(cfg.wandb_enabled, cfg.wandb_project, cfg.wandb_entity, cfg.wandb_run_group, mode)

    if weave is not None:

        @weave.op()
        def traced_eval_call(payload: dict[str, Any]) -> dict[str, Any]:
            return payload

    else:

        def traced_eval_call(payload: dict[str, Any]) -> dict[str, Any]:
            return payload

    sample_rows: list[dict[str, Any]] = []
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    target_values: list[float] = []
    json_valid_flags: list[int] = []
    constraint_flags: list[int] = []
    slot_flags: list[int] = []
    intent_scores: list[float] = []
    sanity_scores: list[float] = []
    latencies: list[float] = []
    costs: list[float] = []

    for idx, item in enumerate(items):
        request_text = str(item.get("request_text", ""))
        weather = str(item.get("weather", "sunny"))
        raw_target_vector = item.get("target_hidden_params", {}).get("vector", {})
        source_scale = int(
            item.get("target_scale")
            or item.get("target_hidden_params", {}).get("target_scale")
            or infer_scale(raw_target_vector)
        )
        target_vector = normalize_to_eval_scale(raw_target_vector, source_scale)
        target_constraints = derive_constraints(target_vector)
        expected_top1 = derive_slot_top1(target_vector)

        trace_id = f"eval_{mode}_{uuid.uuid4().hex[:16]}"
        started = time.perf_counter()

        parse_error: str | None = None
        predicted_vector: dict[str, int] | None = None
        effective_model_id = model_id
        inference_backend = "rule_based"
        if mode == "rule_baseline":
            predicted_vector = rule_based_predict(request_text)
        else:
            if backend_mode == "local-transformers":
                local_spec = resolve_local_transformers_spec(mode, model_id, cfg)
                predicted_vector, parse_error = local_transformers_predict_vector(
                    local_spec,
                    request_text,
                    weather,
                    cfg.hf_token,
                )
                inference_backend = "local_transformers"
                if local_spec.adapter_model_id:
                    effective_model_id = f"{local_spec.base_model_id}+adapter:{local_spec.adapter_model_id}"
                else:
                    effective_model_id = local_spec.base_model_id
            else:
                predicted_vector, parse_error, inference_backend = hf_predict_vector(
                    model_id,
                    request_text,
                    weather,
                    cfg.hf_token,
                    backend_mode,
                    cfg.hf_inference_base_url,
                    cfg.hf_openai_base_url,
                    cfg.hf_openai_model_id,
                    cfg.hf_openai_model_suffix,
                )
            if (
                predicted_vector is None
                and backend_mode != "local-transformers"
                and not cfg.require_hf_direct
                and cfg.mistral_fallback_enabled
                and should_try_mistral_fallback(parse_error)
            ):
                fallback_model_id = cfg.mistral_prompt_model_id
                if mode == "fine_tuned":
                    fallback_model_id = cfg.mistral_fine_tuned_model_id or cfg.mistral_prompt_model_id
                elif mode == "large_baseline":
                    fallback_model_id = cfg.mistral_large_model_id or cfg.mistral_prompt_model_id
                fallback_vector, fallback_error = mistral_predict_vector(
                    fallback_model_id,
                    request_text,
                    weather,
                    cfg.mistral_api_key,
                    cfg.mistral_base_url,
                )
                if fallback_vector is not None:
                    predicted_vector = fallback_vector
                    parse_error = None
                    effective_model_id = fallback_model_id
                    inference_backend = "mistral_chat_fallback"
                elif fallback_error:
                    if parse_error:
                        parse_error = f"{parse_error};{fallback_error}"
                    else:
                        parse_error = fallback_error

        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        cost_usd = cost_per_request(mode)

        json_valid = predicted_vector is not None
        json_valid_flags.append(1 if json_valid else 0)
        latencies.append(elapsed_ms)
        costs.append(cost_usd)

        row: dict[str, Any] = {
            "id": item.get("id") or f"{mode}_{idx:04d}",
            "mode": mode,
            "scenario": item.get("scenario") or item.get("source_type") or "unknown",
            "source_type": item.get("source_type") or item.get("scenario", "unknown"),
            "request_text": request_text,
            "weather": weather,
            "model_source": model_source,
            "model_id": model_id,
            "effective_model_id": effective_model_id,
            "inference_backend": inference_backend,
            "trace_id": trace_id,
            "trace_url": weave_trace_url(cfg.wandb_entity, cfg.weave_project, trace_id),
            "json_valid": json_valid,
            "parse_error": parse_error,
            "latency_ms": round(elapsed_ms),
            "cost_usd": cost_usd,
            "target_vector": target_vector,
        }

        if not json_valid:
            row["output_sanity_score"] = score_output_sanity(False, parse_error, None)
            sanity_scores.append(float(row["output_sanity_score"]))
            traced_eval_call(
                {
                    "trace_id": trace_id,
                    "mode": mode,
                    "scenario": row["scenario"],
                    "request_text": request_text,
                    "json_valid": False,
                    "parse_error": parse_error,
                }
            )
            sample_rows.append(row)
            continue

        abs_error_by_dim: dict[str, float] = {}
        for key in KEYS:
            target_value = float(target_vector.get(key, 0))
            pred_value = float(predicted_vector[key])
            diff = pred_value - target_value
            abs_errors.append(abs(diff))
            sq_errors.append(diff * diff)
            target_values.append(target_value)
            abs_error_by_dim[key] = abs(diff)

        predicted_constraints = derive_constraints(predicted_vector)
        constraint_match = json.dumps(predicted_constraints, sort_keys=True) == json.dumps(target_constraints, sort_keys=True)
        constraint_flags.append(1 if constraint_match else 0)

        predicted_top1 = derive_slot_top1(predicted_vector)
        slot_hits = 0
        for slot in SLOTS:
            hit = predicted_top1.get(slot) == expected_top1.get(slot)
            slot_flags.append(1 if hit else 0)
            slot_hits += 1 if hit else 0

        intent_score = score_intent_from_error(abs_error_by_dim)
        output_sanity_score = score_output_sanity(True, None, predicted_vector)
        intent_scores.append(intent_score)
        sanity_scores.append(float(output_sanity_score))

        row.update(
            {
                "predicted_vector": predicted_vector,
                "abs_error_by_dim": abs_error_by_dim,
                "mae_raw": mean(list(abs_error_by_dim.values())),
                "constraint_match": constraint_match,
                "slot_exact_match": slot_hits / len(SLOTS),
                "intent_score": intent_score,
                "output_sanity_score": output_sanity_score,
            }
        )

        traced_eval_call(
            {
                "trace_id": trace_id,
                "mode": mode,
                "scenario": row["scenario"],
                "request_text": request_text,
                "predicted_vector": predicted_vector,
                "target_vector": target_vector,
                "abs_error_by_dim": abs_error_by_dim,
                "intent_score": intent_score,
            }
        )
        sample_rows.append(row)

    metrics = {
        "target_scale": cfg.target_scale,
        "json_valid_rate": mean(json_valid_flags),
        "vector_mae": mean(abs_errors),
        "mse_raw": mean(sq_errors),
        "mse_norm": mean(sq_errors) / (float(cfg.target_scale) * float(cfg.target_scale)),
        "r2_score": compute_r2(target_values, sq_errors),
        "constraint_match_rate": mean(constraint_flags),
        "slot_exact_match": mean(slot_flags),
        "intent_score_mean": mean(intent_scores),
        "output_sanity_score": mean(sanity_scores),
        "p95_inference_latency_ms": percentile(latencies, 95),
        "p50_inference_latency_ms": percentile(latencies, 50),
        "cost_per_100_requests_usd": mean(costs) * 100.0,
        "hf_direct_rate": mean([1.0 if str(row.get("inference_backend", "")).startswith("hf_") else 0.0 for row in sample_rows]),
        "local_transformers_rate": mean([1.0 if row.get("inference_backend") == "local_transformers" else 0.0 for row in sample_rows]),
        "mistral_fallback_rate": mean([1.0 if row.get("inference_backend") == "mistral_chat_fallback" else 0.0 for row in sample_rows]),
    }

    valid_rows = [row for row in sample_rows if row.get("json_valid") and isinstance(row.get("mae_raw"), (float, int))]
    failures_top_k = sorted(valid_rows, key=lambda x: float(x["mae_raw"]), reverse=True)[: max(1, cfg.top_failures)]

    run_id = f"{datetime.now(timezone.utc).isoformat().replace(':', '-').replace('.', '-')}_{mode}"
    runs_dir = root / "artifacts/eval/runs"
    samples_dir = root / "artifacts/eval/samples"
    runs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    weave_evaluation_summary = maybe_run_weave_evaluation(
        weave_module=weave,
        sample_rows=sample_rows,
        mode=mode,
        run_id=run_id,
    )

    run_payload = {
        "dataset_version": dataset_version,
        "dataset_path": str(cfg.dataset_path),
        "dataset_repo_id": cfg.dataset_repo_id,
        "dataset_split": cfg.dataset_split,
        "dataset_config": cfg.dataset_config,
        "target_scale": cfg.target_scale,
        "mode": mode,
        "model_source": model_source,
        "model_id": model_id,
        "hf_inference_backend": backend_mode,
        "hf_inference_base_url": cfg.hf_inference_base_url,
        "hf_openai_base_url": cfg.hf_openai_base_url,
        "hf_openai_model_id": cfg.hf_openai_model_id,
        "hf_openai_model_suffix": cfg.hf_openai_model_suffix,
        "eval_local_base_model_id": cfg.local_base_model_id,
        "eval_local_adapter_model_id": cfg.local_adapter_model_id,
        "eval_local_fine_tuned_is_adapter": cfg.local_fine_tuned_is_adapter,
        "eval_local_device": cfg.local_device,
        "eval_local_dtype": cfg.local_dtype,
        "eval_local_max_new_tokens": cfg.local_max_new_tokens,
        "require_hf_direct": cfg.require_hf_direct,
        "evaluated_count": len(items),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "weave_project": cfg.weave_project,
        "weave_evaluation_summary": weave_evaluation_summary,
    }
    sample_payload = {
        "run_id": run_id,
        "mode": mode,
        "target_scale": cfg.target_scale,
        "dataset_version": dataset_version,
        "model_source": model_source,
        "model_id": model_id,
        "rows": sample_rows,
        "failures_top_k": failures_top_k,
    }

    run_path = runs_dir / f"{run_id}.json"
    sample_path = samples_dir / f"{run_id}.json"
    run_path.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_path.write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if wandb_run is not None:
        try:
            import wandb

            wandb.log({f"eval/{key}": value for key, value in metrics.items()})
            wandb.log(
                {
                    "eval/evaluated_count": len(items),
                    "eval/model_source": model_source,
                    "eval/model_id": model_id,
                }
            )
            table = wandb.Table(columns=["id", "request_text", "mae_raw", "trace_id", "trace_url"])
            for row in failures_top_k:
                table.add_data(
                    row.get("id"),
                    row.get("request_text"),
                    row.get("mae_raw"),
                    row.get("trace_id"),
                    row.get("trace_url"),
                )
            wandb.log({"eval/failures_top_k": table})

            artifact = wandb.Artifact(f"eval-{mode}-{run_id}", type="eval_run")
            artifact.add_file(str(run_path))
            artifact.add_file(str(sample_path))
            wandb.log_artifact(artifact)
            wandb.finish()
        except Exception:
            pass

    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_path": str(run_path),
                "sample_path": str(sample_path),
                "model_id": model_id,
                "mode": mode,
                **metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
