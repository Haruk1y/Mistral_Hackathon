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
}


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


def clamp(value: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, value))


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


def sanitize_vector(payload: dict[str, Any]) -> dict[str, int]:
    vector: dict[str, int] = {}
    for key in KEYS:
        value = payload.get(key)
        if value is None:
            raise ValueError(f"missing key: {key}")
        vector[key] = int(round(clamp(float(value), 0, 100)))
    return vector


def derive_constraints(vector: dict[str, int]) -> dict[str, Any]:
    return {
        "preferredStyleTags": ["citypop_80s"] if vector["nostalgia"] > 60 else ["pop_2000s"] if vector["brightness"] > 65 else ["hiphop_90s"],
        "preferredGimmickTags": ["filter_rise"] if vector["energy"] > 60 else ["beat_mute"],
        "avoidPartIds": ["style_2000s_pop"] if vector["brightness"] < 22 else [],
    }


def derive_slot_top1(vector: dict[str, int]) -> dict[str, str]:
    style = "style_2000s_pop" if vector["brightness"] > 68 else "style_80s_citypop" if vector["nostalgia"] > 65 else "style_90s_hiphop"
    instrument = "inst_piano_upright" if vector["acousticness"] > 70 else "inst_soft_strings" if vector["warmth"] > 64 else "inst_analog_synth"
    mood = "mood_rain_ambience" if vector["brightness"] < 35 else "mood_sun_glow" if vector["energy"] > 62 else "mood_night_drive"
    gimmick = "gimmick_harmony_stack" if vector["complexity"] > 55 else "gimmick_filter_rise" if vector["energy"] > 62 else "gimmick_beat_mute"
    return {"style": style, "instrument": instrument, "mood": mood, "gimmick": gimmick}


def score_output_sanity(json_valid: bool, parse_error: str | None, vector: dict[str, int] | None) -> int:
    if not json_valid:
        return 12 if parse_error else 20
    if not vector:
        return 30
    spread = statistics.pstdev([vector[k] for k in KEYS]) if len(KEYS) > 1 else 0
    return int(round(clamp(72 + spread * 0.6, 0, 100)))


def score_intent_from_error(abs_error: dict[str, float]) -> float:
    sample_mae = mean([abs_error[k] for k in KEYS])
    return float(clamp(100.0 - sample_mae * 2.1, 0, 100))


def rule_based_predict(request_text: str) -> dict[str, int]:
    lower = request_text.lower()
    default = {"energy": 45, "warmth": 55, "brightness": 50, "acousticness": 65, "complexity": 38, "nostalgia": 60}
    profiles = [
        (["rain", "quiet", "evening", "night"], {"energy": 22, "warmth": 60, "brightness": 24, "acousticness": 72, "complexity": 32, "nostalgia": 72}),
        (["smile", "bright", "market", "sun"], {"energy": 75, "warmth": 58, "brightness": 82, "acousticness": 38, "complexity": 48, "nostalgia": 42}),
        (["focus", "study", "reading", "cafe"], {"energy": 34, "warmth": 54, "brightness": 44, "acousticness": 68, "complexity": 28, "nostalgia": 58}),
        (["memory", "old", "nostalgia", "retro"], {"energy": 40, "warmth": 74, "brightness": 50, "acousticness": 76, "complexity": 40, "nostalgia": 88}),
    ]
    for keywords, vector in profiles:
        if any(keyword in lower for keyword in keywords):
            return vector
    return default


def hf_predict_vector(
    model_id: str,
    request_text: str,
    weather: str,
    token: str,
    timeout_s: int = 90,
) -> tuple[dict[str, int] | None, str | None]:
    if not token:
        return None, "HF_TOKEN missing"

    prompt = "\n".join(
        [
            "You estimate 6 hidden music parameters.",
            "Return strict JSON only with numeric keys:",
            '{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}',
            "Values must be integers between 0 and 100.",
            f"weather={weather}",
            f"request_text={request_text}",
        ]
    )

    url = f"https://api-inference.huggingface.co/models/{model_id}"
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
    prompt_model_id: str
    fine_tuned_model_id: str
    hf_token: str
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
        prompt_model_id=env_str("EVAL_PROMPT_BASELINE_MODEL_ID", env_str("HF_BASE_MODEL_ID", "mistralai/Ministral-3-3B-Instruct-2512")),
        fine_tuned_model_id=env_str("EVAL_FINE_TUNED_MODEL_ID", env_str("HF_FT_OUTPUT_MODEL_ID", "")),
        hf_token=env_str("HF_TOKEN", env_str("HF_API_TOKEN", "")),
        top_failures=env_int("EVAL_TOP_FAILURES", 20),
        weave_project=env_str("WEAVE_PROJECT", "atelier-kotone-weave"),
        weave_enabled=env_bool("EVAL_WEAVE_ENABLED", True),
        wandb_enabled=env_bool("EVAL_WANDB_ENABLED", True),
        wandb_project=env_str("WANDB_PROJECT", "atelier-kotone-ft"),
        wandb_entity=os.getenv("WANDB_ENTITY") or None,
        wandb_run_group=os.getenv("WANDB_RUN_GROUP") or None,
    )


def ensure_mode(mode: str) -> str:
    allowed = {"rule_baseline", "prompt_baseline", "fine_tuned"}
    if mode not in allowed:
        return "prompt_baseline"
    return mode


def cost_per_request(mode: str) -> float:
    env_key = f"EVAL_COST_PER_REQUEST_{mode.upper()}"
    return env_float(env_key, DEFAULT_COST_PER_REQUEST_USD.get(mode, 0.0))


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
                "prompt_model_id": env_str("EVAL_PROMPT_BASELINE_MODEL_ID", ""),
                "fine_tuned_model_id": env_str("EVAL_FINE_TUNED_MODEL_ID", ""),
            },
        )
        return run
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root)
    mode = ensure_mode(cfg.mode)

    raw = cfg.dataset_path.read_text(encoding="utf-8")
    dataset = json.loads(raw)
    items = dataset.get("items", [])

    if not isinstance(items, list) or len(items) == 0:
        raise ValueError(f"No eval items in dataset: {cfg.dataset_path}")

    model_source = mode
    if mode == "rule_baseline":
        model_id = "rule_based_keyword_v1"
    elif mode == "fine_tuned":
        model_id = cfg.fine_tuned_model_id or cfg.prompt_model_id
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

    for item in items:
        request_text = str(item.get("request_text", ""))
        weather = str(item.get("weather", "sunny"))
        target_vector = item.get("target_hidden_params", {}).get("vector", {})
        target_constraints = item.get("target_hidden_params", {}).get("constraints", {})
        expected_top1 = item.get("expected_top1_by_slot", {})

        trace_id = f"eval_{mode}_{uuid.uuid4().hex[:16]}"
        started = time.perf_counter()

        parse_error: str | None = None
        predicted_vector: dict[str, int] | None = None
        if mode == "rule_baseline":
            predicted_vector = rule_based_predict(request_text)
        else:
            predicted_vector, parse_error = hf_predict_vector(model_id, request_text, weather, cfg.hf_token)

        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        cost_usd = cost_per_request(mode)

        json_valid = predicted_vector is not None
        json_valid_flags.append(1 if json_valid else 0)
        latencies.append(elapsed_ms)
        costs.append(cost_usd)

        row: dict[str, Any] = {
            "id": item.get("id"),
            "mode": mode,
            "scenario": item.get("scenario"),
            "source_type": item.get("scenario", "unknown"),
            "request_text": request_text,
            "weather": weather,
            "model_source": model_source,
            "model_id": model_id,
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
        "json_valid_rate": mean(json_valid_flags),
        "vector_mae": mean(abs_errors),
        "mse_raw": mean(sq_errors),
        "mse_norm": mean(sq_errors) / (100.0 * 100.0),
        "r2_score": compute_r2(target_values, sq_errors),
        "constraint_match_rate": mean(constraint_flags),
        "slot_exact_match": mean(slot_flags),
        "intent_score_mean": mean(intent_scores),
        "output_sanity_score": mean(sanity_scores),
        "p95_inference_latency_ms": percentile(latencies, 95),
        "p50_inference_latency_ms": percentile(latencies, 50),
        "cost_per_100_requests_usd": mean(costs) * 100.0,
    }

    valid_rows = [row for row in sample_rows if row.get("json_valid") and isinstance(row.get("mae_raw"), (float, int))]
    failures_top_k = sorted(valid_rows, key=lambda x: float(x["mae_raw"]), reverse=True)[: max(1, cfg.top_failures)]

    run_id = f"{datetime.now(timezone.utc).isoformat().replace(':', '-').replace('.', '-')}_{mode}"
    runs_dir = root / "artifacts/eval/runs"
    samples_dir = root / "artifacts/eval/samples"
    runs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    run_payload = {
        "dataset_version": dataset.get("dataset_version", "unknown"),
        "mode": mode,
        "model_source": model_source,
        "model_id": model_id,
        "evaluated_count": len(items),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "weave_project": cfg.weave_project,
    }
    sample_payload = {
        "run_id": run_id,
        "mode": mode,
        "dataset_version": dataset.get("dataset_version", "unknown"),
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
