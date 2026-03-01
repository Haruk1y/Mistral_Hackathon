#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "weave>=0.52.0",
# ]
# ///

"""Publish per-model comparison evals to Weave Evals."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_project_spec(project: str, entity: str | None) -> str:
    normalized = project.strip()
    if "/" in normalized:
        return normalized
    if entity:
        return f"{entity}/{normalized}"
    return normalized


def load_run_payload(root: Path, run_file: str) -> dict[str, Any]:
    run_path = root / "artifacts/eval/runs" / run_file
    if not run_path.exists():
        return {}
    try:
        payload = read_json(run_path)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def load_sample_payload(root: Path, sample_file: str) -> dict[str, Any]:
    sample_path = root / "artifacts/eval/samples" / sample_file
    if not sample_path.exists():
        return {}
    try:
        payload = read_json(sample_path)
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


def build_example_scores(sample_row: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}

    json_valid = bool(sample_row.get("json_valid") is True)
    scores["meta.json_valid"] = 1.0 if json_valid else 0.0

    latency_ms = safe_float(sample_row.get("latency_ms"))
    if latency_ms is not None:
        scores["meta.latency_ms"] = latency_ms

    output_sanity = safe_float(sample_row.get("output_sanity_score"))
    if output_sanity is not None:
        scores["meta.output_sanity"] = output_sanity

    constraint_match = sample_row.get("constraint_match")
    if isinstance(constraint_match, bool):
        scores["meta.constraint_match"] = 1.0 if constraint_match else 0.0

    slot_exact_match = safe_float(sample_row.get("slot_exact_match"))
    if slot_exact_match is not None:
        scores["meta.slot_exact_match"] = slot_exact_match

    intent_score = safe_float(sample_row.get("intent_score"))
    if intent_score is not None:
        scores["meta.intent_score"] = intent_score

    mae_raw = safe_float(sample_row.get("mae_raw"))
    if mae_raw is not None:
        scores["mae_score.mae"] = mae_raw

    abs_error = sample_row.get("abs_error_by_dim") if isinstance(sample_row.get("abs_error_by_dim"), dict) else {}
    for key in KEYS:
        value = safe_float(abs_error.get(key))
        if value is not None:
            scores[f"mae_score.abs_errors.{key}"] = value

    return scores


def build_eval_inputs(sample_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample_row.get("id"),
        "request_text": str(sample_row.get("request_text", "")),
        "source_type": str(sample_row.get("source_type") or sample_row.get("scenario") or "unknown"),
        "weather": str(sample_row.get("weather", "")),
    }


def build_eval_output(sample_row: dict[str, Any], mode: str, model_id: str, run_file: str, sample_file: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "model_id": model_id,
        "effective_model_id": sample_row.get("effective_model_id", model_id),
        "json_valid": bool(sample_row.get("json_valid") is True),
        "parse_error": sample_row.get("parse_error"),
        "predicted_vector": sample_row.get("predicted_vector"),
        "target_vector": sample_row.get("target_vector"),
        "trace_url": sample_row.get("trace_url"),
        "run_file": run_file,
        "sample_file": sample_file,
    }


def build_summary_scores(run_metrics: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    vector_mae = safe_float(run_metrics.get("vector_mae")) or 0.0
    p95_ms = safe_float(run_metrics.get("p95_inference_latency_ms")) or 0.0
    json_valid_rate = safe_float(run_metrics.get("json_valid_rate")) or 0.0
    output_sanity = safe_float(run_metrics.get("output_sanity_score")) or 0.0
    constraint_match = safe_float(run_metrics.get("constraint_match_rate")) or 0.0
    slot_exact = safe_float(run_metrics.get("slot_exact_match")) or 0.0
    intent_score = safe_float(run_metrics.get("intent_score_mean")) or 0.0

    scores["meta.json_valid"] = json_valid_rate
    scores["meta.latency_ms"] = p95_ms
    scores["meta.output_sanity"] = output_sanity
    scores["meta.constraint_match"] = constraint_match
    scores["meta.slot_exact_match"] = slot_exact
    scores["meta.intent_score"] = intent_score
    scores["mae_score.mae"] = vector_mae

    for key in KEYS:
        dim_key = f"mae_raw_{key}"
        dim_value = safe_float(run_metrics.get(dim_key))
        scores[f"mae_score.abs_errors.{key}"] = dim_value if dim_value is not None else vector_mae
    return scores


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    comparison_path = Path(
        env_str("EVAL_COMPARISON_JSON", str(root / "artifacts/eval/summary/test_model_comparison.json"))
    )
    radar_html_path = Path(
        env_str("EVAL_RADAR_HTML_PATH", str(root / "artifacts/eval/summary/test_model_radar.html"))
    )
    metadata_path = Path(
        env_str(
            "EVAL_COMPARISON_WEAVE_META_PATH",
            str(root / "artifacts/eval/summary/weave_eval_comparison.latest.json"),
        )
    )
    project = env_str("WEAVE_PROJECT", env_str("WANDB_PROJECT", ""))
    entity = os.getenv("WANDB_ENTITY") or None
    evaluation_name = env_str("WEAVE_EVAL_COMPARISON_NAME", "kotone-test-model-comparison")
    dataset_name = env_str("WEAVE_EVAL_COMPARISON_DATASET_NAME", "atelier-kotone-ft-test-comparison-v1")

    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY is required for Weave Evals publishing")
    if not project:
        raise ValueError("WEAVE_PROJECT or WANDB_PROJECT is required")
    if not comparison_path.exists():
        raise FileNotFoundError(f"comparison json not found: {comparison_path}")

    payload = read_json(comparison_path)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    compare_modes = payload.get("compare_modes") if isinstance(payload.get("compare_modes"), list) else []
    if len(rows) == 0:
        raise ValueError(f"comparison rows are empty: {comparison_path}")

    project_spec = resolve_project_spec(project, entity)
    weave_project_url = f"https://wandb.ai/{project_spec}/weave"

    import weave

    weave.init(project_spec)

    radar_rel = relative_or_absolute(radar_html_path, root)
    comparison_rel = relative_or_absolute(comparison_path, root)

    eval_runs: list[dict[str, Any]] = []
    total_examples = 0

    for row in rows:
        if not isinstance(row, dict):
            continue

        mode = str(row.get("mode", "unknown"))
        model_id = str(row.get("model_id", "unknown"))
        run_file = str(row.get("run_file", ""))
        sample_file = str(row.get("sample_file") or run_file)
        run_url = str(row.get("run_url", ""))

        sample_payload = load_sample_payload(root, sample_file)
        run_payload = load_run_payload(root, run_file)
        sample_rows = sample_payload.get("rows") if isinstance(sample_payload.get("rows"), list) else []
        run_metrics = run_payload.get("metrics") if isinstance(run_payload.get("metrics"), dict) else {}

        logger = weave.EvaluationLogger(
            name=evaluation_name,
            model={"name": model_id, "metadata": {"mode": mode}},
            dataset=dataset_name,
            eval_attributes={
                "kind": "model_comparison",
                "source": "artifacts/eval/summary/test_model_comparison.json",
                "generated_at": payload.get("generated_at"),
                "compare_modes": compare_modes,
                "mode": mode,
                "run_file": run_file,
                "sample_file": sample_file,
                "run_url": run_url,
            },
        )

        mode_examples = 0
        for sample_row_any in sample_rows:
            if not isinstance(sample_row_any, dict):
                continue
            sample_row = sample_row_any
            inputs = build_eval_inputs(sample_row)
            output = build_eval_output(sample_row, mode=mode, model_id=model_id, run_file=run_file, sample_file=sample_file)
            scores = build_example_scores(sample_row)
            logger.log_example(inputs=inputs, output=output, scores=scores)
            mode_examples += 1
            total_examples += 1

        if mode_examples == 0:
            # Fallback: publish one synthetic example built from run-level metrics.
            inputs = {
                "id": f"{mode}-summary",
                "request_text": f"{mode} summary-only comparison",
                "source_type": "summary_fallback",
                "weather": "",
            }
            output = {
                "mode": mode,
                "model_id": model_id,
                "effective_model_id": model_id,
                "json_valid": run_metrics.get("json_valid_rate"),
                "parse_error": "sample_rows_missing",
                "predicted_vector": None,
                "target_vector": None,
                "trace_url": run_url or None,
                "run_file": run_file,
                "sample_file": sample_file,
            }
            scores = build_summary_scores(run_metrics)
            logger.log_example(inputs=inputs, output=output, scores=scores)
            mode_examples += 1
            total_examples += 1

        view_lines = [
            f"# {mode} comparison context",
            "",
            f"- Comparison JSON: `{comparison_rel}`",
            f"- Radar HTML: `{radar_rel}`",
            f"- Model: `{model_id}`",
            f"- Sample file: `{sample_file}`",
            "- Open Weave project and use Evals tab Compare for radar/bar visual comparison.",
        ]
        logger.set_view(name="comparison_links", content="\n".join(view_lines), extension=".md")

        summary_output = {
            "mode": mode,
            "model_id": model_id,
            "run_file": run_file,
            "sample_file": sample_file,
            "run_url": run_url,
            "dataset_name": dataset_name,
            "comparison_json_path": comparison_rel,
            "radar_html_path": radar_rel,
            "radar_html_exists": radar_html_path.exists(),
            "examples_logged": mode_examples,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "run_metrics": run_metrics,
        }
        logger.log_summary(summary_output, auto_summarize=True)

        eval_runs.append(
            {
                "mode": mode,
                "model_id": model_id,
                "run_file": run_file,
                "sample_file": sample_file,
                "examples_logged": mode_examples,
                "ui_url": logger.ui_url,
            }
        )

    metadata = {
        "published_at": datetime.now(timezone.utc).isoformat(),
        "project": project_spec,
        "weave_project_url": weave_project_url,
        "evaluation_name": evaluation_name,
        "dataset_name": dataset_name,
        "rows_count_total": total_examples,
        "compare_modes": compare_modes,
        "comparison_json_path": comparison_rel,
        "radar_html_path": radar_rel,
        "eval_runs": eval_runs,
        "ui_url": weave_project_url,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
