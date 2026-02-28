#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "wandb>=0.18.0",
# ]
# ///

"""Collect validation metrics from W&B campaign runs.

Outputs:
- artifacts/hf_jobs/<campaign>_validation_snapshot.json
- artifacts/hf_jobs/<campaign>_validation_snapshot.csv
- artifacts/hf_jobs/<campaign>_hard_dims.json
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb

KEYS = ("energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia")


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    entity: str
    project: str
    campaign: str
    top_dims: int
    out_json: Path
    out_csv: Path
    out_dims: Path
    debug_file_scan: bool


def load_config(root: Path) -> Config:
    campaign = env_str("BALANCED_CAMPAIGN_NAME", "balanced_6run")
    out_dir = root / "artifacts" / "hf_jobs"
    out_prefix = env_str("CAMPAIGN_VALIDATION_OUT_PREFIX", campaign)
    return Config(
        entity=env_str("WANDB_ENTITY", "haruk1y_"),
        project=env_str("WANDB_PROJECT", "atelier-kotone-ft"),
        campaign=campaign,
        top_dims=env_int("CAMPAIGN_HARD_DIM_TOP_K", 2),
        out_json=out_dir / f"{out_prefix}_validation_snapshot.json",
        out_csv=out_dir / f"{out_prefix}_validation_snapshot.csv",
        out_dims=out_dir / f"{out_prefix}_hard_dims.json",
        debug_file_scan=env_bool("CAMPAIGN_DEBUG_FILE_SCAN", False),
    )


def summary_metric(summary: dict[str, Any], key: str) -> float | None:
    eval_key = f"eval/{key}"
    iter_key = f"iter_eval/{key}"
    value = summary.get(eval_key)
    if isinstance(value, (int, float)):
        return float(value)
    value = summary.get(iter_key)
    if isinstance(value, (int, float)):
        return float(value)
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def latest_iter_metrics(run: Any) -> dict[str, float]:
    keys = ["iter_eval/step", "iter_eval/mae_raw", "iter_eval/mse_raw"]
    keys.extend([f"iter_eval/mae_raw_{key}" for key in KEYS])
    latest: dict[str, float] = {}
    try:
        for row in run.scan_history(keys=keys, page_size=500):
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("iter_eval/mae_raw"), (int, float)):
                latest["mae_raw"] = float(row["iter_eval/mae_raw"])
                if isinstance(row.get("iter_eval/mse_raw"), (int, float)):
                    latest["mse_raw"] = float(row["iter_eval/mse_raw"])
                for key in KEYS:
                    dim_key = f"iter_eval/mae_raw_{key}"
                    if isinstance(row.get(dim_key), (int, float)):
                        latest[f"mae_raw_{key}"] = float(row[dim_key])
    except Exception:
        return latest
    return latest


def run_matches_campaign(run: Any, campaign: str) -> bool:
    group = str(run.group or "")
    name = str(run.name or "")
    if group.startswith(campaign):
        return True
    return name.startswith("balanced-run")


def rank_dims(metric_row: dict[str, float]) -> list[dict[str, float]]:
    ranked = []
    for key in KEYS:
        metric_key = f"mae_raw_{key}"
        value = metric_row.get(metric_key)
        if isinstance(value, (int, float)):
            ranked.append({"dim": key, "mae_raw": float(value)})
    ranked.sort(key=lambda x: x["mae_raw"], reverse=True)
    return ranked


def run_attr(run: Any, snake_name: str, camel_name: str) -> Any:
    value = getattr(run, snake_name, None)
    if value is not None:
        return value
    attrs = getattr(run, "_attrs", {})
    if isinstance(attrs, dict):
        return attrs.get(camel_name)
    return None


def safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def hard_case_dim_mae_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    sums = {key: 0.0 for key in KEYS}
    counts = {key: 0 for key in KEYS}
    for row in rows:
        by_dim = row.get("abs_error_raw")
        if not isinstance(by_dim, dict):
            continue
        for key in KEYS:
            value = by_dim.get(key)
            if isinstance(value, (int, float)):
                sums[key] += float(value)
                counts[key] += 1
    output: dict[str, float] = {}
    for key in KEYS:
        if counts[key] > 0:
            output[key] = sums[key] / counts[key]
    return output


def fetch_hard_case_stats(run: Any) -> dict[str, Any]:
    # Prefer per-run artifact that includes hard_cases.valid.jsonl.
    for artifact in run.logged_artifacts():
        names = []
        try:
            names = [file.name for file in artifact.files()]
        except Exception:
            names = []
        if "hard_cases.valid.jsonl" not in names:
            continue
        with tempfile.TemporaryDirectory(prefix="campaign_hard_cases_") as tmp:
            try:
                downloaded = artifact.download(root=tmp)
                artifact_dir = (
                    Path(downloaded)
                    if isinstance(downloaded, (str, os.PathLike))
                    else Path(tmp)
                )
                local_path = artifact_dir / "hard_cases.valid.jsonl"
                if not local_path.exists():
                    matches = list(artifact_dir.rglob("hard_cases.valid.jsonl"))
                    if not matches:
                        continue
                    local_path = matches[0]
            except Exception:
                continue
            rows = safe_read_jsonl(local_path)
            return {
                "artifact_name": artifact.name,
                "artifact_type": artifact.type,
                "artifact_url": getattr(artifact, "url", None),
                "hard_cases_count": len(rows),
                "hard_case_dim_mae_raw": hard_case_dim_mae_from_rows(rows),
            }

    # Fallback to run files in case hard_cases is uploaded as regular output.
    try:
        files = list(run.files())
    except Exception:
        files = []
    for wb_file in files:
        file_name = str(getattr(wb_file, "name", ""))
        if not file_name.endswith("hard_cases.valid.jsonl"):
            continue
        with tempfile.TemporaryDirectory(prefix="campaign_hard_cases_file_") as tmp:
            try:
                downloaded = wb_file.download(root=tmp, replace=True)
                if isinstance(downloaded, (str, os.PathLike)):
                    local_path = Path(downloaded)
                else:
                    local_name = getattr(downloaded, "name", None)
                    if isinstance(local_name, str) and local_name:
                        local_path = Path(local_name)
                    else:
                        local_path = Path(tmp) / Path(file_name).name
            except Exception:
                continue
            rows = safe_read_jsonl(local_path)
            return {
                "artifact_name": file_name,
                "artifact_type": "run_file",
                "artifact_url": getattr(wb_file, "url", None),
                "hard_cases_count": len(rows),
                "hard_case_dim_mae_raw": hard_case_dim_mae_from_rows(rows),
            }

    return {
        "artifact_name": None,
        "artifact_type": None,
        "artifact_url": None,
        "hard_cases_count": 0,
        "hard_case_dim_mae_raw": {},
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root)
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = list(api.runs(f"{cfg.entity}/{cfg.project}"))

    rows: list[dict[str, Any]] = []
    debug_printed = False
    for run in runs:
        if not run_matches_campaign(run, cfg.campaign):
            continue
        if cfg.debug_file_scan and not debug_printed:
            debug_printed = True
            try:
                file_names = [f.name for f in run.files()]
            except Exception as exc:
                file_names = [f"<files_error:{exc}>"]
            try:
                artifact_names = []
                artifact_file_map: dict[str, list[str]] = {}
                for artifact in run.logged_artifacts():
                    label = f"{artifact.name} ({artifact.type})"
                    artifact_names.append(label)
                    try:
                        artifact_file_map[label] = [file.name for file in artifact.files()]
                    except Exception as exc:
                        artifact_file_map[label] = [f"<artifact_files_error:{exc}>"]
            except Exception as exc:
                artifact_names = [f"<artifacts_error:{exc}>"]
                artifact_file_map = {}
            print(
                json.dumps(
                    {
                        "debug_run_id": run.id,
                        "debug_run_name": run.name,
                        "debug_file_names": file_names[:200],
                        "debug_logged_artifacts": artifact_names[:200],
                        "debug_artifact_files": artifact_file_map,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

        summary = dict(run.summary)

        metrics: dict[str, float] = {}
        direct_mae = summary_metric(summary, "mae_raw")
        direct_mse = summary_metric(summary, "mse_raw")
        if isinstance(direct_mae, float):
            metrics["mae_raw"] = direct_mae
        if isinstance(direct_mse, float):
            metrics["mse_raw"] = direct_mse

        for key in KEYS:
            value = summary_metric(summary, f"mae_raw_{key}")
            if isinstance(value, float):
                metrics[f"mae_raw_{key}"] = value

        # If summary is not fully populated, fill from latest iter_eval history.
        if "mae_raw" not in metrics or any(f"mae_raw_{key}" not in metrics for key in KEYS):
            latest = latest_iter_metrics(run)
            for metric_key, value in latest.items():
                metrics.setdefault(metric_key, value)

        hard_case_stats = fetch_hard_case_stats(run)
        ranked_dims = rank_dims(metrics)
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "group": run.group,
                "state": run.state,
                "url": run.url,
                "created_at": run_attr(run, "created_at", "createdAt"),
                "updated_at": run_attr(run, "updated_at", "updatedAt"),
                "metrics": metrics,
                "hard_case_stats": hard_case_stats,
                "top_dims": ranked_dims[: cfg.top_dims],
            }
        )

    rows.sort(key=lambda x: str(x.get("created_at") or ""))

    # Aggregate per-dimension MAE across runs with available metrics.
    dim_sums = {key: 0.0 for key in KEYS}
    dim_counts = {key: 0 for key in KEYS}
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key in KEYS:
            value = metrics.get(f"mae_raw_{key}")
            if isinstance(value, (int, float)):
                dim_sums[key] += float(value)
                dim_counts[key] += 1

    dim_rank = []
    for key in KEYS:
        if dim_counts[key] <= 0:
            continue
        dim_rank.append(
            {
                "dim": key,
                "mean_mae_raw": dim_sums[key] / dim_counts[key],
                "count": dim_counts[key],
            }
        )
    dim_rank.sort(key=lambda x: x["mean_mae_raw"], reverse=True)

    # Aggregate hard-case dimensional MAE if artifacts are available.
    hard_sums = {key: 0.0 for key in KEYS}
    hard_counts = {key: 0 for key in KEYS}
    for row in rows:
        hard_stats = row.get("hard_case_stats", {})
        if not isinstance(hard_stats, dict):
            continue
        by_dim = hard_stats.get("hard_case_dim_mae_raw", {})
        if not isinstance(by_dim, dict):
            continue
        for key in KEYS:
            value = by_dim.get(key)
            if isinstance(value, (int, float)):
                hard_sums[key] += float(value)
                hard_counts[key] += 1

    hard_dim_rank = []
    for key in KEYS:
        if hard_counts[key] <= 0:
            continue
        hard_dim_rank.append(
            {
                "dim": key,
                "mean_mae_raw": hard_sums[key] / hard_counts[key],
                "count": hard_counts[key],
            }
        )
    hard_dim_rank.sort(key=lambda x: x["mean_mae_raw"], reverse=True)
    recommended = [row["dim"] for row in (hard_dim_rank if hard_dim_rank else dim_rank)[: cfg.top_dims]]

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entity": cfg.entity,
        "project": cfg.project,
        "campaign": cfg.campaign,
        "run_count": len(rows),
        "runs": rows,
        "aggregate_dim_mae_raw": dim_rank,
        "aggregate_hard_case_dim_mae_raw": hard_dim_rank,
        "recommended_focus_dims": recommended,
    }

    cfg.out_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "run_id",
        "run_name",
        "group",
        "state",
        "url",
        "created_at",
        "updated_at",
        "mae_raw",
        "mse_raw",
    ] + [f"mae_raw_{key}" for key in KEYS]
    with cfg.out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            payload = {key: row.get(key) for key in csv_fields}
            if isinstance(metrics, dict):
                for metric_key, value in metrics.items():
                    if metric_key in csv_fields:
                        payload[metric_key] = value
            writer.writerow(payload)

    dims_payload = {
        "generated_at": snapshot["generated_at"],
        "campaign": cfg.campaign,
        "recommended_focus_dims": snapshot["recommended_focus_dims"],
        "aggregate_dim_mae_raw": dim_rank,
        "aggregate_hard_case_dim_mae_raw": hard_dim_rank,
        "source_snapshot": str(cfg.out_json),
    }
    cfg.out_dims.write_text(json.dumps(dims_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_count": len(rows),
                "focus_dims": dims_payload["recommended_focus_dims"],
                "snapshot_path": str(cfg.out_json),
                "csv_path": str(cfg.out_csv),
                "dims_path": str(cfg.out_dims),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
