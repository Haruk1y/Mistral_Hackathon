#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "wandb>=0.18.0",
# ]
# ///

"""Fallback context fetcher for self-improvement loop.

Reads local eval artifacts and optionally enriches with W&B API run metadata.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_fallback_snapshot(root: Path, top_k: int) -> dict[str, Any]:
    summary_path = root / "artifacts/eval/summary/latest_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    latest_ft = summary.get("latest_by_mode", {}).get("fine_tuned", {})
    run_file = latest_ft.get("file")
    run_id = run_file.replace(".json", "") if isinstance(run_file, str) else None
    sample_path = root / "artifacts/eval/samples" / f"{run_id}.json" if run_id else None

    failures_top_k: list[dict[str, Any]] = []
    if sample_path and sample_path.exists():
        sample_payload = read_json(sample_path)
        failures = sample_payload.get("failures_top_k")
        if isinstance(failures, list):
            failures_top_k = failures[:top_k]
        else:
            rows = sample_payload.get("rows", [])
            valid = [row for row in rows if row.get("json_valid") and isinstance(row.get("mae_raw"), (int, float))]
            valid.sort(key=lambda x: float(x["mae_raw"]), reverse=True)
            failures_top_k = valid[:top_k]

    return {
        "source": "local_summary_fallback",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "eval_summary": summary,
        "failures_top_k": failures_top_k,
        "recent_runs": [],
        "snapshot_from_run_id": run_id,
        "snapshot_from_sample_path": str(sample_path) if sample_path else None,
    }


def enrich_with_wandb_api(snapshot: dict[str, Any], top_k: int) -> dict[str, Any]:
    if not os.getenv("WANDB_API_KEY"):
        return snapshot

    try:
        import wandb
    except Exception:  # noqa: BLE001
        return snapshot

    try:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        api = wandb.Api(timeout=30)
    except Exception:  # noqa: BLE001
        return snapshot

    entity = os.getenv("WANDB_ENTITY")
    project = env_str("WANDB_PROJECT", "atelier-kotone-ft")
    if entity:
        path = f"{entity}/{project}"
    else:
        # fallback to current default entity
        default_entity = getattr(api.viewer, "entity", None)
        path = f"{default_entity}/{project}" if default_entity else project

    try:
        runs = list(api.runs(path=path, per_page=20, order="-created_at"))
    except Exception:  # noqa: BLE001
        return snapshot

    recent_runs: list[dict[str, Any]] = []
    for run in runs[:10]:
        cfg = dict(run.config or {})
        summary = dict(run.summary or {})
        recent_runs.append(
            {
                "run_id": run.id,
                "name": run.name,
                "url": run.url,
                "state": run.state,
                "created_at": run.created_at,
                "job_type": run.job_type,
                "group": run.group,
                "config": {
                    "learning_rate": cfg.get("learning_rate") or cfg.get("HF_FT_LR"),
                    "lora_r": cfg.get("lora_r") or cfg.get("HF_FT_LORA_R"),
                    "lora_alpha": cfg.get("lora_alpha") or cfg.get("HF_FT_LORA_ALPHA"),
                    "lora_dropout": cfg.get("lora_dropout") or cfg.get("HF_FT_LORA_DROPOUT"),
                    "epochs": cfg.get("epochs") or cfg.get("HF_FT_EPOCHS"),
                    "dataset_version": cfg.get("dataset_version"),
                },
                "summary": {
                    "eval/mse_norm": summary.get("eval/mse_norm"),
                    "eval/mae_raw": summary.get("eval/mae_raw"),
                    "eval/json_valid_rate": summary.get("eval/json_valid_rate"),
                    "objective/train_loss": summary.get("objective/train_loss"),
                },
            }
        )

    out = dict(snapshot)
    out["source"] = "wandb_api_fallback"
    out["recent_runs"] = recent_runs
    out["recent_run_count"] = len(recent_runs)
    out["top_k"] = top_k
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    top_k = env_int("WANDB_MCP_TOP_K", 20)
    out_path = Path(
        env_str(
            "WANDB_MCP_SNAPSHOT_PATH",
            str(root / "artifacts/loop/cycle_1/mcp_eval_snapshot.json"),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = local_fallback_snapshot(root, top_k)
    snapshot = enrich_with_wandb_api(snapshot, top_k)
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_path": str(out_path), "source": snapshot.get("source"), "recent_run_count": snapshot.get("recent_run_count", 0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
