# W&B Report Outline (Hackathon Submission)

最終更新: 2026-02-28

## 1. Problem / Goal

- Task: `request_text -> hidden_params` 推定の精度改善
- Why FT: prompt-only baselineを上回る必要があるため
- Constraints: 4 slots (`style`, `instrument`, `mood`, `gimmick`), p95 latency <= 1200ms

## 2. Setup

- Dataset version: `frozen_eval_set.v1`
- Base model (HF): `mistralai/Ministral-3-3B-Instruct-2512` (main)
- Compare model: `mistralai/Ministral-8B-Instruct-*` or managed FT baseline
- Training infra: `HF Jobs` (`a10g-small` with 20 USD credits)
- Baselines:
  - rule baseline
  - prompt baseline

## 3. Metrics (11 required)

1. `json_valid_rate`
2. `vector_mae`
3. `mse_raw`
4. `mse_norm`
5. `constraint_match_rate`
6. `slot_exact_match`
7. `intent_score_mean`
8. `output_sanity_score`
9. `auto_improvement_delta`
10. `loop_completion_rate`
11. `p95_inference_latency_ms`
12. `cost_per_100_requests_usd`

## 4. Experiments

- Run table:
  - run id
  - HF Job id
  - model
  - hparams
  - dataset version
  - core metrics
- Sweep highlights:
  - best/worst runs and reason

## 4.1 HF Jobs Cost Snapshot

- flavor / runtime / estimated cost
- run count vs remaining credits
- cost_per_100_requests_usd trend

## 5. Weave Traces

- Trace groups:
  - commission generation
  - interpreter
  - score
- Failure case links (3+):
  - case A
  - case B
  - case C

## 6. Self-Improvement Loop

- Loop definition: `eval -> analysis -> update -> re-eval`
- Cycle logs:
  - cycle 1 delta
  - cycle 2 delta
- Evidence:
  - MCP query logs (`mcp_eval_snapshot.json`, `mcp_decision_input.json`)
  - generated configs / artifacts (`hparam_patch.yaml`, `augmentation_spec.json`, `before_after_metrics.csv`)

## 7. Skills 활용 / Best Skills Award Section

- Used skills:
  - `weights-and-biases`
  - `firebase-*`
- Generated assets:
  - `next_hparams.yaml`
  - `augmentation_spec.json`
  - `before_after_metrics.csv`
- Repro steps and automation scope

## 8. Latency / Cost Viability

- p50/p95 latency
- cost per 100 requests
- FT vs prompt-only tradeoff

## 9. Conclusion

- What improved
- What remains
- Next iteration

## 10. Link Checklist

- W&B project URL:
- W&B runs URL:
- W&B sweep URL:
- W&B report URL:
- Weave trace board URL:
- HF model/artifact URL:
- HF jobs URL:
- HF dataset URL:
- Repo commit/PR URL:
