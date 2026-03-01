# W&B Report Outline (Hackathon Submission)

Last updated: 2026-03-01

## 1. Goal

- Task: `request_text -> hidden_params` regression (`energy`, `warmth`, `brightness`, `acousticness`, `complexity`, `nostalgia`)
- Objective: improve quality while keeping latency practical
- Constraint: reproducible evidence with W&B + Weave + MCP links

## 2. Setup

- Base model: `mistralai/Ministral-3-3B-Instruct-2512`
- FT model repo: `Haruk1y/atelier-kotone-ministral3b-ft`
- Dataset repo: `Haruk1y/atelier-kotone-ft-request-hidden`
- Eval set: `data/eval/test_eval_set.from_ft_test.v1.json` (100 samples)
- Weave comparison eval name: `kotone-test-model-comparison-v3`
- W&B project: `haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301`

## 3. Latest Results (v3)

| mode | model_id | json_valid_rate | vector_mae | mse_norm | p95_latency_ms |
|---|---|---:|---:|---:|---:|
| prompt_baseline | mistralai/Ministral-3-3B-Instruct-2512 | 0.89 | 2.0581 | 0.0760 | 1472.80 |
| fine_tuned | Haruk1y/atelier-kotone-ministral3b-ft | 0.90 | 2.0944 | 0.0789 | 1290.12 |
| large_baseline | mistral-large-latest | 0.83 | 1.8775 | 0.0650 | 2738.71 |

Prompt vs FT delta:
- `json_valid_rate`: `+0.0100`
- `vector_mae`: `+0.0364` (FT worse)
- `mse_norm`: `+0.0029` (FT worse)
- `p95_latency_ms`: `-182.68` (FT faster)

## 4. Critical Interpretation Note

The current v3 comparison is not a clean FT-vs-base isolation.

Observed from sample logs:
- prompt_baseline backend mix: `mistral_chat_fallback=89`, `hf_router_hf_inference=11`
- fine_tuned backend mix: `mistral_chat_fallback=90`, `hf_router_hf_inference=10`
- valid predictions are effectively from fallback `ministral-3b-latest`
- invalid rows are mostly `rule_prompt` with `http_404`

Implication:
- reported MAE gap should be treated as monitoring signal, not final FT quality proof.

## 5. MCP Evidence (Cycle 4)

Paths:
- `artifacts/loop/cycle_4/mcp_eval_snapshot.json`
- `artifacts/loop/cycle_4/mcp_decision_input.json`
- `artifacts/loop/cycle_4/decision_log.md`

MCP source compliance:
- snapshot source is `wandb_mcp_tools_call`
- MCP tools path uses hosted endpoint and `tools/list + tools/call`

Training signal from MCP recent run (`p11k5s51`):
- `objective/train_loss = 19.5289`
- `eval/json_valid_rate = 0.16`
- `eval/mae_raw = 2.7183`

Failure concentration from Weave:
- top dims: `complexity`, `nostalgia`
- playbook: `artifacts/wandb/weave_failure_playbook.md`

## 6. Self-Improvement Plan (Submission Version)

### Step A: Fix evaluation validity

- Evaluate FT without fallback mixing (local PEFT path or merged model)
- Publish clean comparison as `kotone-test-model-comparison-v4`
- Keep validation/test fixed

Gate:
- fallback served ratio <= 5%

### Step B: Train-only data reinforcement

Use cycle_4 generated artifacts:
- `artifacts/loop/cycle_4/generated_augmented_pairs.jsonl`
- `data/ft/teacher_pairs.cycle_4.jsonl`

Spec:
- focus dims: `nostalgia`, `complexity`
- +104 train rows (90 generated + 14 hard-case replay)
- no change to validation/test split

### Step C: Hyperparameter tuning

From cycle_4 recommendation:
- `learning_rate: 2e-5 -> 1.2e-5`
- `lora_alpha: 32 -> 64`
- keep `lora_r=16`, `epochs=4`

Run plan:
1. conservative retrain with above settings
2. replay ratio ablation (`0.15` vs `0.25`)
3. source-type mix ablation (reduce `rule_prompt` proportion)

### Step D: Promotion criteria

- `vector_mae` and `mse_norm` better than prompt baseline in clean v4 eval
- focus dims (`nostalgia`, `complexity`) improved vs cycle_4 baseline
- include >= 3 trace-linked failure -> remediation examples in report

## 7. Deliverables / Links

- W&B project: https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301
- W&B runs: https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301/runs
- Weave traces: https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301/weave/traces
- Weave eval comparison meta: `artifacts/eval/summary/weave_eval_comparison.latest.json`
- Radar HTML: `artifacts/eval/summary/test_model_radar.html`
- Report draft: `artifacts/wandb/report_draft.md`
- Failure playbook: `artifacts/wandb/weave_failure_playbook.md`
- Skills log: `docs/hackathon/SKILLS_WORKFLOW_LOG.md`
- HF model: https://huggingface.co/Haruk1y/atelier-kotone-ministral3b-ft
- HF dataset: https://huggingface.co/datasets/Haruk1y/atelier-kotone-ft-request-hidden
