# W&B Report Outline (Hackathon Submission)

最終更新: 2026-02-28

## 1. Problem / Goal

- Task: `request_text -> hidden_params` 回帰 (`energy`, `warmth`, `brightness`, `acousticness`, `complexity`, `nostalgia`)
- Goal: `mae_raw_*` 改善と、提出要件メトリクスの完全記録
- Constraint: 低レイテンシ運用（目安 `p95 <= 1200ms`）

## 2. Setup

- Base model: `mistralai/Ministral-3-3B-Instruct-2512`
- Fine-tuned model: `Haruk1y/atelier-kotone-ministral3b-ft`
- Dataset: `Haruk1y/atelier-kotone-ft-request-hidden`
- Campaign: `balanced_6run` + `cycle_2` retrain
- Eval set: `frozen_eval_set.v1_teacher_mistral`

### 2.1 MCP Requirement Compliance

- MCP context fetch is **fallback-free**:
  - `scripts/wandb/fetch_mcp_eval_context.mjs` uses only `tools/list` + `tools/call`
  - snapshot source is `wandb_mcp_tools_call`
  - no Python/API fallback in this path
- Evidence (latest):
  - `artifacts/loop/cycle_1/mcp_eval_snapshot.json`
  - `artifacts/loop/cycle_1/mcp_decision_input.json`

## 3. Key Results

### 3.1 Eval aggregate (latest)

- prompt_baseline:
  - `json_valid_rate=0.9857`
  - `vector_mae=20.3671`
  - `p95_inference_latency_ms=1558.7694`
- fine_tuned:
  - `json_valid_rate=1.0000`
  - `vector_mae=26.1286`
  - `p95_inference_latency_ms=1203.7898`
- auto_improvement_delta:
  - `intent_score_mean=-12.0990`
  - `vector_mae=+5.7614`
  - `json_valid_rate=+0.0143`

### 3.2 6-run FT validation (`mae_raw_*`)

- best run: `balanced-run3-cycle1-tuned-augmented` (`mae_raw=21.1508`)
- retrain run: `balanced-run6-cycle2-retrain-harddims` (`mae_raw=25.0061`)
- run5 -> run6 delta:
  - `mae_raw +3.2550`
  - `mae_raw_acousticness +13.0489`
  - `mae_raw_nostalgia +7.3305`

### 3.3 Near-Prod Run1 (2026-02-28)

- W&B run: `zc0av2md` (`prod-cycle1-run1-base-20260228`)
- HF job: `69a2cd97dfb316ac3f7bfe12` (finished)
- MCP snapshot source: `wandb_mcp_tools_call`
- MCP eval source: `train_iter_eval_runs`
- train/iter_eval summary:
  - `eval/mae_raw=25.9905`
  - `eval/mse_norm=0.09257`
  - `objective/train_loss=0.6434`
- Weave traces in near-prod project:
  - `root traces=20`
  - `failures_top_k=20`

## 4. Hard-case Analysis and Augmentation

- hard-case集計対象: finished run 5本、各 `hard_cases_count=80`
- 集計上位次元:
  - `nostalgia (27.6716)`
  - `complexity (26.7398)`
- cycle_2では `nostalgia, acousticness` を重点次元として増強:
  - generated 100 rows + replay 15 rows
  - merged dataset size: `500 -> 615`

## 5. Weave Trace Evidence

- project: `haruk1y_/atelier-kotone-ft` (MCP selected trace project)
- trace board:
  - https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces
- failure traces (fine_tuned example):
  - https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces?query=eval_fine_tuned_eee110309ea24536
  - https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces?query=eval_fine_tuned_452b1864069540e8
  - https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces?query=eval_fine_tuned_12726f15d89749bd
- failure-to-action artifact:
  - `artifacts/wandb/weave_failure_cases.json`
  - `artifacts/wandb/weave_failure_playbook.md`

### 5.1 Near-Prod Weave Failure Findings (run1)

- trace project: `haruk1y_/atelier-kotone-ft-prod`
- top error dims from MCP failures:
  - `nostalgia (45.3285)`
  - `warmth (44.7675)`
  - `acousticness (42.0967)`
- trace examples:
  - https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/weave/traces?query=ft_eval_ae501cb1a5fc43bd
  - https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/weave/traces?query=ft_eval_f6e38c094c164594
  - https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/weave/traces?query=ft_eval_7687a9a3ec71496f

## 6. Precision Improvement Cycle (MCP + Weave + Model)

- Step 1: Model evalを3モードで更新（rule / prompt / fine_tuned）
  - 出力: `artifacts/eval/summary/latest_summary.json`
- Step 2: MCPでW&B Models + Weave tracesを収集（pure `tools/list` / `tools/call`）
  - 出力: `artifacts/loop/cycle_n/mcp_eval_snapshot.json`
- Step 3: Weave失敗事例から弱点次元（`mae_raw_*`）を抽出し、増強データを作成
  - 出力: `artifacts/loop/cycle_n/generated_augmented_pairs.jsonl`
- Step 4: HF Jobsで再学習し、W&Bでvalidation指標を再収集
  - 重点: `mae_raw_*`, `iter_eval/*`, `eval/*`
- Step 5: 失敗事例と改善アクションをレポート化
  - 出力: `artifacts/wandb/weave_failure_playbook.md`, `artifacts/wandb/report_draft.md`

判定ゲート:
- `json_valid_rate >= 0.98`
- focus次元 (`mae_raw_<dim>`) が前サイクルより改善
- `vector_mae`, `mse_norm` が prompt baseline 比で悪化しない

### 6.1 Planned Cycle_2 (Not Executed Yet)

- Inputs:
  - `artifacts/loop/cycle_1/mcp_eval_snapshot.json`
  - `artifacts/wandb/weave_failure_cases.json`
  - `artifacts/loop/cycle_1/summary.json`
- Hyperparameter plan:
  - `learning_rate: 2.0e-5 -> 1.2e-5`
  - `epochs: 2 -> 3`
  - `lora_r/lora_alpha/lora_dropout: keep (16/32/0.05)`
- Data plan:
  - focus dims: `nostalgia`, `acousticness`
  - synthetic hard-case augmentation: `+100`
  - hard-case replay: `+15`
  - merged dataset size: `500 -> 615`
- Promotion gate:
  - include at least 3 MCP trace-linked failure remediations in report
  - require improvement on focus dims before model promotion

## 7. Message for Judges (Latency vs Quality)

- 大きなモデルは一般に推論遅延が増えやすいので、まず3B系でレイテンシを制御
- 小さいモデルは出力品質が落ちやすいため、`prompt tuning -> fine tuning -> hard-case再学習` の順で強化
- 今回は JSON整形/レイテンシ面は改善した一方、回帰誤差 (`mae_raw_*`) は再学習で悪化したため、次は増強分布と重みづけを再設計する

## 8. Link Checklist

- W&B project URL:
  - https://wandb.ai/haruk1y_/atelier-kotone-ft
- W&B project URL (near-prod):
  - https://wandb.ai/haruk1y_/atelier-kotone-ft-prod
- W&B runs URL:
  - https://wandb.ai/haruk1y_/atelier-kotone-ft/runs
- W&B runs URL (near-prod):
  - https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/runs
- W&B report list URL:
  - https://wandb.ai/haruk1y_/atelier-kotone-ft/reports
- W&B report list URL (near-prod):
  - https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/reports
- Weave trace board URL:
  - https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces
- Weave trace board URL (near-prod):
  - https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/weave/traces
- HF model URL:
  - https://huggingface.co/Haruk1y/atelier-kotone-ministral3b-ft
- HF dataset URL:
  - https://huggingface.co/datasets/Haruk1y/atelier-kotone-ft-request-hidden
- HF jobs URL (latest retrain):
  - https://huggingface.co/jobs/Haruk1y/69a2b901dfb316ac3f7bfd33
- HF jobs URL (near-prod run1):
  - https://huggingface.co/jobs/Haruk1y/69a2cd97dfb316ac3f7bfe12
