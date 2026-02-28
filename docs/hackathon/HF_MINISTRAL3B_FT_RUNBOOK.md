# HF Ministral-3B FT Runbook

最終更新: 2026-02-28

対象モデル:
- https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512

## 1. 目的

- Mistral API Managed FT ではなく、HF HubモデルをHF JobsでSFTする
- W&B提出要件（run/artifact/report）を満たす

## 2. 前提

- HFログイン済み (`hf auth login`)
- Namespaceが `hf auth whoami` と一致（例: `Haruk1y`）
- `WANDB_API_KEY` 設定済み

確認:

```bash
hf auth whoami
hf jobs ps
```

## 3. データ準備

```bash
npm run ft:generate
npm run ft:filter
npm run ft:split
npm run eval:freeze
```

## 4. Dataset公開（任意だが推奨）

```bash
HF_UPLOAD_SUBMIT=true npm run hf:dataset:publish
```

## 5. HF Jobs投入

まずdry-run:

```bash
npm run hf:job:submit
```

実投入:

```bash
HF_JOB_SUBMIT=true \
HF_JOB_FLAVOR=a10g-small \
HF_JOB_TIMEOUT=2h \
HF_NAMESPACE=Haruk1y \
HF_FT_DATASET_REPO_ID=Haruk1y/atelier-kotone-ft-request-hidden \
HF_FT_OUTPUT_MODEL_ID=Haruk1y/atelier-kotone-ministral3b-ft \
HF_FT_RUN_NAME=ministral3b-sft-cycle1 \
npm run hf:job:submit
```

6-run balanced campaign manifest（dry-run）:

```bash
npm run hf:campaign:balanced
```

実投入:

```bash
BALANCED_CAMPAIGN_SUBMIT=true npm run hf:campaign:balanced
```

提出時ログは自動保存される:
- `artifacts/hf_jobs/submissions.jsonl`

## 6. 監視

```bash
hf jobs ps
hf jobs logs <job_id>
hf jobs inspect <job_id>
```

補足:
- `submissions.jsonl` の `job_id` を使って `hf jobs inspect` すると確認が速い。

## 7. 評価

```bash
EVAL_MODE=rule_baseline npm run eval:run
EVAL_MODE=prompt_baseline npm run eval:run
EVAL_MODE=fine_tuned npm run eval:run
npm run eval:aggregate
npm run wandb:report:assets
```

評価ごとのサンプル誤差は以下に保存:
- `artifacts/eval/samples/<run_id>.json`

補足:
- `eval:run` は `scripts/wandb/weave_eval_runner.py` を `uv run` で実行する
- Python依存解決に失敗した環境では `scripts/eval/run_eval_local.mjs` に自動フォールバックする
- 各 sample row には `trace_id`, `trace_url`, `model_source`, `model_id`, `abs_error_by_dim` を記録する

## 8. 自己改善ループ

```bash
WANDB_MCP_ENABLED=true npm run wandb:mcp:fetch
LOOP_CYCLE_ID=cycle_1 npm run loop:cycle
LOOP_CYCLE_ID=cycle_2 npm run loop:cycle
npm run eval:aggregate
```

生成物:
- `artifacts/loop/cycle_*/next_hparams.yaml`
- `artifacts/loop/cycle_*/hparam_patch.yaml`
- `artifacts/loop/cycle_*/mcp_eval_snapshot.json`
- `artifacts/loop/cycle_*/mcp_decision_input.json`
- `artifacts/loop/cycle_*/augmentation_spec.json`
- `artifacts/loop/cycle_*/before_after_metrics.csv`
- `artifacts/loop/cycle_*/generated_augmented_pairs.jsonl`
- `data/ft/teacher_pairs.cycle_*.jsonl`
- `artifacts/wandb/report_draft.md`

次サイクル学習データ化:
```bash
FT_SOURCE_PATH=data/ft/teacher_pairs.cycle_1.jsonl npm run ft:split
```

細粒度W&Bログ（推奨）:
```bash
HF_FT_LOGGING_STEPS=1 \
HF_FT_EVAL_STEPS=25 \
HF_FT_DETAILED_EVAL_STEPS=25 \
HF_FT_HARD_CASE_TOP_K=80 \
WANDB_RUN_GROUP=hf-ft-balanced \
ENABLE_WEAVE_TRACE=true \
HF_JOB_SUBMIT=true npm run hf:job:submit
```

## 9. 提出物チェック

- W&B Run URL
- W&B Report URL
- Weave trace URL
- HF Job URL
- HF Model URL
- HF Dataset URL
- `docs/hackathon/SKILLS_WORKFLOW_LOG.md`
- `artifacts/loop/*`
