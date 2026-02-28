# HF Ministral-3B FT Runbook

最終更新: 2026-02-28

対象モデル:
- https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512

## 1. 目的

- Mistral API Managed FT ではなく、HF HubモデルをHF JobsでSFTする
- W&B提出要件（run/artifact/report）を満たす

## 2. 前提

- HFログイン済み (`hf auth login`)
- 組織参加済み: `mistral-hackaton-2026`
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
HF_NAMESPACE=mistral-hackaton-2026 \
HF_FT_DATASET_REPO_ID=mistral-hackaton-2026/atelier-kotone-ft-request-hidden \
HF_FT_OUTPUT_MODEL_ID=mistral-hackaton-2026/atelier-kotone-ministral3b-ft \
HF_FT_RUN_NAME=ministral3b-sft-cycle1 \
npm run hf:job:submit
```

## 6. 監視

```bash
hf jobs ps
hf jobs logs <job_id>
hf jobs inspect <job_id>
```

## 7. 評価

```bash
EVAL_MODE=rule_baseline npm run eval:run
EVAL_MODE=prompt_baseline npm run eval:run
EVAL_MODE=fine_tuned npm run eval:run
npm run eval:aggregate
npm run wandb:report:assets
```

## 8. 自己改善ループ

```bash
LOOP_CYCLE_ID=cycle_1 npm run loop:cycle
LOOP_CYCLE_ID=cycle_2 npm run loop:cycle
npm run eval:aggregate
```

生成物:
- `artifacts/loop/cycle_*/next_hparams.yaml`
- `artifacts/loop/cycle_*/augmentation_spec.json`
- `artifacts/wandb/report_draft.md`

## 9. 提出物チェック

- W&B Run URL
- W&B Report URL
- Weave trace URL
- HF Job URL
- HF Model URL
- HF Dataset URL
- `docs/hackathon/SKILLS_WORKFLOW_LOG.md`
- `artifacts/loop/*`
