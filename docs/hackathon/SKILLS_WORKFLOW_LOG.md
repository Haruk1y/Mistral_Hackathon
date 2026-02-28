# Skills Workflow Log

最終更新: 2026-02-28

このログは `Best Skills活用賞` / `W&B Mini Challenge` 向けの証跡です。  
各サイクルで「評価 -> 分析 -> 改善 -> 再評価」を記録します。

## 1. Workflow Table

| Cycle | Goal | Inputs (MCP/Skill) | Automated Action | Output Artifact | Evidence URL | Metric Delta |
|---|---|---|---|---|---|---|
| 0 | Baseline確立 | Weave eval summary | baseline run集計 | `artifacts/eval/runs/*` | TODO | `json_valid_rate`, `vector_mae` baseline |
| 1 | ハイパラ調整 | Weave failure traces | `next_hparams.yaml` 自動生成 | `artifacts/loop/cycle_1/` | TODO | `auto_improvement_delta` |
| 2 | 苦手軸データ増強 | weak-dim profile | `augmentation_spec.json` + dataset拡張 | `artifacts/loop/cycle_2/` | TODO | `loop_completion_rate` |

## 2. Skills / MCP Usage Log

| Timestamp | Agent | Tool / Skill | Purpose | Output |
|---|---|---|---|---|
| 2026-02-28 | Codex | `weights-and-biases` skill | Weave/Modelsメトリクス設計 | loop design docs |
| 2026-02-28 | Codex | `firebase-*` skills | callable migration方針 | firebase config + callable stubs |
| 2026-02-28 | Codex | `hugging-face-model-trainer` skill | TRL SFT script for Ministral 3B | `scripts/hf/train_sft_request_to_hidden.py` |
| 2026-02-28 | Codex | `hugging-face-jobs` skill | HF Jobs submit/publish automation | `scripts/hf/submit_sft_job.mjs`, `scripts/hf/publish_ft_dataset.mjs` |

## 3. Generated Assets for Submission

- `artifacts/skills/`:
  - prompt templates
  - config diffs
  - generated skill snippets
- `artifacts/loop/`:
  - per-cycle decision log
  - before/after metrics

## 4. Repro Command Memo

```bash
npm run eval:freeze
npm run eval:run
EVAL_MODE=fine_tuned npm run eval:run
npm run eval:aggregate

npm run ft:generate
npm run ft:filter
npm run ft:split
npm run hf:job:submit
npm run wandb:report:assets
npm run loop:cycle
```

## 5. Notes

- 主要提出先: W&B Report + Weave trace links + this workflow log
- 審査説明では「どの判断を agent に委譲したか」を必ず明記する
