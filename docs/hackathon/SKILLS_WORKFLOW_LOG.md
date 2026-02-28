# Skills Workflow Log

最終更新: 2026-02-28 (run2 submitted)

このログは `Best Skills活用賞` / `W&B Mini Challenge` 向けに、評価と自己改善ループの実行証跡をまとめたものです。

## 1. Workflow Table

| Cycle | Goal | Inputs (MCP/Skill) | Automated Action | Output Artifact | Evidence URL | Metric Delta |
|---|---|---|---|---|---|---|
| 0 | Baseline確立 | `weights-and-biases` / Weave eval summary | `eval:run` x3 + `eval:aggregate` | `artifacts/eval/summary/latest_summary.json` | https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces | prompt baseline: `vector_mae=20.3671`, `p95=1558.77ms` |
| 1 | 6-run FT campaign | `hugging-face-jobs` + `weights-and-biases` | `BALANCED_CAMPAIGN_SUBMIT=true npm run hf:campaign:balanced` | `artifacts/hf_jobs/submissions.jsonl`, `balanced_6run_validation_snapshot.json` | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/qyyzbzes | best `mae_raw=21.1508` (run3) |
| 2 | hard-case次元抽出 + 再学習 | `weights-and-biases` + MCP context | `hf:campaign:collect` -> `loop:cycle` -> `hf:job:submit` (retrain) | `balanced_6run_hard_dims.json`, `artifacts/loop/cycle_2/*` | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/3y872rl4 | run5->run6 `mae_raw +3.2550` (悪化), `acousticness +13.0489`, `nostalgia +7.3305` |
| 3 | MCP主導の改善入力再構成 | `weights-and-biases` + `hf-mcp` | `wandb:mcp:fetch` -> `loop:cycle` -> `wandb:failures:analyze` | `artifacts/loop/cycle_3/*`, `artifacts/wandb/weave_failure_playbook.md` | https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces | `weak_dims_source=mcp_failures_top_k`, focus=`nostalgia,brightness` |
| 4 | run2実行前の計画確定 | `weights-and-biases` + MCP snapshot (`tools/call`) | cycle_1失敗分布からrun2の保守的設定を確定（実行は保留） | `docs/hackathon/HF_MINISTRAL3B_FT_RUNBOOK.md`, `docs/hackathon/WANDB_REPORT_OUTLINE.md`, `artifacts/wandb/report_draft.md` | https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/runs/zc0av2md | plan: `focus=nostalgia,acousticness`, `add_ratio=0.12`, `replay_ratio=0.25`, `lr=1.2e-5` |
| 5 | run2実投入 | `weights-and-biases` + `hugging-face-jobs` | `loop:cycle` (stale MCP snapshot) -> `ft:split` -> `hf:dataset:publish` -> `hf:job:submit` | `artifacts/loop/cycle_1/*`, `data/ft/ft_request_param_*.jsonl`, `artifacts/hf_jobs/submissions.jsonl` | https://huggingface.co/jobs/Haruk1y/69a3033adfb316ac3f7c0055 | first submit failed (schema cast), retried job is `RUNNING` |

## 2. Skills / MCP Usage Log

| Timestamp (UTC) | Agent | Tool / Skill | Purpose | Output |
|---|---|---|---|---|
| 2026-02-28 09:35 | Codex | `hugging-face-jobs` | 6-run campaign実投入 | HF Job IDs: `69a2b6dc...` - `69a2b6e9...` |
| 2026-02-28 09:40 | Codex | `weights-and-biases` + `hf-mcp` | cycle_2用の弱点次元選定 | `artifacts/loop/cycle_2/mcp_eval_snapshot.json` |
| 2026-02-28 09:43 | Codex | `weights-and-biases` | hard-dim増強データ作成 | `generated_augmented_pairs.jsonl` (100) + replay (15) |
| 2026-02-28 09:44 | Codex | `hugging-face-jobs` | cycle_2 retrain投入 | https://huggingface.co/jobs/Haruk1y/69a2b901dfb316ac3f7bfd33 |
| 2026-02-28 09:59 | Codex | `weights-and-biases` | validation再収集 (`mae_raw_*`, hard cases) | `balanced_6run_validation_snapshot.json` |
| 2026-02-28 10:26 | Codex | `weights-and-biases` | Weave失敗事例をtrace付きで抽出し改善案を自動生成 | `artifacts/wandb/weave_failure_playbook.md` |
| 2026-02-28 10:55 | Codex | `weights-and-biases` + `hf-mcp` | `tools/list` / `tools/call` の pure MCP でloop入力を再生成 | `artifacts/loop/cycle_3/mcp_eval_snapshot.json` (`source=wandb_mcp_tools_call`) |
| 2026-02-28 12:30 | Codex | `weights-and-biases` + MCP snapshot | run1失敗事例に基づくrun2計画の文書化（未実行） | runbook / report outline / report draft updated |
| 2026-02-28 14:50 | Codex | `weights-and-biases` | run2用MCP fetchを再試行し、DNS不安定時は stale snapshot 継続方針に切替 | `artifacts/loop/cycle_1/mcp_eval_snapshot.json` |
| 2026-02-28 14:54 | Codex | `hugging-face-jobs` | run2 SFT job投入（初回） | https://huggingface.co/jobs/Haruk1y/69a301bd5672f7593677022d |
| 2026-02-28 14:58 | Codex | `weights-and-biases` + `hugging-face-jobs` | データ型不整合（float/int）を修正し、run2bを再投入 | https://huggingface.co/jobs/Haruk1y/69a3033adfb316ac3f7c0055 |

## 3. Repro Command Memo

```bash
npm run hf:campaign:balanced
BALANCED_CAMPAIGN_SUBMIT=true npm run hf:campaign:balanced

npm run hf:campaign:collect
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_2 npm run wandb:mcp:fetch
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_2 LOOP_FORCE_WEAK_DIMS=nostalgia,acousticness npm run loop:cycle

FT_SOURCE_PATH=data/ft/teacher_pairs.cycle_2.jsonl npm run ft:split
HF_UPLOAD_SUBMIT=true npm run hf:dataset:publish
HF_JOB_SUBMIT=true HF_FT_RUN_NAME=balanced-run6-cycle2-retrain-harddims npm run hf:job:submit
npm run wandb:failures:analyze
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_3 npm run loop:cycle

# run2 plan (pending)
WANDB_PROJECT=atelier-kotone-ft-prod WANDB_ENTITY=haruk1y_ WEAVE_PROJECT=atelier-kotone-ft-prod TRACKIO_PROJECT=atelier-kotone-ft-prod WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_2 LOOP_FORCE_WEAK_DIMS=nostalgia,acousticness LOOP_ADD_RATIO=0.12 LOOP_HARD_CASE_REPLAY_RATIO=0.25 npm run loop:cycle
HF_JOB_SUBMIT=true HF_FT_RUN_NAME=prod-cycle2-run2-mcp-conservative-20260228 HF_FT_LR=0.000012 HF_FT_EPOCHS=3 LOOP_CYCLE_ID=cycle_2 WANDB_PROJECT=atelier-kotone-ft-prod WANDB_ENTITY=haruk1y_ WEAVE_PROJECT=atelier-kotone-ft-prod TRACKIO_PROJECT=atelier-kotone-ft-prod WANDB_RUN_GROUP=prod_cycle2 FT_DATASET_VERSION=cycle_2_mcp_conservative FT_SOURCE_TYPE_MIX=request_text+rule_prompt+harddim_aug ENABLE_WEAVE_TRACE=true npm run hf:job:submit
```

## 4. Evidence URLs

- W&B Project: https://wandb.ai/haruk1y_/atelier-kotone-ft
- W&B Runs: https://wandb.ai/haruk1y_/atelier-kotone-ft/runs
- Weave Traces: https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces
- HF Model: https://huggingface.co/Haruk1y/atelier-kotone-ministral3b-ft
- HF Dataset: https://huggingface.co/datasets/Haruk1y/atelier-kotone-ft-request-hidden
- Latest retrain job: https://huggingface.co/jobs/Haruk1y/69a2b901dfb316ac3f7bfd33
- Skills manifest: /Users/yajima/Desktop/dev/Mistral_Hackathon/artifacts/skills/skills_manifest.json
