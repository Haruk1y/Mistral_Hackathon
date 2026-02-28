# HF Ministral-3B FT Runbook

最終更新: 2026-02-28

対象モデル:
- https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512

## 1. 目的

- HF Jobsで `request_text -> hidden_params` のSFTを回し、validation指標 (`mae_raw_*`) を継続収集する
- hard-caseの誤差次元を抽出してデータ増強し、サイクル学習（cycle_2 以降）で再学習する
- W&B / Weave / HF URLつきで提出証跡を残す

## 2. 前提

- HFログイン済み (`hf auth login`)
- `WANDB_API_KEY`, `HF_TOKEN` が設定済み
- Namespace: `Haruk1y`
- W&B project（baseline）: `haruk1y_/atelier-kotone-ft`
- W&B project（near-prod推奨）: `haruk1y_/atelier-kotone-ft-prod`

確認:

```bash
hf auth whoami
hf jobs ps
```

near-prod運用（推奨）:
- 既存検証と混線しないよう、`WANDB_PROJECT`, `WEAVE_PROJECT`, `TRACKIO_PROJECT` を `atelier-kotone-ft-prod` に分離
- run名は `prod-cycle{n}-run{k}-...`、groupは `prod_cycle{n}` で固定

## 3. データ準備

```bash
npm run ft:generate
npm run ft:filter
npm run ft:split
npm run eval:freeze
```

## 4. Dataset公開

```bash
HF_UPLOAD_SUBMIT=true npm run hf:dataset:publish
```

公開先:
- https://huggingface.co/datasets/Haruk1y/atelier-kotone-ft-request-hidden

## 5. 6-run campaign投入

dry-run:

```bash
npm run hf:campaign:balanced
```

実投入:

```bash
BALANCED_CAMPAIGN_SUBMIT=true npm run hf:campaign:balanced
```

記録:
- `artifacts/hf_jobs/submissions.jsonl`
- `artifacts/hf_jobs/balanced_6run_validation_snapshot.json`
- `artifacts/hf_jobs/balanced_6run_hard_dims.json`

## 6. Validation収集

```bash
npm run hf:campaign:collect
```

出力:
- `artifacts/hf_jobs/balanced_6run_validation_snapshot.json`
- `artifacts/hf_jobs/balanced_6run_validation_snapshot.csv`
- `artifacts/hf_jobs/balanced_6run_hard_dims.json`

## 7. hard-case増強 + cycle_2再学習（実績）

MCP文脈取得（pure MCP）:

```bash
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_2 npm run wandb:mcp:fetch
```

hard-dim固定で増強（cycle_2時点の再現コマンド）:

```bash
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_2 LOOP_FORCE_WEAK_DIMS=nostalgia,acousticness npm run loop:cycle
FT_SOURCE_PATH=data/ft/teacher_pairs.cycle_2.jsonl npm run ft:split
```

cycle_2再学習:

```bash
HF_JOB_SUBMIT=true \
HF_FT_OBJECTIVE=next_token_json_sft \
HF_FT_RUN_NAME=balanced-run6-cycle2-retrain-harddims \
HF_FT_DATASET_REPO_ID=Haruk1y/atelier-kotone-ft-request-hidden \
HF_FT_LR=0.000009 \
HF_FT_EPOCHS=3 \
LOOP_CYCLE_ID=cycle_2 \
WANDB_RUN_GROUP=balanced_6run-cycle2-retrain \
FT_DATASET_VERSION=cycle_2_harddim_nostalgia_acousticness \
FT_SOURCE_TYPE_MIX=request_text+rule_prompt+harddim_aug \
ENABLE_WEAVE_TRACE=true \
npm run hf:job:submit
```

## 8. 実投入結果 (2026-02-28)

| Run | HF Job | W&B Run | State | mae_raw | mae_raw_acousticness | mae_raw_nostalgia |
|---|---|---|---|---:|---:|---:|
| balanced-run1-cycle1-default | https://huggingface.co/jobs/Haruk1y/69a2b6dcdfb316ac3f7bfd1f | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/kt04tejg | finished | 21.7748 | 22.5142 | 22.8073 |
| balanced-run2-cycle1-tuned | https://huggingface.co/jobs/Haruk1y/69a2b6dfdfb316ac3f7bfd21 | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/kz4r8fdp | failed | 23.9076 | 23.6917 | 26.9540 |
| balanced-run3-cycle1-tuned-augmented | https://huggingface.co/jobs/Haruk1y/69a2b6e25672f7593677018e | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/qyyzbzes | finished | 21.1508 | 19.8023 | 22.1932 |
| balanced-run4-cycle2-tuned | https://huggingface.co/jobs/Haruk1y/69a2b6e65672f75936770190 | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/pudh5lxw | finished | 21.9254 | 22.7912 | 23.0036 |
| balanced-run5-cycle2-tuned-augmented | https://huggingface.co/jobs/Haruk1y/69a2b6e95672f75936770192 | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/tsyaom4h | finished | 21.7512 | 20.6538 | 22.7623 |
| balanced-run6-cycle2-retrain-harddims | https://huggingface.co/jobs/Haruk1y/69a2b901dfb316ac3f7bfd33 | https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/3y872rl4 | finished | 25.0061 | 33.7027 | 30.0928 |

## 9. cycle_2差分 (run5 -> run6)

- `mae_raw`: `+3.2550` (悪化)
- `mse_raw`: `+203.9791` (悪化)
- `mae_raw_acousticness`: `+13.0489` (悪化)
- `mae_raw_nostalgia`: `+7.3305` (悪化)
- `mae_raw_energy`: `-3.1304` (改善)
- hard-case平均でも `acousticness`, `nostalgia` が悪化

解釈:
- hard-case抽出自体は成立 (`hard_cases_count=80` x finished 5 runs)
- ただし cycle_2増強の分布が強すぎ、対象次元でオーバーシュートした可能性が高い

## 10. 評価メッセージ (Latency vs Quality)

- prompt_tuned baseline (`Ministral-3-3B`):  
  `p95_latency=1558.77ms`, `vector_mae=20.3671`
- fine_tuned model (`Haruk1y/atelier-kotone-ministral3b-ft`):  
  `p95_latency=1203.79ms`, `vector_mae=26.1286`

伝え方:
- 大きいモデルは一般にレイテンシ増のリスクがあるため、まず3Bで運用可能な遅延に寄せる
- 小さいモデルは品質が落ちやすいため、`prompt tuning -> FT -> hard-case増強` の順で底上げする
- 今回は遅延とJSON整形は改善、回帰精度は要再調整という結果

## 11. 精度改善サイクル（MCP + Weave + Model）

目的:
- `mae_raw_*` を改善しつつ、失敗事例と改善アクションをW&B提出資料に接続する

運用フロー（cycle_n）:
1. Model評価を更新
```bash
EVAL_ALLOW_LOCAL_FALLBACK=false EVAL_MODE=rule_baseline npm run eval:run
EVAL_ALLOW_LOCAL_FALLBACK=false EVAL_MODE=prompt_baseline npm run eval:run
EVAL_ALLOW_LOCAL_FALLBACK=false EVAL_MODE=fine_tuned npm run eval:run
npm run eval:aggregate
```
2. MCPでW&B Models + Weave tracesを取得
```bash
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_n npm run wandb:mcp:fetch
```
3. Weave失敗事例から弱点次元を抽出し、増強データ生成
```bash
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_n npm run loop:cycle
FT_SOURCE_PATH=data/ft/teacher_pairs.cycle_n.jsonl npm run ft:split
```
4. Model再学習（HF Jobs）
```bash
HF_JOB_SUBMIT=true \
HF_FT_OBJECTIVE=next_token_json_sft \
HF_FT_RUN_NAME=balanced-run-cycle_n-mcp-weave-retrain \
LOOP_CYCLE_ID=cycle_n \
WANDB_RUN_GROUP=balanced_6run-cycle_n-retrain \
FT_DATASET_VERSION=cycle_n_mcp_weave_aug \
FT_SOURCE_TYPE_MIX=request_text+rule_prompt+harddim_aug \
ENABLE_WEAVE_TRACE=true \
npm run hf:job:submit
```
5. 失敗事例→改善案レポート化
```bash
npm run wandb:failures:analyze
npm run wandb:report:assets
```

判定ゲート（推奨）:
- `vector_mae` / `mse_norm` が prompt_baseline 比で悪化しない
- focus次元 (`mae_raw_<dim>`) が cycle_{n-1} 比で改善
- `json_valid_rate >= 0.98` を維持

### MCP DNS / Connection Troubleshooting

- このリポジトリのMCP fetchは `https://mcp.withwandb.com/mcp` を使用（W&B公式MCP endpoint）
- `loop:cycle` は `WANDB_MCP_ENABLED=true` 時、MCP fetch失敗を黙って継続しない（明示的に失敗）
- 一時的なDNS失敗に備えてリトライ可能:

```bash
WANDB_MCP_ENABLED=true \
WANDB_MCP_RETRY_MAX=5 \
WANDB_MCP_RETRY_BACKOFF_MS=1200 \
LOOP_CYCLE_ID=cycle_n \
npm run loop:cycle
```

またはresilient script:

```bash
LOOP_CYCLE_ID=cycle_n npm run loop:cycle:resilient
```

- 既存snapshotの再利用を許可する場合のみ:

```bash
WANDB_MCP_ENABLED=true \
WANDB_MCP_ALLOW_STALE_SNAPSHOT=true \
LOOP_CYCLE_ID=cycle_n \
npm run loop:cycle
```

- hosted MCPへのDNS解決が難しい環境では、W&B公式MCPサーバーをローカルHTTPで起動して接続:

```bash
WANDB_MCP_TRANSPORT=http \
WANDB_MCP_PORT=8080 \
uvx --from git+https://github.com/wandb/wandb-mcp-server \
  wandb_mcp_server
```

```bash
WANDB_MCP_ENABLED=true \
WANDB_MCP_BASE_URL=http://127.0.0.1:8080/mcp \
LOOP_CYCLE_ID=cycle_n \
npm run loop:cycle
```

補足:
- `fetch_mcp_eval_context.mjs` はデフォルトで `https://mcp.withwandb.com/mcp` を試し、失敗時に `http://127.0.0.1:8080/mcp` も候補に入れて再試行する。

## 12. URL Checklist

- W&B Project: https://wandb.ai/haruk1y_/atelier-kotone-ft
- W&B Project (near-prod): https://wandb.ai/haruk1y_/atelier-kotone-ft-prod
- W&B Runs (campaign): https://wandb.ai/haruk1y_/atelier-kotone-ft/runs
- W&B Runs (near-prod): https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/runs
- W&B cycle_2 retrain run: https://wandb.ai/haruk1y_/atelier-kotone-ft/runs/3y872rl4
- W&B prod cycle_1 run_1: https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/runs/zc0av2md
- W&B Reports list: https://wandb.ai/haruk1y_/atelier-kotone-ft/reports
- Weave traces: https://wandb.ai/haruk1y_/atelier-kotone-ft/weave/traces
- Weave traces (near-prod): https://wandb.ai/haruk1y_/atelier-kotone-ft-prod/weave/traces
- HF Model: https://huggingface.co/Haruk1y/atelier-kotone-ministral3b-ft
- HF Dataset: https://huggingface.co/datasets/Haruk1y/atelier-kotone-ft-request-hidden
- HF Jobs: https://huggingface.co/jobs/Haruk1y/69a2b901dfb316ac3f7bfd33

## 13. Run2 Plan (MCP-Based) and Submission Status

目的:
- run1 (`prod-cycle1-run1-base-20260228`) の失敗分布をMCP経由で再利用し、run2を保守的設定で再学習する

入力エビデンス:
- `artifacts/loop/cycle_1/mcp_eval_snapshot.json`
- `artifacts/loop/cycle_1/mcp_decision_input.json`
- `artifacts/loop/cycle_1/decision_log.md`
- `artifacts/wandb/weave_failure_playbook.md`

MCP起点の重点次元:
- primary: `nostalgia`, `acousticness` (`mcp_decision_input` の `focus_dims`)
- secondary check: `warmth` (top failure tracesで頻出)

run2方針（cycle_2）:
- 増強は強すぎるシフトを避ける
- `LOOP_ADD_RATIO=0.12`（前回0.2より縮小）
- `LOOP_HARD_CASE_REPLAY_RATIO=0.25`（失敗再現の比率を増やす）
- 学習率/epochはcycle_1提案値を採用
- `HF_FT_LR=0.000012`
- `HF_FT_EPOCHS=3`

実行コマンド（run2投入時の実績）:

```bash
WANDB_PROJECT=atelier-kotone-ft-prod \
WANDB_ENTITY=haruk1y_ \
WEAVE_PROJECT=atelier-kotone-ft-prod \
TRACKIO_PROJECT=atelier-kotone-ft-prod \
WANDB_MCP_ENABLED=true \
WANDB_MCP_ALLOW_STALE_SNAPSHOT=true \
LOOP_CYCLE_ID=cycle_1 \
LOOP_FORCE_WEAK_DIMS=nostalgia,acousticness \
LOOP_ADD_RATIO=0.12 \
LOOP_HARD_CASE_REPLAY_RATIO=0.25 \
npm run loop:cycle
```

```bash
FT_SOURCE_PATH=data/ft/teacher_pairs.cycle_2.jsonl npm run ft:split
```

```bash
HF_JOB_SUBMIT=true \
HF_FT_RUN_NAME=prod-cycle2-run2-mcp-conservative-20260228 \
HF_FT_DATASET_REPO_ID=Haruk1y/atelier-kotone-ft-request-hidden \
HF_FT_LR=0.000012 \
HF_FT_EPOCHS=3 \
HF_FT_EVAL_STEPS=25 \
HF_FT_DETAILED_EVAL_STEPS=1 \
HF_FT_HARD_CASE_TOP_K=80 \
LOOP_CYCLE_ID=cycle_2 \
WANDB_PROJECT=atelier-kotone-ft-prod \
WANDB_ENTITY=haruk1y_ \
WEAVE_PROJECT=atelier-kotone-ft-prod \
TRACKIO_PROJECT=atelier-kotone-ft-prod \
WANDB_RUN_GROUP=prod_cycle2 \
FT_DATASET_VERSION=cycle_2_mcp_conservative \
FT_SOURCE_TYPE_MIX=request_text+rule_prompt+harddim_aug \
ENABLE_WEAVE_TRACE=true \
npm run hf:job:submit
```

run2投入結果:
- first submit: https://huggingface.co/jobs/Haruk1y/69a301bd5672f7593677022d
  - status: failed (dataset schema cast error: float -> int64)
- resubmitted run2b: https://huggingface.co/jobs/Haruk1y/69a3033adfb316ac3f7c0055
  - status: running
- submitted_at: 2026-02-28 14:54 / 15:01 (local)
- note: hosted MCP DNSが一時失敗したため、`cycle_1` のMCP artifactsを stale-snapshot モードで再利用して増強データを生成

run2完了後の必須更新:
- `npm run hf:campaign:collect`
- `WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_2 npm run wandb:mcp:fetch`
- `npm run wandb:failures:analyze`
- `npm run wandb:report:assets`

判定ゲート:
- `eval/mae_raw` が run1 (`25.9905`) を下回る
- `mae_raw_nostalgia`, `mae_raw_acousticness` が run1 比で改善
- `json_valid_rate >= 0.98` を維持
