# 依頼文 -> 隠しパラメータ推定 Fine-tuning 設計（HF Jobs / Ministral 3B）

最終更新: 2026-02-28

## 1. 目的

- ゲーム体験の向上のため、推論は高速な小型モデルで行う
- 入力: `request_text`（自然言語）
- 出力: 6軸の隠しパラメータJSON
- 主評価: `mse_norm` / `vector_mae` / `json_valid_rate`

対象パラメータ:
- `energy`
- `warmth`
- `brightness`
- `acousticness`
- `complexity`
- `nostalgia`

## 2. モデル戦略（更新）

### Primary（本命）
- Hugging Face上の `mistralai/Ministral-3-3B-Instruct-2512` を LoRA/QLoRA でSFT
- 実行基盤は `HF Jobs`（`a10g-small` 起点）
- ログは `W&B`（必須） + `Trackio`（任意）

### Secondary（比較）
- Mistral Managed FT は比較用（fallback）に限定

## 3. データ生成設計

### 3.1 Teacher pair
- hidden paramsをランダム生成
- Teacher（`mistral-large-latest`）で依頼文を生成
- ペア: `(request_text, target_hidden_params)`

### 3.2 品質フィルタ
- 数値リーク除去（パラメータ名/JSON露出の禁止）
- 重複除去
- 長さ/言語チェック

### 3.3 分割
- train/valid/test = 80/10/10
- 生成統計を保存（`data/ft/ft_split_stats.json`）

## 4. 学習タスク定義

出力形式（厳密JSON）:

```json
{"energy":42,"warmth":77,"brightness":39,"acousticness":81,"complexity":28,"nostalgia":74}
```

ルール:
- キー順固定
- 数値は0-100
- 余計な自然言語を出さない

## 5. 実装済みスクリプト

- データ生成:
  - `scripts/ft/generate_teacher_pairs.mjs`
  - `scripts/ft/quality_filter.mjs`
  - `scripts/ft/build_train_valid_test.mjs`
- Eval:
  - `scripts/eval/generate_frozen_eval_set.mjs`
  - `scripts/eval/run_eval.mjs`
  - `scripts/eval/aggregate_metrics.mjs`
- HF実行:
  - `scripts/hf/train_sft_request_to_hidden.py`
  - `scripts/hf/publish_ft_dataset.mjs`
  - `scripts/hf/submit_sft_job.mjs`

## 6. W&B記録設計

### 6.1 Models
記録メトリクス:
- `train/loss`, `eval/loss`
- `json_valid_rate`
- `vector_mae`
- `mse_raw`, `mse_norm`
- `constraint_match_rate`
- `slot_exact_match`
- `intent_score_mean`
- `p95_inference_latency_ms`
- `cost_per_100_requests_usd`

### 6.2 Artifacts
- dataset（train/valid/test）
- eval summary
- model adapter

### 6.3 Weave
traceタグ:
- `dataset_version`
- `model_id`
- `run_id`
- `scenario`

## 7. 実行手順（ローカル）

```bash
# 1) データ準備
npm run ft:generate
npm run ft:filter
npm run ft:split
npm run eval:freeze

# 2) （任意）HF datasetへアップロード
HF_UPLOAD_SUBMIT=true npm run hf:dataset:publish

# 3) HF Jobsへ学習投入
HF_JOB_SUBMIT=true npm run hf:job:submit

# 4) 評価と集約
EVAL_MODE=rule_baseline npm run eval:run
EVAL_MODE=prompt_baseline npm run eval:run
EVAL_MODE=fine_tuned npm run eval:run
npm run eval:aggregate

# 5) W&B report素材生成
npm run wandb:report:assets
```

## 8. 受け入れ条件

1. `json_valid_rate >= 0.98`
2. `vector_mae`, `mse_norm`, `intent_score_mean` でFTがprompt baselineより改善
3. `loop_completion_rate >= 1.0`（2サイクル以上）
4. W&B Reportに run/sweep/trace/artifact URL が揃う
