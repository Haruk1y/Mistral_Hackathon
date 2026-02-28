# ことねのアトリエ (Atelier kotone)

言葉から音楽を作るゲーム体験を題材に、`request_text -> hidden_params` 推定モデルを継続改善するプロジェクトです。  
実装の中心は「評価指標ごとに改善施策を分解し、再学習サイクルで検証する」ことにあります。

## 最新評価スナップショット

- 生成時刻: `2026-02-28T10:00:39.546Z` (`artifacts/eval/summary/latest_summary.json`)
- 評価データ: `frozen_eval_set.v1_teacher_mistral`
- 評価件数: `70`

| mode | json_valid_rate | vector_mae | constraint_match_rate | slot_exact_match | intent_score_mean | output_sanity_score | p95_latency_ms | cost_per_100_requests_usd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rule_baseline | 1.0000 | 19.0762 | 0.2714 | 0.4143 | 59.9400 | 82.1143 | 0.0311 | 0.0200 |
| prompt_baseline | 0.9857 | 20.3671 | 0.2174 | 0.4203 | 57.2290 | 83.4143 | 1558.7694 | 0.2800 |
| fine_tuned | 1.0000 | 26.1286 | 0.1143 | 0.3607 | 45.1300 | 87.8429 | 1203.7898 | 0.1900 |

補足:
- `auto_improvement_delta` (`fine_tuned - prompt_baseline`):  
  `intent_score_mean=-12.0990`, `vector_mae=+5.7614`, `json_valid_rate=+0.0143`
- loop 完了率: `1.0` (`2/2 cycle`)

## 評価指標ごとの実装工夫

### 1) `json_valid_rate`（JSON整形の安定性）

- 厳密JSONの出力を促す固定プロンプトを評価・推論の両方で統一
  - `lib/interpreter.ts`
  - `scripts/eval/run_eval_local.mjs`
  - `scripts/wandb/weave_eval_runner.py`
- fenced code block / 生JSON どちらも復元する `extractJsonBlock` 実装でパース耐性を向上
- `zod` スキーマ検証に通らないレスポンスは採用せず、ルールベース推定へフォールバック
  - `lib/schemas.ts`
  - `lib/interpreter.ts`

### 2) `vector_mae`, `mse_raw`, `mse_norm`, `r2_score`（回帰精度）

- 評価スケールを `target_scale=10` に固定し、入力データ側のスケール差を正規化して比較可能性を担保
  - `scripts/eval/run_eval_local.mjs`
  - `scripts/wandb/weave_eval_runner.py`
- 次元別誤差 (`mae_raw_*`) を計測し、hard case 上位を継続的に抽出
  - `lib/metrics/hidden_params.ts`
  - `scripts/wandb/generate_weave_failure_playbook.mjs`
- 教師データ生成 -> 品質フィルタ -> train/valid/test分割 -> 再学習のデータパイプラインを分離
  - `scripts/ft/generate_teacher_pairs.mjs`
  - `scripts/ft/quality_filter.mjs`
  - `scripts/ft/build_train_valid_test.mjs`

### 3) `constraint_match_rate`, `slot_exact_match`（制約整合・離散選択精度）

- 予測ベクトルから `constraints` と `slot top1` を導出する評価関数を固定化し、比較を自動化
  - `lib/metrics/constraints.ts`
  - `scripts/eval/run_eval_local.mjs`
  - `scripts/wandb/weave_eval_runner.py`
- スロット種別不整合（例: style枠にinstrumentを入れる）を防ぐ整合性チェックを実装
  - `lib/score.ts` (`ensureSlotCategoryIntegrity`)

### 4) `intent_score_mean`（ゲーム体験としての意図一致）

- スコアを単一値でなく内訳に分解:
  - `vectorScore`（距離）
  - `tagScore`（required/optional/forbidden）
  - `preferenceScore`
  - `synergyScore` / `antiSynergyPenalty`
- 上記内訳に基づくコーチング文言をスロット単位で返し、改善方向を可視化
  - `lib/score.ts`

### 5) `output_sanity_score`（出力妥当性）

- JSON不正時の明示的ペナルティ + ベクトル分散に応じた健全性評価を導入
  - `scripts/eval/run_eval_local.mjs`
  - `scripts/wandb/weave_eval_runner.py`
- parse error と trace を必ず記録し、失敗原因を追跡可能にした

### 6) `p95_inference_latency_ms`, `cost_per_100_requests_usd`（運用効率）

- `p50/p95` を同時記録し、平均値だけでなく tail latency を監視
- modeごとに単価を分離管理し、100リクエスト換算のコストで比較
- HF推論失敗時の Mistral fallback を用意し、停止より継続性を優先
  - `scripts/eval/run_eval_local.mjs`
  - `scripts/wandb/weave_eval_runner.py`

## 実験管理と改善ループ

- 各評価サンプルに `trace_id` / `trace_url` を付与し、Weaveで失敗ケースを直接追跡
- `artifacts/eval/runs` と `artifacts/eval/samples` を分離して保存し、再集計を容易化
- サイクル管理 (`cycle_n`) で、失敗分析 -> データ増強 -> 再学習 -> 再評価を自動化
  - `scripts/loop/run_self_improvement_cycle.mjs`
  - `scripts/eval/aggregate_metrics.mjs`
  - `scripts/hf/check_prize_readiness.mjs`

## 再現コマンド

```bash
# 開発
npm run dev

# 単体テスト
npm test

# 評価実行（Python runner -> 必要時 Node fallback）
npm run eval:run

# 評価サマリ集計
npm run eval:aggregate

# Fine-tuning用データ作成
npm run ft:generate
npm run ft:filter
npm run ft:split

# 自己改善サイクル実行
npm run loop:cycle
```

## 関連ドキュメント

- `docs/hackathon/WANDB_REPORT_OUTLINE.md`
- `docs/FINETUNE_REQUEST_TO_HIDDEN_PARAMS_PLAN.md`
- `docs/hackathon/HF_MINISTRAL3B_FT_RUNBOOK.md`
