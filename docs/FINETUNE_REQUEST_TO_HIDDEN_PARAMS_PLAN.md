# 依頼文 -> 隠しパラメータ推定 Fine-tuning 設計

最終更新: 2026-02-28

## 1. 目的

- ゲーム体験の向上のため、推論は高速な小型モデルで行う。
- 入力: ユーザー依頼文（自然言語）
- 出力: 隠しパラメータ（数値ベクトル）
- 評価指標: 予測値と正解値の MSE を最小化

対象パラメータ（`lib/types.ts` の `ProfileVector` に合わせる）:

- `energy`
- `warmth`
- `brightness`
- `acousticness`
- `complexity`
- `nostalgia`

## 2. 基本戦略（提案の要点）

1. ランダムに隠しパラメータを生成する
2. その数値ベクトルを大きいMistralモデルに与えて自然な依頼文を生成する
3. 「依頼文, 数値ベクトル」のペアを学習データにする
4. 小さいMistralモデルを、依頼文 -> 数値推定タスクでFine-tuningする
5. 検証時にMSEを計測し、改善を可視化する

## 3. データ生成設計（Teacher: 大きいモデル）

### 3.1 サンプリング

- 各パラメータは `0-100` の範囲で生成する
- 一様乱数だけでなく、実ゲームに寄せた分布も混ぜる
- 追加サンプル方針
- 空間全体をカバーする一様サンプル
- 実運用分布に寄せたクラスタサンプル
- 極端値（0付近/100付近）のエッジケース

### 3.2 生成プロンプト（Teacherへの指示）

- Teacherには「数値を暗に表現した自然文」を生成させる
- 数値の直接露出を禁止する
- 文体バリエーション（丁寧/口語/短文/長文）を制御する

期待する生成例:

- 入力ベクトル: `{energy: 20, warmth: 80, ...}`
- 出力文: 「雨上がりの夕方に合う、やわらかくて落ち着いた曲にしてほしい」

### 3.3 品質フィルタ

生成データには以下の自動チェックを入れる。

- 数値リーク検出: 文中に具体的な数値やパラメータ名が出ていないか
- 長さ制約: 極端に短い/長い依頼文を除外
- 言語検証: 想定言語（日本語/英語）の判定
- 重複除去: 高類似文を除いて多様性を確保

### 3.4 データ分割

- `train/valid/test` は依頼文重複が跨らないように分割
- 再現性のため、乱数seedと生成設定を保存
- 分割後の統計（平均・分散・分位点）を記録

## 4. Fine-tuning 設計（Student: 小さいモデル）

推奨候補:

- `mistral-small-latest`（第一候補）
- 必要なら `ministral-8b-latest` と比較

### 4.1 学習タスク定義

- 入力: 依頼文（+ 任意で天候などの文脈）
- 出力: 固定JSONフォーマットの数値ベクトル

出力フォーマット例（厳密）:

```json
{"energy":42,"warmth":77,"brightness":39,"acousticness":81,"complexity":28,"nostalgia":74}
```

フォーマット運用ルール:

- キー順を固定
- 整数化または小数桁を固定（例: 小数1桁）
- 余計な自然言語を出さない

### 4.2 MSE最適化の扱い（重要）

MistralのマネージドFine-tuningは通常、トークン予測損失（クロスエントロピー）で学習される。  
そのため、MSEを「学習損失として直接最適化」するのではなく、以下の運用でMSE最小化を実現する。

- SFTでJSON数値出力の精度を高める
- 検証セットでMSEを計測する
- MSEが最小のジョブを採用する
- データ分布・出力フォーマットを改善して反復する

## 5. 評価設計（MSE中心）

主指標:

- `mse_raw`: 6次元の平均二乗誤差
- `mse_norm`: 0-100スケールを0-1に正規化したMSE

補助指標:

- `mae_raw`
- `r2_score`
- `json_valid_rate`（JSON parse成功率）
- `schema_valid_rate`（必須キーがすべてある率）
- `within_tolerance@5`（絶対誤差5以内の次元割合）
- `latency_ms_p50/p95`（ゲーム体験向け）

判定ルール例:

- `json_valid_rate` と `schema_valid_rate` が高いモデルを残す
- その中で `mse_norm` が最小のモデルを選定する

## 6. W&B 記録設計（Models + Weave Trace）

### 6.1 W&B Models（学習実験）

`wandb.config` に記録:

- `teacher_model`
- `student_model`
- `dataset_version`
- `sampling_seed`
- `sampling_strategy`
- `output_format_version`
- `train_size`, `valid_size`, `test_size`
- `hyperparameters`（steps/lrなど）

`wandb.log` する主要メトリクス:

- 学習系: `train/loss`, `valid/loss`
- 品質系: `eval/mse_raw`, `eval/mse_norm`, `eval/mae_raw`, `eval/r2`
- 形式系: `eval/json_valid_rate`, `eval/schema_valid_rate`
- 体験系: `eval/latency_ms_p50`, `eval/latency_ms_p95`

Artifacts:

- `dataset: train/valid/test.jsonl`
- `predictions: eval_predictions.jsonl`
- `errors: worst_cases.csv`
- `model: fine-tuned model reference`

### 6.2 W&B Weave（Trace提出用）

提出で見せるべきTraceの最小単位:

- `input.request_text`
- `model.output_raw`
- `model.output_parsed`（JSON）
- `target.vector`
- `error.per_dim`
- `error.mse`
- `latency_ms`

推奨タグ:

- `split` (`valid` / `test`)
- `model_id`
- `dataset_version`
- `run_id`
- `scenario`（`rainy`, `high_nostalgia` など）

これにより、審査時に「どの入力で、どの程度改善したか」をTrace上で説明しやすくなる。

## 7. 実行チェックリスト

1. データ生成seed・分布設定を固定したか
2. 数値リーク検出を通したか
3. JSON出力フォーマットを固定したか
4. `json_valid_rate` が十分高いか
5. `mse_norm` がベースラインより改善したか
6. `latency_ms_p95` がゲーム要件内か
7. W&B Artifacts と Weave Trace を提出可能な形で残したか

## 8. このリポジトリ向け次アクション

- `data/` に以下を追加する
- `ft_request_param_train.jsonl`
- `ft_request_param_valid.jsonl`
- `ft_request_param_test.jsonl`
- 評価スクリプトを追加し、MSE系メトリクスを自動算出する
- 推論API（`/api/interpreter`）でfine-tuned modelの切り替えフラグを用意する
