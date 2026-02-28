# Mistral Fine-tuning まとめ（2026-02-28確認）

このドキュメントは、以下の公式ページを確認して作成した要約です。

- https://docs.mistral.ai/capabilities/finetuning
- https://docs.mistral.ai/capabilities/finetuning/text_vision_finetuning

## 1. ざっくり全体像

- MistralのFine-tuningは、プロンプトだけでは難しい一貫性や専門タスク性能を上げるための手段。
- まず `mistral-small-latest` で試す運用が推奨されている。
- 最低課金は1回あたり `$4`、作成したファインチューニング済みモデルの保存は `$2 / model / month`。

## 2. 対応モデル（公式ページ記載ベース）

`text_vision_finetuning` ページで確認できる対応モデル:

- `pixtral-12b-latest`
- `ministral-3b-latest`
- `ministral-8b-latest`
- `mistral-small-latest`
- `devstral-small-latest`

補足:

- APIリファレンス側では `mistral-medium-latest` も `fine_tuning.jobs.create` の対象として記載あり。
- 実運用では、必ずその時点のAPIリファレンスのモデル一覧も合わせて確認すること。

## 3. 学習データ形式（重要）

- 形式は `JSONL`。
- 1行ごとに `messages` を持つ会話データ。
- ロールは `system` / `user` / `assistant` / `tool` が利用可能。
- 学習で誤差逆伝播に使われるのは `assistant` 側トークンのみ。

### Text fine-tuning

- 通常の会話形式データをJSONLで用意する。

### Vision fine-tuning

- 公式記載では「Vision APIで使うフォーマットと同一」。
- つまり、画像付きメッセージを含むマルチモーダル会話をJSONL化して学習データにする。

## 4. 実行フロー（実務向け）

1. 学習データを作る  
`train.jsonl`、必要なら `valid.jsonl` / `test.jsonl` を用意。

2. ファイルをアップロードする  
Files API（`POST /v1/files`）でMistral Cloudへアップロード。  
ファイルサイズ上限は `512MB`（APIリファレンス記載）。

3. Fine-tuningジョブを作る  
`POST /v1/fine_tuning/jobs` で作成。  
主要パラメータ:
- `model`
- `training_files`
- `validation_files`（任意）
- `hyperparameters`（任意）
- `auto_start`（任意）
- `integrations`（任意、例: Weights & Biases）

4. ジョブ状態を監視する  
`GET /v1/fine_tuning/jobs/{job_id}` などで確認。

5. 学習開始/停止  
- `auto_start=false` の場合は `POST /v1/fine_tuning/jobs/{job_id}/start`
- 中止は `POST /v1/fine_tuning/jobs/{job_id}/cancel`

6. 完了後にモデルを推論へ利用  
返却された `fine_tuned_model` を推論時のモデルIDとして使う。

7. 使わないモデルは削除  
`DELETE /v1/models/{model_id}` で削除可能（課金最適化）。

## 5. 代表ハイパーパラメータ（API記載）

- `training_steps`（最大 `10000`）
- `learning_rate`（`0.000001` 以上 `1` 以下）
- `weight`（学習ファイルの重み、`0` 以上 `1` 以下）

## 6. このリポジトリでの進め方（推奨）

- まずは Text fine-tuning から開始（データ検証が簡単）。
- Vision fine-tuning は、画像付きデータの品質基準（解像度・ラベル一貫性）を定義してから投入。
- プレースホルダー画像運用の間は、まずプロンプト解釈精度の改善を優先し、アセット最終版投入後に再学習する。

## 7. 参照リンク

- Capabilities: Fine-tuning  
  https://docs.mistral.ai/capabilities/finetuning
- Capabilities: Text & Vision Fine-tuning  
  https://docs.mistral.ai/capabilities/finetuning/text_vision_finetuning
- API: Fine-tuning endpoints  
  https://docs.mistral.ai/api/endpoint/fine-tuning
- API: Files endpoints  
  https://docs.mistral.ai/api/endpoint/files

## 8. このプロジェクトの具体案

依頼文から隠しパラメータを推定するための具体設計は以下を参照:

- [`docs/FINETUNE_REQUEST_TO_HIDDEN_PARAMS_PLAN.md`](./FINETUNE_REQUEST_TO_HIDDEN_PARAMS_PLAN.md)
