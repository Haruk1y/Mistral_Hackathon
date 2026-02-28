# Atelier kotone Demo Game Spec (v0.2)

最終更新: 2026-02-28

## 1. 目的

デモ版のゲーム仕様を固定し、実装・評価・レポートの前提を揃える。

## 2. 確定フロー

1. ファインチューニング済みの小型Mistralモデルが「依頼文」と「隠しパラメータ」を出力する。
2. プレイヤーは依頼文を読んで、手持ちの `kotone` パーツを4スロットで組み合わせる。
3. 選択パーツから、ElevenLabs向けプロンプトをルールベースで構築する。
4. 同じ小型Mistralモデルで、以下2入力のどちらからでも隠しパラメータを推定できるようにする。
   - 依頼文 (`request_text`)
   - ルールベース生成プロンプト (`rule_prompt`)
5. 推定結果と正解隠しパラメータの距離（MSE/MAE）で評価する。

## 3. モデル仕様（推定モデル）

- 単一モデルで2タスク兼用:
  - `request_text -> hidden_params`
  - `rule_prompt -> hidden_params`
- 学習データは `source_type` を持つ混合データで構成:
  - `source_type: request_text`
  - `source_type: rule_prompt`
- 目的関数:
  - 6軸ベクトル回帰のMSE（0-1正規化空間）

## 4. デモ用パーツセット（固定）

`lib/catalog.ts` の `CATALOG_PARTS` を固定セットとして使用する。

現在の固定件数:

- Style: 8
- Instrument: 14
- Mood: 10
- Gimmick: 13
- Total: 45

重複整理方針（確定）:

- Luteは `Sun Lute` のみ採用
- Mandolinは `Round Mandolin` のみ採用
- Recorderは `Tall Recorder` のみ採用
- Crowd系は `Chatty Crowd` のみ採用
- Insect系は `Cricket Pulse` のみ採用
- Moonwind系は `Moonwind Spark` のみ採用
- Rise/Wind系は `Filter Rise` のみ採用
- Whisper系は `Whisper Left` のみ採用

## 5. 実装対応箇所

- パーツ定義:
  - `lib/catalog.ts`
- ElevenLabs向けルールベースプロンプト:
  - `lib/music-provider.ts`
- FT/評価系:
  - `scripts/hf/train_sft_request_to_hidden.py`
  - `scripts/eval/*`
  - `scripts/loop/*`
