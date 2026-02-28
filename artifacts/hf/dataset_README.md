---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Atelier kotone FT Dataset

This dataset is for request-to-hidden-params SFT from:

- `request_text`
- `rule_prompt`

Each JSONL line is exactly one sample.

## Splits

- train: 800
- validation: 100
- test: 100

## Columns

- `source_type`: `request_text` or `rule_prompt`
- `request_text`: model input text
- `target_hidden_params`: supervision object
  - `target_hidden_params.vector`: 6-dim integer vector in `0..10`
- `messages`: chat-style format for compatibility
