---
name: mistral-hidden-params-ft
description: Run end-to-end fine-tuning for request_text-to-hidden-params JSON prediction with Ministral-3-3B in this repository. Use when rebuilding train/validation/test splits from train-only data, converting data to prompt-completion format, launching TRL SFT on Hugging Face Jobs, validating strict JSON output behavior, merging LoRA adapters into full models, and reporting job/model status with links.
---

# Mistral Hidden Params Ft

Use this skill to run the full loop from dataset preparation to deployed model validation.

## Key Files

- `scripts/ft/build_train_valid_test.mjs`
- `scripts/ft/convert_to_prompt_completion_dataset.py`
- `scripts/hf/train_sft_request_to_hidden_lm.py`
- `scripts/hf/debug_full_model_json_inference.py`
- `scripts/hf/merge_and_upload_full_model.py`
- `artifacts/hf_jobs/submissions.jsonl`

## Workflow

1. Rebuild split quality first.
- Use train-only data and regenerate `train/validation/test` before training when existing `validation/test` quality is low.

2. Normalize data to prompt-completion.
- Ensure each sample is `{prompt, completion}` with completion as strict JSON text.
- Keep the inference prompt text exactly aligned between conversion, training, and debug scripts.

3. Launch TRL SFT job on Hugging Face Jobs.
- Train with `scripts/hf/train_sft_request_to_hidden_lm.py`.
- Record job IDs and model IDs in `artifacts/hf_jobs/submissions.jsonl`.

4. Validate JSON output behavior with real generations.
- Run `scripts/hf/debug_full_model_json_inference.py` against either:
- Base + adapter mode (`--base-model-id` + `--adapter-model-id`)
- Full merged model mode (`--model-id`)
- Report `valid_cases=X/Y` and include representative malformed output when failures occur.

5. Merge adapter into full model only when distribution simplicity is needed.
- Use `scripts/hf/merge_and_upload_full_model.py`.
- Explain that merge does not inherently improve prediction quality; it simplifies inference deployment.

6. Report status with concrete links.
- Return Hugging Face job URL(s), output model repo URL(s), and whether each repo is adapter-only or full.

## Guardrails

- Keep prompt wording identical across train/inference unless explicitly requested.
- Keep target keys exactly: `energy`, `warmth`, `brightness`, `acousticness`, `complexity`, `nostalgia`.
- Fail evaluation when output is not valid JSON object with exactly the required keys and integer range `0..10`.
- Prefer evidence from actual job logs over assumptions.
