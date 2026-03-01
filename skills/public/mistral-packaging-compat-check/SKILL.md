---
name: mistral-packaging-compat-check
description: Validate compatibility between Mistral model packaging format and inference path, including adapter versus merged full-model loading. Use when merged models fail generation or runtime/tokenizer artifacts are inconsistent.
---

# Mistral Packaging Compat Check

Use this skill to avoid packaging/runtime mismatches for Mistral-family fine-tuned models.

## When To Use

- User reports that adapter model works but merged full model fails.
- Generation crashes around PEFT wrapping or tokenization.
- Model repo uploads succeeded but inference output is broken.

## Compatibility Workflow

1. Identify artifact type.
- Adapter repo usually contains adapter config/weights.
- Full repo should contain merged model weights and tokenizer/runtime metadata.

2. Verify inference path matches artifact type.
- Adapter path: load base model + apply PEFT adapter.
- Full path: load merged model directly with `AutoModelForCausalLM`.

3. Check generation compatibility hooks.
- For some PEFT + Transformers combinations, ensure `prepare_inputs_for_generation` exists on the wrapped module.

4. Verify runtime metadata files for merged distribution.
- Confirm presence and consistency of:
- `tekken.json`
- `params.json`
- `chat_template.jinja`
- `processor_config.json` (if applicable)
- `generation_config.json`

5. Run smoke inference before publish.
- Use the exact training-aligned prompt.
- Check strict JSON validity and key constraints.

## Repository Mapping

- Merge/upload utility: `scripts/hf/merge_and_upload_full_model.py`
- JSON debug inference: `scripts/hf/debug_full_model_json_inference.py`
- Training prompt reference: `scripts/hf/train_sft_request_to_hidden_lm.py`

## Common Fixes

1. Adapter works, full model fails.
- Re-merge from the exact base model used for adapter training.
- Re-copy required metadata files from base model during merge.

2. Full model loads but output quality is wrong.
- Check that inference prompt matches training prompt exactly.
- Check tokenizer backend and chat template path.

3. PEFT generation errors in adapter mode.
- Add compatibility shim for `prepare_inputs_for_generation` before wrapping.

## Delivery Guidance

- Prefer adapter distribution for iteration speed.
- Publish merged full model when consumers need one-repo direct loading.
- Treat merge as packaging compatibility step, not quality-improvement step.

## References

- `references/packaging-compat-matrix.md`
