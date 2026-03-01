---
name: mistral-model-ft-playbook
description: Build, fine-tune, evaluate, and ship Mistral-family models with Hugging Face and PEFT, with emphasis on strict JSON outputs and model-format compatibility. Use when choosing Mistral model variants, deciding prompting vs fine-tuning, preparing `{prompt, completion}` or chat datasets, running SFT jobs, validating generation failures, and deciding adapter-only vs merged full-model distribution.
---

# Mistral Model Ft Playbook

Use this skill to turn Mistral model experimentation into a repeatable production workflow.

## Workflow

1. Choose model and endpoint strategy.
- Read `references/mistral-official-models-and-customization.md`.
- Prefer dated model IDs for stability.
- Choose API-hosted model vs open weights based on latency, cost, and deployment constraints.

2. Decide customization depth.
- Start with system prompt + eval.
- Escalate to fine-tuning only when prompting fails on format, style, or domain consistency.
- Use classifier/moderation layer as a separate safety control, not as a replacement for task tuning.

3. Normalize dataset format before training.
- Prefer `{prompt, completion}` when training strict output format tasks.
- Keep training prompt and inference prompt text aligned exactly.
- Keep evaluation split isolated from training split.

4. Run SFT with reproducible configuration.
- Log exact base model ID, dataset version, LoRA hyperparameters, and max sequence length.
- Treat adapter as default output artifact.
- Track job IDs and links for every run.

5. Validate generation quality with explicit parsers.
- Run inference checks on multiple fixed test cases.
- Enforce exact key set and value constraints.
- Fail fast when parse errors occur (`no_json_start`, `unterminated_json`, type/range mismatch).

6. Decide distribution format.
- Ship adapter when consumers can load base+adapter.
- Merge to full model only for easier deployment compatibility.
- Remember that merge changes packaging, not learned behavior quality.

## Mistral-Specific Guardrails

- Match tokenizer/runtime artifacts to the serving path (`tekken.json`, `params.json`, `chat_template.jinja`, `processor_config.json` when applicable).
- For Ministral-3 style HF models, apply LoRA to the language model module expected by runtime.
- If PEFT integration fails at generation time, patch or ensure `prepare_inputs_for_generation` compatibility before wrapping.
- Preserve EOS handling explicitly in training and generation configuration for format-sensitive tasks.

## This Repository Mapping

- Training: `scripts/hf/train_sft_request_to_hidden_lm.py`
- Dataset conversion: `scripts/ft/convert_to_prompt_completion_dataset.py`
- JSON debug inference: `scripts/hf/debug_full_model_json_inference.py`
- Adapter merge/upload: `scripts/hf/merge_and_upload_full_model.py`
- Job history: `artifacts/hf_jobs/submissions.jsonl`

## References

- Official model/customization guidance: `references/mistral-official-models-and-customization.md`
