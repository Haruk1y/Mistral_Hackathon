# Mistral Official Models and Customization Notes

Last checked: 2026-03-01.

## Primary Sources

- Models page: https://docs.mistral.ai/getting-started/models
- Model customization guide: https://docs.mistral.ai/getting-started/customization
- Fine-tuning overview: https://docs.mistral.ai/capabilities/finetuning
- Text & Vision fine-tuning: https://docs.mistral.ai/capabilities/finetuning/text_vision_finetuning

## Practical Takeaways

1. Model catalog and versioning
- Mistral exposes both premier and open models.
- Prefer dated model IDs for stability; `*-latest` aliases can move.
- Track deprecation/retirement windows before long-lived deployments.

2. Prompting vs fine-tuning
- Start with prompting first for speed and lower cost.
- Use fine-tuning when prompt-only behavior is not stable enough, especially for strict format, domain style, or task-specific behavior.

3. Fine-tuning data format
- Chat-style dataset uses a `messages` list with roles (`system`, `user`, `assistant`, `tool`).
- Loss is computed on assistant tokens.
- Validation/test splits are recommended in addition to training split.

4. Customization stack
- Combine three layers: system prompt, model tuning, and moderation/classifier guardrails.
- Treat moderation as separate safety control rather than task-quality tuning.

5. Model weights and licensing
- Open and research-license weights have different license constraints.
- Confirm license before commercial use and before redistributing merged checkpoints.

## Field-Proven Engineering Patterns (Generalized)

1. Adapter and full model have different goals
- Adapter-first is usually best for iteration speed and storage efficiency.
- Full-model merge is primarily a distribution/runtime simplification step.
- Merging does not inherently improve prediction quality.

2. Strict format tasks require parser-based evals
- For JSON tasks, evaluate with explicit parsing and schema/range validation.
- Track failure modes such as:
- `no_json_start`
- `unterminated_json`
- type/range mismatches
- Fail evaluation when format constraints are violated, even if task loss looks good.

3. Keep train-time and inference-time prompts aligned
- Use the exact same instruction template when task behavior depends on output schema.
- Small prompt drift can cause disproportionate format regressions.

4. Preserve Mistral runtime artifact compatibility
- Keep tokenizer/runtime metadata aligned with serving path (`tekken.json`, `params.json`, `chat_template.jinja`, `processor_config.json` where needed).
- Treat packaging/runtime mismatch as a first-class debugging target.

5. Handle PEFT integration points explicitly
- Confirm LoRA target module path for the exact Mistral architecture in use.
- Ensure generation compatibility hooks exist before PEFT wrapping when required (for example, `prepare_inputs_for_generation` availability).

6. EOS handling matters for format-sensitive generation
- Configure EOS/pad behavior explicitly in both training and inference configs.
- Avoid implicit assumptions about EOS behavior across toolchains.

7. Maintain reproducible observability
- Log base model ID, dataset version, hyperparameters, and exact job/model links for each run.
- Keep a compact experiment ledger to support fast rollback and comparison.

## Optional Improvement Loop with Weights & Biases / Weave

1. Collect hard cases automatically
- After each eval run, save malformed outputs and high-error samples.

2. Trace failure clusters
- Use Weave traces to group failures by prompt type, input length, source type, or parse-error category.

3. Convert failures into training data actions
- Add or rebalance examples targeting dominant failure clusters.
- Re-run with same eval suite to validate actual reduction in format failures.

4. Gate promotion
- Promote only runs that improve both task metrics and strict-format validity.

## How to Use This Reference

- Read this file first when choosing a base model and customization method.
- Use the patterns above as default guardrails before launching new fine-tuning cycles.
