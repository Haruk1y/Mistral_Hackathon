---
name: strict-json-output-hardening
description: Improve strict JSON generation reliability for fine-tuned language models using parser-based evaluation, prompt alignment, and targeted retraining loops. Use when outputs are malformed, have extra text, or violate schema/range constraints.
---

# Strict Json Output Hardening

Use this skill when JSON format correctness is a release gate.

## Target Contract

Expected output is exactly one JSON object with keys:
- `energy`
- `warmth`
- `brightness`
- `acousticness`
- `complexity`
- `nostalgia`

Constraints:
- All values are integers.
- All values are in range `0..10`.
- No extra keys, no wrapper object, no markdown.
- No trailing tokens after the final `}`.

## Workflow

1. Freeze prompt contract.
- Keep train prompt and inference prompt text aligned.
- Do not mix multiple output schemas across runs.

2. Measure failures with parser-first eval.
- Parse raw completions as strict JSON object.
- Track failure categories, not only aggregate validity rate.

3. Reproduce with focused debug runs.
- Compare prompt variants before retraining.
- Keep decoding deterministic (`do_sample=false`) during diagnosis.

4. Apply fixes in order.
- Inference-side controls: EOS, max tokens, stop behavior.
- Prompt wording hardening: explicit "single JSON object only".
- Data-side fixes: add hard cases and prompt-completion rows.
- Training-side fixes: keep completion-only loss and split hygiene.

5. Re-evaluate with hard gates.
- Gate on `json_valid_rate`.
- Only compare quality metrics after format validity is stable.

## Repository Mapping

- Prompt variant comparison: `scripts/hf/debug_json_prompt_variants.py`
- Full/adpater debug inference: `scripts/hf/debug_full_model_json_inference.py`
- Local output audit: `scripts/hf/debug_local_eval_outputs.py`
- Training pipeline: `scripts/hf/train_sft_request_to_hidden_lm.py`
- Dataset conversion to prompt-completion: `scripts/ft/convert_to_prompt_completion_dataset.py`

## Required Reporting

Always report:
- Current failure taxonomy with counts.
- Top 3 most frequent categories and representative outputs.
- Applied fixes and expected metric movement.
- Pass/fail gate recommendation for promotion.

## References

- `references/json-failure-taxonomy.md`
