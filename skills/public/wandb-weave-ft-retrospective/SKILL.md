---
name: wandb-weave-ft-retrospective
description: Analyze W&B, Weave, and local fine-tuning evaluation artifacts, then produce a concrete next-run improvement plan with data, prompt, and training actions. Use after each SFT or eval cycle.
---

# Wandb Weave Ft Retrospective

Use this skill to turn experiment traces into an actionable fine-tuning plan.

## When To Use

- User asks for a retrospective, run diagnosis, or "next fine-tuning plan".
- You have W&B runs but no clear priority of what to fix first.
- You need to convert hard-case outputs into concrete dataset or prompt changes.

## Required Inputs

- Submission records:
- `artifacts/hf_jobs/submissions.jsonl`
- `artifacts/hf_jobs/eval_submissions.jsonl`
- Per-run outputs:
- `outputs/*/final_metrics.json`
- `outputs/*/iter_eval_metrics.jsonl`
- `outputs/*/hard_cases.valid.jsonl`
- Optional trace export:
- `outputs/*/weave_eval_traces.json`

## Workflow

1. Pin the comparison set.
- Pick one baseline run and one candidate run.
- Record base model id, adapter/full model id, dataset version, and run ids.

2. Check gate metrics first.
- Use `json_valid_rate` and `parse_error_rate` as hard gates.
- Then inspect quality metrics (`mae_raw`, `mse_raw`, per-dimension `mae_raw_*`).

3. Cluster dominant failures.
- Group hard cases by `parse_error`, `source_type`, and request pattern.
- Separate format failures from value-quality failures.

4. Convert clusters into fixes.
- Format failures first: output contract prompt text, decode config, EOS handling.
- Value-quality failures next: add targeted training rows for high-error dimensions.
- Hyperparameter tuning last: only after format validity is stable.

5. Produce a run-ready plan.
- Include exact env var overrides for the next run.
- Include promotion criteria and rollback criteria.

## Repository Helpers

- `python scripts/hf/collect_campaign_validation.py`
- `python scripts/hf/debug_json_prompt_variants.py`
- `python scripts/hf/debug_local_eval_outputs.py`

## Output Contract

Always return all of the following:

- Short diagnosis summary.
- Top 3 failure clusters with evidence references.
- Concrete next-run changes (dataset, prompt, train config).
- Clear success gates for promotion.

## References

- `references/retrospective-template.md`
