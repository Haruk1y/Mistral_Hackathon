# Retrospective Template

## Inputs

- Run ids and links (baseline + candidate)
- `final_metrics.json`
- `iter_eval_metrics.jsonl`
- `hard_cases.valid.jsonl`
- `weave_eval_traces.json` (if available)

## Scoreboard

Use this minimal table:

- `json_valid_rate` (higher is better)
- `parse_error_rate` (lower is better)
- `mae_raw` (lower is better)
- `mae_raw_energy`
- `mae_raw_warmth`
- `mae_raw_brightness`
- `mae_raw_acousticness`
- `mae_raw_complexity`
- `mae_raw_nostalgia`

## Failure Clusters

Build top 3 clusters with:

- cluster name
- evidence count
- representative sample ids
- expected root cause
- highest-impact fix

Recommended cluster boundaries:

- JSON format failures (`json_block_not_found`, `json_decode_error`, extra text)
- Schema failures (missing/extra keys, type mismatch)
- Value failures (out-of-range, wrong dimension tendencies)
- Domain failures (specific `source_type` or request style)

## Action Plan

Order actions by leverage:

1. Output-format reliability fixes.
2. Prompt and decode controls.
3. Data augmentation from hard cases.
4. Hyperparameter changes.

For each action include:

- exact change
- why this addresses the dominant cluster
- expected metric movement
- rollback trigger

## Next-Run Gate

Promote only if:

- `json_valid_rate` is not worse than baseline
- `parse_error_rate` is not worse than baseline
- `mae_raw` improves or remains within agreed tolerance

If format gates regress, do not promote even if loss improves.
