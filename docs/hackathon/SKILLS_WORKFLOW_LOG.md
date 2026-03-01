# Skills Workflow Log

Last updated: 2026-03-01

This log records skill usage and evidence for hackathon submission.

## 1. Workflow Summary

| Step | Skill | Action | Evidence Artifact | External URL |
|---|---|---|---|---|
| 1 | `weights-and-biases` | ran 3-model eval (`prompt_baseline`, `fine_tuned`, `large_baseline`) on test set | `artifacts/eval/runs/*.json`, `artifacts/eval/samples/*.json` | https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301/runs |
| 2 | `weights-and-biases` | published Weave eval comparison (`kotone-test-model-comparison-v3`) | `artifacts/eval/summary/weave_eval_comparison.latest.json` | https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301/weave |
| 3 | `weights-and-biases` + `hf-mcp` | fetched MCP eval context (`tools/list`, `tools/call`) into cycle_4 | `artifacts/loop/cycle_4/mcp_eval_snapshot.json` | https://mcp.withwandb.com/mcp |
| 4 | `weights-and-biases` | generated failure playbook from Weave traces | `artifacts/wandb/weave_failure_playbook.md` | https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301/weave/traces |
| 5 | `weights-and-biases` | generated report assets (draft + metrics csv) | `artifacts/wandb/report_draft.md`, `artifacts/wandb/report_metrics.csv` | https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301/reports |
| 6 | `hugging-face-jobs` | tracked FT training job outcome and artifacts | `artifacts/hf_jobs/submissions.jsonl` | https://huggingface.co/jobs/Haruk1y/69a326185672f759367702cb |
| 7 | `weights-and-biases` + `hf-mcp` | executed self-improvement cycle with train-only augmentation plan | `artifacts/loop/cycle_4/summary.json`, `artifacts/loop/cycle_4/augmentation_spec.json` | https://wandb.ai/haruk1y_/atelier-kotone-ft-kotoneonly-v1-20260301/weave |

## 2. Cycle 4 Outputs

- weak dims: `nostalgia`, `complexity`
- generated augmentation: `90`
- hard-case replay: `14`
- total added train rows: `104`
- merged train file: `data/ft/teacher_pairs.cycle_4.jsonl`
- recommended hparams:
  - `learning_rate: 2e-5 -> 1.2e-5`
  - `lora_alpha: 32 -> 64`
  - `lora_r: 16 (keep)`
  - `epochs: 4 (keep)`

## 3. Commands Used (Repro)

```bash
# 3-model eval
EVAL_MODE=prompt_baseline npm run eval:run
EVAL_MODE=fine_tuned npm run eval:run
EVAL_MODE=large_baseline npm run eval:run

# aggregate + radar + weave eval publish
npm run eval:radar:test
WEAVE_EVAL_COMPARISON_NAME=kotone-test-model-comparison-v3 npm run wandb:evals:publish
npm run eval:aggregate

# MCP context + loop plan
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_4 npm run wandb:mcp:fetch:resilient
WANDB_MCP_ENABLED=true LOOP_CYCLE_ID=cycle_4 npm run loop:cycle:resilient

# failure and report assets
LOOP_CYCLE_ID=cycle_4 npm run wandb:failures:analyze
LOOP_CYCLE_ID=cycle_4 npm run wandb:report:assets
```

## 4. Notes

- Current prompt vs FT comparison is partially confounded by fallback serving (`http_404` on HF router path for both model IDs on some rows).
- A clean next cycle should evaluate merged FT model without fallback mixing before promotion decisions.
