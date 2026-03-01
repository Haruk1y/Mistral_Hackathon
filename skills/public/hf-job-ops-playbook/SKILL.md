---
name: hf-job-ops-playbook
description: Operate Hugging Face Jobs reliably for training and evaluation in this repository, with reproducible submission records, monitoring, and retry flow. Use when submitting, monitoring, or triaging HF Jobs.
---

# Hf Job Ops Playbook

Use this skill for end-to-end Hugging Face Jobs operations in `Mistral_Hackathon`.

## When To Use

- User asks to submit training or eval jobs on Hugging Face.
- User asks for job status, logs, failure triage, or rerun.
- You need reproducible submission ledgers and job ids.

## Preflight

1. Authenticate CLI:
- `hf auth whoami`

2. Confirm required secrets are set:
- `HF_TOKEN`
- `WANDB_API_KEY` (if W&B logging is enabled)
- `MISTRAL_API_KEY` (if Mistral API fallback is used by eval)

3. Run dry-run first:
- SFT: `node scripts/hf/submit_sft_job.mjs`
- Eval: `node scripts/hf/submit_eval_job.mjs`

## Submit Workflow

1. Submit SFT job.
- `HF_JOB_SUBMIT=true node scripts/hf/submit_sft_job.mjs`
- Record is appended to `artifacts/hf_jobs/submissions.jsonl`.

2. Submit eval job.
- `HF_EVAL_JOB_SUBMIT=true node scripts/hf/submit_eval_job.mjs`
- Record is appended to `artifacts/hf_jobs/eval_submissions.jsonl`.

3. Capture job id from CLI and ledger.
- If id parse fails, keep `stdout_tail` and `stderr_tail` for manual lookup.

## Monitoring Workflow

Use HF CLI directly:

- Running jobs: `hf jobs ps --namespace Haruk1y`
- All jobs: `hf jobs ps -a --namespace Haruk1y`
- Inspect: `hf jobs inspect <job_id>`
- Logs: `hf jobs logs -f <job_id>`
- Usage stats: `hf jobs stats <job_id>`

## Failure Triage

1. Submission failure:
- Check `hf auth whoami`.
- Verify required secrets and env vars.
- Re-run dry-run command and inspect built command string.

2. Runtime failure:
- Inspect logs and classify as data, dependency, OOM, or auth issue.
- Apply smallest config change needed and re-submit.

3. Post-run artifact missing:
- Confirm `push_to_hub` and output model id settings.
- Confirm local metrics files exist in output dir.

## Repository Mapping

- SFT submission: `scripts/hf/submit_sft_job.mjs`
- Eval submission: `scripts/hf/submit_eval_job.mjs`
- Training script: `scripts/hf/train_sft_request_to_hidden_lm.py`
- Eval runner: `scripts/wandb/weave_eval_runner.py`
- Submission ledgers: `artifacts/hf_jobs/submissions.jsonl`, `artifacts/hf_jobs/eval_submissions.jsonl`

## References

- `references/job-ops-checklist.md`
