# HF Job Ops Checklist

## Pre-Submit

- Confirm `hf auth whoami` succeeds.
- Confirm target namespace and model ids.
- Confirm required secrets are configured:
- `HF_TOKEN`
- `WANDB_API_KEY` when W&B enabled
- `MISTRAL_API_KEY` when API fallback enabled
- Run dry-run submit commands before real submission.

## Submit Commands

- SFT:
- `HF_JOB_SUBMIT=true node scripts/hf/submit_sft_job.mjs`
- Eval:
- `HF_EVAL_JOB_SUBMIT=true node scripts/hf/submit_eval_job.mjs`

## Monitoring Commands

- `hf jobs ps --namespace Haruk1y`
- `hf jobs ps -a --namespace Haruk1y`
- `hf jobs inspect <job_id>`
- `hf jobs logs -f <job_id>`
- `hf jobs stats <job_id>`

## Ledger Files

- SFT submissions: `artifacts/hf_jobs/submissions.jsonl`
- Eval submissions: `artifacts/hf_jobs/eval_submissions.jsonl`

Each row should keep at minimum:

- submission timestamp
- run name
- output model id
- dataset repo id
- submitted command
- parsed job id
- exit code
- stdout/stderr tail

## Retry Policy

Retry once immediately for transient infra failures:

- temporary pull failure
- transient network/timeout
- temporary hardware assignment issue

Do not auto-retry without change for deterministic failures:

- auth/permission
- bad dataset path
- invalid model id
- script/runtime exception

For deterministic failures, patch config first, then re-submit.
