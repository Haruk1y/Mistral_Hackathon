import { appendFile, mkdir } from "node:fs/promises";
import { execSync, spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
const submissionLogPath = resolve(root, "artifacts/hf_jobs/eval_submissions.jsonl");
loadEnvFiles(root);

const flavor = process.env.HF_EVAL_JOB_FLAVOR || process.env.HF_JOB_FLAVOR || "a10g-small";
const timeout = process.env.HF_EVAL_JOB_TIMEOUT || "2h";
const namespace = process.env.HF_NAMESPACE || "Haruk1y";
const shouldSubmit = process.env.HF_EVAL_JOB_SUBMIT === "true";
const scriptPath = resolve(root, "scripts/wandb/weave_eval_runner.py");

const withPackages = (
  process.env.HF_EVAL_JOB_WITH_PACKAGES ||
  "torch>=2.6.0,transformers>=4.52.0,peft>=0.13.0,accelerate>=1.2.0,datasets>=3.2.0"
)
  .split(",")
  .map((value) => value.trim())
  .filter((value) => value.length > 0);

const envConfig = {
  EVAL_MODE: process.env.EVAL_MODE || "fine_tuned",
  EVAL_DATASET_PATH: process.env.EVAL_DATASET_PATH || "data/eval/frozen_eval_set.v1.json",
  EVAL_DATASET_REPO_ID: process.env.EVAL_DATASET_REPO_ID || process.env.HF_FT_DATASET_REPO_ID || "Haruk1y/atelier-kotone-ft-request-hidden",
  EVAL_DATASET_CONFIG: process.env.EVAL_DATASET_CONFIG || process.env.HF_FT_DATASET_CONFIG || "",
  EVAL_DATASET_SPLIT: process.env.EVAL_DATASET_SPLIT || "test",
  EVAL_DATASET_MAX_SAMPLES: process.env.EVAL_DATASET_MAX_SAMPLES || "0",
  EVAL_TARGET_SCALE: process.env.EVAL_TARGET_SCALE || "10",
  EVAL_TOP_FAILURES: process.env.EVAL_TOP_FAILURES || "20",
  EVAL_WEAVE_ENABLED: process.env.EVAL_WEAVE_ENABLED || "true",
  EVAL_WANDB_ENABLED: process.env.EVAL_WANDB_ENABLED || "true",
  HF_INFERENCE_BACKEND: process.env.HF_INFERENCE_BACKEND || "local_transformers",
  EVAL_LOCAL_BASE_MODEL_ID: process.env.EVAL_LOCAL_BASE_MODEL_ID || process.env.HF_BASE_MODEL_ID || "mistralai/Ministral-3-3B-Instruct-2512",
  EVAL_LOCAL_ADAPTER_MODEL_ID: process.env.EVAL_LOCAL_ADAPTER_MODEL_ID || process.env.EVAL_FINE_TUNED_MODEL_ID || process.env.HF_FT_OUTPUT_MODEL_ID || "",
  EVAL_LOCAL_FINE_TUNED_IS_ADAPTER: process.env.EVAL_LOCAL_FINE_TUNED_IS_ADAPTER || "true",
  EVAL_LOCAL_DEVICE: process.env.EVAL_LOCAL_DEVICE || "auto",
  EVAL_LOCAL_DTYPE: process.env.EVAL_LOCAL_DTYPE || "auto",
  EVAL_LOCAL_MAX_NEW_TOKENS: process.env.EVAL_LOCAL_MAX_NEW_TOKENS || "96",
  EVAL_LOCAL_TRUST_REMOTE_CODE: process.env.EVAL_LOCAL_TRUST_REMOTE_CODE || "false",
  EVAL_LOCAL_CACHE_DIR: process.env.EVAL_LOCAL_CACHE_DIR || "",
  EVAL_PROMPT_BASELINE_MODEL_ID:
    process.env.EVAL_PROMPT_BASELINE_MODEL_ID || process.env.HF_BASE_MODEL_ID || "mistralai/Ministral-3-3B-Instruct-2512",
  EVAL_FINE_TUNED_MODEL_ID: process.env.EVAL_FINE_TUNED_MODEL_ID || process.env.HF_FT_OUTPUT_MODEL_ID || "",
  EVAL_LARGE_BASELINE_MODEL_ID: process.env.EVAL_LARGE_BASELINE_MODEL_ID || process.env.MISTRAL_LARGE_MODEL_ID || "mistral-large-latest",
  EVAL_REQUIRE_HF_DIRECT: process.env.EVAL_REQUIRE_HF_DIRECT || "false",
  EVAL_MISTRAL_FALLBACK_ENABLED: process.env.EVAL_MISTRAL_FALLBACK_ENABLED || "false",
  WANDB_PROJECT: process.env.WANDB_PROJECT || "atelier-kotone-ft",
  WANDB_ENTITY: process.env.WANDB_ENTITY || "",
  WANDB_RUN_GROUP: process.env.WANDB_RUN_GROUP || "hf-eval-local-transformers",
  WEAVE_PROJECT: process.env.WEAVE_PROJECT || process.env.WANDB_PROJECT || "atelier-kotone-ft",
};

const envArgs = Object.entries(envConfig).flatMap(([key, value]) => ["--env", `${key}=${value}`]);
const withArgs = withPackages.flatMap((pkg) => ["--with", pkg]);
const secretNames = ["HF_TOKEN", "WANDB_API_KEY", "MISTRAL_API_KEY"].filter(
  (name) => typeof process.env[name] === "string" && process.env[name].length > 0,
);
const secretArgs = secretNames.flatMap((name) => ["--secrets", name]);

const args = [
  "jobs",
  "uv",
  "run",
  "--flavor",
  flavor,
  "--timeout",
  timeout,
  "--namespace",
  namespace,
  "--detach",
  ...envArgs,
  ...withArgs,
  ...secretArgs,
  scriptPath,
];

const printable = `hf ${args.map((x) => (x.includes(" ") ? JSON.stringify(x) : x)).join(" ")}`;

const extractJobId = (text) => {
  const patterns = [
    /job started with id:\s*([a-zA-Z0-9-]{8,})/i,
    /\bjob[_ -]?id[:= ]+([a-zA-Z0-9-]{8,})/i,
    /https:\/\/huggingface\.co\/jobs\/([a-zA-Z0-9-]{8,})/i,
    /\b([a-zA-Z0-9]{8,}-[a-zA-Z0-9-]{8,})\b/,
  ];
  for (const pattern of patterns) {
    const matched = text.match(pattern);
    if (matched?.[1]) {
      return matched[1];
    }
  }
  return null;
};

const writeSubmissionRecord = async (record) => {
  await mkdir(resolve(root, "artifacts/hf_jobs"), { recursive: true });
  await appendFile(submissionLogPath, `${JSON.stringify(record)}\n`, "utf8");
};

if (!shouldSubmit) {
  console.log("[DRY-RUN] Set HF_EVAL_JOB_SUBMIT=true to actually submit.");
  console.log(printable);
  process.exit(0);
}

try {
  const whoami = execSync("hf auth whoami", { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
  if (!whoami || whoami.includes("Not logged in")) {
    console.error("HF CLI is not authenticated. Run: hf auth login");
    process.exit(1);
  }
} catch {
  console.error("HF CLI is not authenticated. Run: hf auth login");
  process.exit(1);
}

const runAt = new Date().toISOString();
const result = spawnSync("hf", args, {
  cwd: root,
  env: process.env,
  encoding: "utf8",
});

if (result.stdout) process.stdout.write(result.stdout);
if (result.stderr) process.stderr.write(result.stderr);

const mixedOutput = `${result.stdout ?? ""}\n${result.stderr ?? ""}`;
const jobId = extractJobId(mixedOutput);

const record = {
  submitted_at: runAt,
  namespace,
  flavor,
  timeout,
  command: printable,
  mode: envConfig.EVAL_MODE,
  backend: envConfig.HF_INFERENCE_BACKEND,
  dataset_path: envConfig.EVAL_DATASET_PATH,
  local_base_model_id: envConfig.EVAL_LOCAL_BASE_MODEL_ID,
  local_adapter_model_id: envConfig.EVAL_LOCAL_ADAPTER_MODEL_ID,
  exit_code: result.status ?? 1,
  job_id: jobId,
  stdout_tail: (result.stdout ?? "").split("\n").slice(-30).join("\n"),
  stderr_tail: (result.stderr ?? "").split("\n").slice(-30).join("\n"),
};
await writeSubmissionRecord(record);

if (result.status !== 0) {
  console.error(`HF eval job submission failed. Logged to ${submissionLogPath}`);
  process.exit(result.status ?? 1);
}

if (jobId) {
  console.log(`Submitted HF Eval Job ID: ${jobId}`);
} else {
  console.log("Submitted HF eval job, but job_id could not be parsed from CLI output.");
}
console.log(`Submission record saved: ${submissionLogPath}`);
