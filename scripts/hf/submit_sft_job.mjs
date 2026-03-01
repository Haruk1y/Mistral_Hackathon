import { appendFile, mkdir } from "node:fs/promises";
import { execSync, spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
const submissionLogPath = resolve(root, "artifacts/hf_jobs/submissions.jsonl");
loadEnvFiles(root);

const flavor = process.env.HF_JOB_FLAVOR || "a10g-small";
const timeout = process.env.HF_JOB_TIMEOUT || "2h";
const namespace = process.env.HF_NAMESPACE || "Haruk1y";
const shouldSubmit = process.env.HF_JOB_SUBMIT === "true";
const ftObjective = process.env.HF_FT_OBJECTIVE || "next_token_json_sft";
const trainScriptRelPath =
  ftObjective === "mse_regression_head"
    ? "scripts/hf/train_sft_request_to_hidden.py"
    : "scripts/hf/train_sft_request_to_hidden_lm.py";
const scriptPath = resolve(root, trainScriptRelPath);

const envConfig = {
  HF_BASE_MODEL_ID: process.env.HF_BASE_MODEL_ID || "mistralai/Ministral-3-3B-Instruct-2512",
  HF_FT_DATASET_REPO_ID: process.env.HF_FT_DATASET_REPO_ID || "",
  HF_FT_DATASET_CONFIG: process.env.HF_FT_DATASET_CONFIG || "",
  HF_FT_TRAIN_SPLIT: process.env.HF_FT_TRAIN_SPLIT || "train",
  HF_FT_VALID_SPLIT: process.env.HF_FT_VALID_SPLIT || "validation",
  HF_FT_OUTPUT_MODEL_ID: process.env.HF_FT_OUTPUT_MODEL_ID || "Haruk1y/atelier-kotone-ministral3b-ft",
  HF_FT_OUTPUT_DIR: process.env.HF_FT_OUTPUT_DIR || "outputs/ministral3b-request-hidden",
  HF_FT_INIT_ADAPTER_MODEL_ID: process.env.HF_FT_INIT_ADAPTER_MODEL_ID || "",
  HF_FT_RUN_NAME: process.env.HF_FT_RUN_NAME || `ministral3b-nexttoken-${Date.now()}`,
  HF_FT_OBJECTIVE: ftObjective,
  HF_FT_EPOCHS: process.env.HF_FT_EPOCHS || "2",
  HF_FT_LR: process.env.HF_FT_LR || "0.00002",
  HF_FT_BATCH_SIZE: process.env.HF_FT_BATCH_SIZE || "2",
  HF_FT_GRAD_ACCUM: process.env.HF_FT_GRAD_ACCUM || "8",
  HF_FT_WARMUP_RATIO: process.env.HF_FT_WARMUP_RATIO || "0.1",
  HF_FT_MAX_LENGTH: process.env.HF_FT_MAX_LENGTH || "768",
  HF_FT_TARGET_SCALE: process.env.HF_FT_TARGET_SCALE || process.env.FT_TARGET_SCALE || "10",
  HF_FT_LOGGING_STEPS: process.env.HF_FT_LOGGING_STEPS || "1",
  HF_FT_EVAL_STEPS: process.env.HF_FT_EVAL_STEPS || "25",
  HF_FT_DETAILED_EVAL_STEPS: process.env.HF_FT_DETAILED_EVAL_STEPS || "50",
  HF_FT_DETAILED_EVAL_MAX_SAMPLES: process.env.HF_FT_DETAILED_EVAL_MAX_SAMPLES || "25",
  HF_FT_GENERATION_MAX_NEW_TOKENS: process.env.HF_FT_GENERATION_MAX_NEW_TOKENS || "96",
  HF_FT_MAX_STEPS: process.env.HF_FT_MAX_STEPS || "-1",
  HF_FT_HARD_CASE_TOP_K: process.env.HF_FT_HARD_CASE_TOP_K || "80",
  HF_FT_LORA_R: process.env.HF_FT_LORA_R || "16",
  HF_FT_LORA_ALPHA: process.env.HF_FT_LORA_ALPHA || "32",
  HF_FT_LORA_DROPOUT: process.env.HF_FT_LORA_DROPOUT || "0.05",
  HF_FT_PUSH_TO_HUB: process.env.HF_FT_PUSH_TO_HUB || "true",
  FT_DATASET_VERSION: process.env.FT_DATASET_VERSION || "v1",
  FT_SOURCE_TYPE_MIX: process.env.FT_SOURCE_TYPE_MIX || "request_text+rule_prompt",
  WANDB_PROJECT: process.env.WANDB_PROJECT || "atelier-kotone-ft",
  WANDB_ENTITY: process.env.WANDB_ENTITY || "",
  WANDB_RUN_GROUP: process.env.WANDB_RUN_GROUP || "hf-ft-balanced",
  WEAVE_PROJECT: process.env.WEAVE_PROJECT || "atelier-kotone-weave",
  ENABLE_WEAVE_TRACE: process.env.ENABLE_WEAVE_TRACE || "true",
  ENABLE_TRACKIO: process.env.ENABLE_TRACKIO || "false",
  TRACKIO_PROJECT: process.env.TRACKIO_PROJECT || "atelier-kotone-ft",
  TRACKIO_SPACE_ID: process.env.TRACKIO_SPACE_ID || "",
};

const envArgs = Object.entries(envConfig).flatMap(([key, value]) => ["--env", `${key}=${value}`]);
const secretArgs = ["--secrets", "HF_TOKEN", "--secrets", "WANDB_API_KEY"];

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
  ...secretArgs,
  scriptPath,
];

const printable = `hf ${args.map((x) => (x.includes(" ") ? JSON.stringify(x) : x)).join(" ")}`;

const extractJobId = (text) => {
  const patterns = [
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
  console.log("[DRY-RUN] Set HF_JOB_SUBMIT=true to actually submit.");
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
  objective: ftObjective,
  train_script: trainScriptRelPath,
  run_name: envConfig.HF_FT_RUN_NAME,
  output_model_id: envConfig.HF_FT_OUTPUT_MODEL_ID,
  dataset_repo_id: envConfig.HF_FT_DATASET_REPO_ID,
  command: printable,
  exit_code: result.status ?? 1,
  job_id: jobId,
  stdout_tail: (result.stdout ?? "").split("\n").slice(-30).join("\n"),
  stderr_tail: (result.stderr ?? "").split("\n").slice(-30).join("\n"),
};
await writeSubmissionRecord(record);

if (result.status !== 0) {
  console.error(`HF job submission failed. Logged to ${submissionLogPath}`);
  process.exit(result.status ?? 1);
}

if (jobId) {
  console.log(`Submitted HF Job ID: ${jobId}`);
} else {
  console.log("Submitted HF job, but job_id could not be parsed from CLI output.");
}
console.log(`Submission record saved: ${submissionLogPath}`);
