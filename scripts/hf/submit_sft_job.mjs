import { spawn } from "node:child_process";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const scriptPath = resolve(root, "scripts/hf/train_sft_request_to_hidden.py");

const flavor = process.env.HF_JOB_FLAVOR || "a10g-small";
const timeout = process.env.HF_JOB_TIMEOUT || "2h";
const namespace = process.env.HF_NAMESPACE || "mistral-hackaton-2026";
const shouldSubmit = process.env.HF_JOB_SUBMIT === "true";

const baseModel = process.env.HF_BASE_MODEL_ID || "mistralai/Ministral-3-3B-Instruct-2512";
const datasetRepo = process.env.HF_FT_DATASET_REPO_ID || "";
const trainSplit = process.env.HF_FT_TRAIN_SPLIT || "train";
const validSplit = process.env.HF_FT_VALID_SPLIT || "validation";
const outputModelId = process.env.HF_FT_OUTPUT_MODEL_ID || "mistral-hackaton-2026/atelier-kotone-ministral3b-ft";
const runName = process.env.HF_FT_RUN_NAME || `ministral3b-sft-${Date.now()}`;

const envArgs = [
  "--env",
  `HF_BASE_MODEL_ID=${baseModel}`,
  "--env",
  `HF_FT_DATASET_REPO_ID=${datasetRepo}`,
  "--env",
  `HF_FT_TRAIN_SPLIT=${trainSplit}`,
  "--env",
  `HF_FT_VALID_SPLIT=${validSplit}`,
  "--env",
  `HF_FT_OUTPUT_MODEL_ID=${outputModelId}`,
  "--env",
  `HF_FT_RUN_NAME=${runName}`,
  "--env",
  `WANDB_PROJECT=${process.env.WANDB_PROJECT || "atelier-kotone-ft"}`
];

const secretArgs = ["--secrets", "HF_TOKEN", "--secrets", "WANDB_API_KEY"];

const args = [
  "jobs",
  "uv",
  "run",
  scriptPath,
  "--flavor",
  flavor,
  "--timeout",
  timeout,
  "--namespace",
  namespace,
  "--detach",
  ...envArgs,
  ...secretArgs
];

const printable = `hf ${args.map((x) => (x.includes(" ") ? JSON.stringify(x) : x)).join(" ")}`;

if (!shouldSubmit) {
  console.log("[DRY-RUN] Set HF_JOB_SUBMIT=true to actually submit.");
  console.log(printable);
  process.exit(0);
}

const child = spawn("hf", args, {
  cwd: root,
  stdio: "inherit",
  env: process.env
});

child.on("exit", (code) => {
  process.exit(code ?? 1);
});
