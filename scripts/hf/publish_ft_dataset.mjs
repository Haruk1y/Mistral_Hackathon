import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);
const datasetRepo = process.env.HF_FT_DATASET_REPO_ID || "mistral-hackaton-2026/atelier-kotone-ft-request-hidden";
const shouldUpload = process.env.HF_UPLOAD_SUBMIT === "true";
const readmePath = resolve(root, "artifacts/hf/dataset_README.md");

const files = [
  { local: "data/ft/ft_request_param_train.jsonl", remote: "train.jsonl" },
  { local: "data/ft/ft_request_param_valid.jsonl", remote: "validation.jsonl" },
  { local: "data/ft/ft_request_param_test.jsonl", remote: "test.jsonl" },
  { local: "data/ft/ft_split_stats.json", remote: "ft_split_stats.json" },
  { local: "artifacts/hf/dataset_README.md", remote: "README.md" }
];

const countLines = async (path) => {
  const raw = await readFile(path, "utf8");
  return raw.split("\n").filter((line) => line.trim().length > 0).length;
};

const buildReadme = async () => {
  const trainCount = await countLines(resolve(root, "data/ft/ft_request_param_train.jsonl"));
  const validCount = await countLines(resolve(root, "data/ft/ft_request_param_valid.jsonl"));
  const testCount = await countLines(resolve(root, "data/ft/ft_request_param_test.jsonl"));

  const body = `---
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
  - split: test
    path: test.jsonl
---

# Atelier kotone FT Dataset

This dataset is for hidden-parameter regression from:

- \`request_text\`
- \`rule_prompt\`

Each JSONL line is exactly one sample.

## Splits

- train: ${trainCount}
- validation: ${validCount}
- test: ${testCount}

## Columns

- \`source_type\`: \`request_text\` or \`rule_prompt\`
- \`request_text\`: model input text
- \`target_hidden_params\`: supervision object
- \`messages\`: chat-style format for compatibility
`;

  await mkdir(resolve(root, "artifacts/hf"), { recursive: true });
  await writeFile(readmePath, body);
};

const run = async () => {
  await buildReadme();

  const commands = [
    ["repo", "create", datasetRepo, "--repo-type", "dataset", "--exist-ok"],
    ...files.map((file) => [
      "upload",
      datasetRepo,
      resolve(root, file.local),
      file.remote,
      "--repo-type",
      "dataset"
    ])
  ];

  if (!shouldUpload) {
    console.log("[DRY-RUN] Set HF_UPLOAD_SUBMIT=true to upload dataset artifacts.");
    for (const cmd of commands) {
      console.log(`hf ${cmd.map((x) => (x.includes(" ") ? JSON.stringify(x) : x)).join(" ")}`);
    }
    return;
  }

  for (const cmd of commands) {
    await new Promise((resolvePromise, rejectPromise) => {
      const child = spawn("hf", cmd, {
        cwd: root,
        stdio: "inherit",
        env: process.env
      });

      child.on("exit", (code) => {
        if (code === 0) {
          resolvePromise(undefined);
          return;
        }
        rejectPromise(new Error(`hf ${cmd.join(" ")} failed with code ${code}`));
      });
    });
  }
};

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
