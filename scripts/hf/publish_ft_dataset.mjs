import { spawn } from "node:child_process";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const datasetRepo = process.env.HF_FT_DATASET_REPO_ID || "mistral-hackaton-2026/atelier-kotone-ft-request-hidden";
const shouldUpload = process.env.HF_UPLOAD_SUBMIT === "true";

const files = [
  "data/ft/ft_request_param_train.jsonl",
  "data/ft/ft_request_param_valid.jsonl",
  "data/ft/ft_request_param_test.jsonl",
  "data/eval/frozen_eval_set.v1.json"
];

const run = async () => {
  const commands = [
    ["repo", "create", datasetRepo, "--type", "dataset", "--exist-ok"],
    ...files.map((file) => ["upload", datasetRepo, resolve(root, file), file, "--repo-type", "dataset"])
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
