import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const shouldSubmit = process.env.BALANCED_CAMPAIGN_SUBMIT === "true";
const campaignName = process.env.BALANCED_CAMPAIGN_NAME || "balanced_6run";

const cycle1Dataset = process.env.CYCLE1_DATASET_REPO_ID || process.env.HF_FT_DATASET_REPO_ID || "";
const cycle2Dataset = process.env.CYCLE2_DATASET_REPO_ID || cycle1Dataset;

const runs = [
  {
    id: "run_0_baselines_only",
    type: "eval_only",
    note: "rule + prompt baseline evaluation",
  },
  {
    id: "run_1_ft_cycle1_default",
    type: "train",
    env: {
      HF_FT_RUN_NAME: "balanced-run1-cycle1-default",
      HF_FT_DATASET_REPO_ID: cycle1Dataset,
      LOOP_CYCLE_ID: "cycle_1",
      WANDB_RUN_GROUP: `${campaignName}-cycle1`,
    },
  },
  {
    id: "run_2_ft_cycle1_tuned",
    type: "train",
    env: {
      HF_FT_RUN_NAME: "balanced-run2-cycle1-tuned",
      HF_FT_DATASET_REPO_ID: cycle1Dataset,
      LOOP_CYCLE_ID: "cycle_1",
      WANDB_RUN_GROUP: `${campaignName}-cycle1`,
      HF_FT_LR: process.env.CYCLE1_TUNED_LR || "0.000012",
      HF_FT_EPOCHS: process.env.CYCLE1_TUNED_EPOCHS || "3",
    },
  },
  {
    id: "run_3_ft_cycle1_tuned_augmented",
    type: "train",
    env: {
      HF_FT_RUN_NAME: "balanced-run3-cycle1-tuned-augmented",
      HF_FT_DATASET_REPO_ID: process.env.CYCLE1_AUG_DATASET_REPO_ID || cycle1Dataset,
      LOOP_CYCLE_ID: "cycle_1",
      WANDB_RUN_GROUP: `${campaignName}-cycle1`,
      HF_FT_LR: process.env.CYCLE1_TUNED_LR || "0.000012",
      HF_FT_EPOCHS: process.env.CYCLE1_TUNED_EPOCHS || "3",
    },
  },
  {
    id: "run_4_ft_cycle2_tuned",
    type: "train",
    env: {
      HF_FT_RUN_NAME: "balanced-run4-cycle2-tuned",
      HF_FT_DATASET_REPO_ID: cycle2Dataset,
      LOOP_CYCLE_ID: "cycle_2",
      WANDB_RUN_GROUP: `${campaignName}-cycle2`,
      HF_FT_LR: process.env.CYCLE2_TUNED_LR || "0.00001",
      HF_FT_EPOCHS: process.env.CYCLE2_TUNED_EPOCHS || "3",
    },
  },
  {
    id: "run_5_ft_cycle2_tuned_augmented",
    type: "train",
    env: {
      HF_FT_RUN_NAME: "balanced-run5-cycle2-tuned-augmented",
      HF_FT_DATASET_REPO_ID: process.env.CYCLE2_AUG_DATASET_REPO_ID || cycle2Dataset,
      LOOP_CYCLE_ID: "cycle_2",
      WANDB_RUN_GROUP: `${campaignName}-cycle2`,
      HF_FT_LR: process.env.CYCLE2_TUNED_LR || "0.00001",
      HF_FT_EPOCHS: process.env.CYCLE2_TUNED_EPOCHS || "3",
    },
  },
];

const timestamp = new Date().toISOString();
const outDir = resolve(root, "artifacts/hf_jobs");
const manifestPath = resolve(outDir, `${campaignName}.manifest.json`);

const runSubmit = (envPatch) => {
  const env = { ...process.env, ...envPatch, HF_JOB_SUBMIT: "true" };
  const child = spawnSync("npm", ["run", "hf:job:submit"], {
    cwd: root,
    env,
    encoding: "utf8",
  });
  if (child.stdout) process.stdout.write(child.stdout);
  if (child.stderr) process.stderr.write(child.stderr);
  return child.status ?? 1;
};

const main = async () => {
  await mkdir(outDir, { recursive: true });
  const manifest = {
    generated_at: timestamp,
    campaign: campaignName,
    submit_enabled: shouldSubmit,
    runs,
  };
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2));

  if (!shouldSubmit) {
    console.log("[DRY-RUN] Set BALANCED_CAMPAIGN_SUBMIT=true to submit HF jobs.");
    console.log(`Manifest saved: ${manifestPath}`);
    for (const run of runs) {
      if (run.type !== "train") continue;
      const args = Object.entries(run.env || {})
        .map(([key, value]) => `${key}=${JSON.stringify(String(value))}`)
        .join(" ");
      console.log(`${args} HF_JOB_SUBMIT=true npm run hf:job:submit`);
    }
    return;
  }

  for (const run of runs) {
    if (run.type !== "train") continue;
    console.log(`Submitting ${run.id} ...`);
    const status = runSubmit(run.env || {});
    if (status !== 0) {
      throw new Error(`Submission failed for ${run.id} with code ${status}`);
    }
  }

  console.log(`Completed campaign submissions. Manifest: ${manifestPath}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
