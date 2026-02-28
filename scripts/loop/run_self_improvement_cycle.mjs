import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const summaryPath = resolve(root, "artifacts/eval/summary/latest_summary.json");
const loopRoot = resolve(root, "artifacts/loop");

const now = new Date().toISOString();
const cycle = process.env.LOOP_CYCLE_ID || "cycle_1";
const cycleDir = resolve(loopRoot, cycle);

const asNumber = (value, fallback) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);

const loadSummary = async () => JSON.parse(await readFile(summaryPath, "utf8"));

const deriveNextHparams = (summary) => {
  const prompt = summary.latest_by_mode?.prompt_baseline?.metrics ?? {};
  const ft = summary.latest_by_mode?.fine_tuned?.metrics ?? {};

  const current = {
    learning_rate: asNumber(Number(process.env.HF_FT_LR), 2e-5),
    lora_r: asNumber(Number(process.env.HF_FT_LORA_R), 16),
    lora_alpha: asNumber(Number(process.env.HF_FT_LORA_ALPHA), 32),
    lora_dropout: asNumber(Number(process.env.HF_FT_LORA_DROPOUT), 0.05),
    epochs: asNumber(Number(process.env.HF_FT_EPOCHS), 2),
  };

  const next = { ...current };
  const reasons = [];

  const jsonValid = asNumber(ft.json_valid_rate, 0);
  if (jsonValid < 0.98) {
    next.learning_rate = Math.max(1e-6, current.learning_rate * 0.6);
    next.epochs = Math.min(3, current.epochs + 1);
    reasons.push("json_valid_rate below target -> lower LR and increase epochs");
  }

  const mseNorm = asNumber(ft.mse_norm, 0);
  const promptMseNorm = asNumber(prompt.mse_norm, 0);
  if (mseNorm >= promptMseNorm && promptMseNorm > 0) {
    next.lora_r = current.lora_r >= 32 ? 32 : current.lora_r === 16 ? 32 : 16;
    next.lora_alpha = next.lora_r * 2;
    reasons.push("mse_norm not improved vs prompt baseline -> increase LoRA rank");
  }

  const p95 = asNumber(ft.p95_inference_latency_ms, 0);
  if (p95 > 1200) {
    next.lora_r = Math.min(next.lora_r, current.lora_r);
    reasons.push("p95 latency above target -> avoid larger adapters");
  }

  if (!reasons.length) {
    reasons.push("all core metrics improving -> keep conservative update");
  }

  return { current, next, reasons };
};

const deriveAugmentationSpec = (summary) => {
  const delta = summary.auto_improvement_delta ?? {};

  const focusDims = [];
  if (asNumber(delta.vector_mae, 0) >= 0) {
    focusDims.push("nostalgia", "brightness");
  }
  if (focusDims.length === 0) {
    focusDims.push("energy", "complexity");
  }

  return {
    dataset_version_from: process.env.FT_DATASET_VERSION_FROM || "v1",
    dataset_version_to: process.env.FT_DATASET_VERSION_TO || "v2",
    focus_dims: focusDims,
    focus_buckets: {
      nostalgia: [70, 95],
      brightness: [0, 30],
      energy: [70, 95],
      complexity: [0, 30]
    },
    add_ratio: 0.2,
    hard_case_replay_ratio: 0.15,
    teacher_model: process.env.MISTRAL_TEACHER_MODEL || "mistral-large-latest",
    filters: {
      deny_numeric_leak: true,
      deduplicate: true,
      language_allow: ["ja", "en"]
    }
  };
};

const toYaml = (obj) => {
  const lines = [];
  for (const [key, value] of Object.entries(obj)) {
    lines.push(`${key}: ${value}`);
  }
  return `${lines.join("\n")}\n`;
};

const main = async () => {
  const summary = await loadSummary();
  const hparams = deriveNextHparams(summary);
  const augmentation = deriveAugmentationSpec(summary);

  await mkdir(cycleDir, { recursive: true });

  await writeFile(resolve(cycleDir, "next_hparams.yaml"), toYaml(hparams.next));
  await writeFile(resolve(cycleDir, "augmentation_spec.json"), JSON.stringify(augmentation, null, 2));

  const decisionLog = [
    `# Loop Decision Log: ${cycle}`,
    "",
    `Generated at: ${now}`,
    "",
    "## Current HParams",
    "```json",
    JSON.stringify(hparams.current, null, 2),
    "```",
    "",
    "## Next HParams",
    "```json",
    JSON.stringify(hparams.next, null, 2),
    "```",
    "",
    "## Reasons",
    ...hparams.reasons.map((reason) => `- ${reason}`),
    "",
    "## Augmentation Spec",
    "```json",
    JSON.stringify(augmentation, null, 2),
    "```",
  ].join("\n");

  await writeFile(resolve(cycleDir, "decision_log.md"), decisionLog);

  const summaryPayload = {
    cycle,
    completed: true,
    generated_at: now,
    loop_completion_rate_hint: 1.0,
    auto_improvement_delta: summary.auto_improvement_delta ?? null
  };

  await writeFile(resolve(cycleDir, "summary.json"), JSON.stringify(summaryPayload, null, 2));

  console.log(`Generated loop artifacts in ${cycleDir}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
