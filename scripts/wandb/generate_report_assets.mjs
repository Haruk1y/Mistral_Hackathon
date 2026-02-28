import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const summaryPath = resolve(root, "artifacts/eval/summary/latest_summary.json");
const outDir = resolve(root, "artifacts/wandb");

const fmt = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
};

const metricKeys = [
  "json_valid_rate",
  "vector_mae",
  "mse_raw",
  "mse_norm",
  "constraint_match_rate",
  "slot_exact_match",
  "intent_score_mean",
  "output_sanity_score",
  "p95_inference_latency_ms",
  "cost_per_100_requests_usd"
];

const scoreDelta = (higherBetter, ft, baseline) => {
  if (typeof ft !== "number" || typeof baseline !== "number") return 0;
  return higherBetter ? ft - baseline : baseline - ft;
};

const main = async () => {
  const summary = JSON.parse(await readFile(summaryPath, "utf8"));
  const latest = summary.latest_by_mode || {};

  const rule = latest.rule_baseline?.metrics || {};
  const prompt = latest.prompt_baseline?.metrics || {};
  const ft = latest.fine_tuned?.metrics || {};

  const csvRows = ["metric,rule_baseline,prompt_baseline,fine_tuned,ft_vs_prompt_delta"];
  for (const key of metricKeys) {
    const higherBetter = !["vector_mae", "mse_raw", "mse_norm", "p95_inference_latency_ms", "cost_per_100_requests_usd"].includes(key);
    const delta = scoreDelta(higherBetter, ft[key], prompt[key]);
    csvRows.push(`${key},${fmt(rule[key])},${fmt(prompt[key])},${fmt(ft[key])},${fmt(delta)}`);
  }

  const reportMd = [
    "# W&B Report Draft Assets",
    "",
    `Generated at: ${new Date().toISOString()}`,
    "",
    "## Core Comparison (Prompt Baseline vs Fine-Tuned)",
    "",
    "| Metric | Prompt Baseline | Fine-Tuned | Delta |",
    "|---|---:|---:|---:|",
    ...metricKeys.map((key) => {
      const higherBetter = !["vector_mae", "mse_raw", "mse_norm", "p95_inference_latency_ms", "cost_per_100_requests_usd"].includes(key);
      const delta = scoreDelta(higherBetter, ft[key], prompt[key]);
      return `| ${key} | ${fmt(prompt[key])} | ${fmt(ft[key])} | ${fmt(delta)} |`;
    }),
    "",
    "## Loop Metrics",
    "",
    `- loop_completion_rate: ${fmt(summary.loop_completion_rate)}`,
    `- auto_improvement_delta.intent_score_mean: ${fmt(summary.auto_improvement_delta?.intent_score_mean)}`,
    `- auto_improvement_delta.vector_mae: ${fmt(summary.auto_improvement_delta?.vector_mae)}`,
    `- auto_improvement_delta.json_valid_rate: ${fmt(summary.auto_improvement_delta?.json_valid_rate)}`,
    "",
    "## Links To Fill",
    "",
    "- W&B Runs URL:",
    "- W&B Sweep URL:",
    "- W&B Report URL:",
    "- Weave Trace URL:",
    "- HF Model URL:",
    "- HF Dataset URL:",
  ];

  await mkdir(outDir, { recursive: true });
  await writeFile(resolve(outDir, "report_metrics.csv"), `${csvRows.join("\n")}\n`);
  await writeFile(resolve(outDir, "report_draft.md"), `${reportMd.join("\n")}\n`);

  console.log(`Wrote ${resolve(outDir, "report_metrics.csv")}`);
  console.log(`Wrote ${resolve(outDir, "report_draft.md")}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
