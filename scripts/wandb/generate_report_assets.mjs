import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { readdir } from "node:fs/promises";

const root = resolve(new URL("../../", import.meta.url).pathname);
const summaryPath = resolve(root, "artifacts/eval/summary/latest_summary.json");
const outDir = resolve(root, "artifacts/wandb");
const loopDir = resolve(root, "artifacts/loop");
const failureCasesPath = resolve(root, "artifacts/wandb/weave_failure_cases.json");
const cycleId = process.env.LOOP_CYCLE_ID || "";
const wandbEntity = process.env.WANDB_ENTITY || "haruk1y_";
const wandbProject = process.env.WANDB_PROJECT || "atelier-kotone-ft";
const hfModelUrl = process.env.HF_MODEL_URL || "https://huggingface.co/Haruk1y/atelier-kotone-ministral3b-ft";
const hfDatasetUrl =
  process.env.HF_DATASET_URL || "https://huggingface.co/datasets/Haruk1y/atelier-kotone-ft-request-hidden";

const fmt = (value) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
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

const extractTraceQuery = (url) => {
  if (typeof url !== "string" || url.length === 0) return null;
  try {
    const parsed = new URL(url);
    const value = parsed.searchParams.get("query");
    return value && value.length > 0 ? value : null;
  } catch {
    const match = url.match(/[?&]query=([^&]+)/u);
    return match?.[1] ? decodeURIComponent(match[1]) : null;
  }
};

const normalizeTraceUrls = (urls, entity, project) => {
  const out = [];
  const seen = new Set();
  for (const url of urls) {
    const query = extractTraceQuery(url);
    const normalized = query
      ? `https://wandb.ai/${entity}/${project}/weave/traces?query=${encodeURIComponent(query)}`
      : url;
    if (typeof normalized !== "string" || normalized.length === 0) continue;
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
};

const loadLatestLoopSummary = async () => {
  if (cycleId && /^cycle_\d+$/u.test(cycleId)) {
    try {
      const path = resolve(loopDir, cycleId, "summary.json");
      const raw = await readFile(path, "utf8");
      return { cycle: cycleId, path, payload: JSON.parse(raw) };
    } catch {
      // Fallback to latest available cycle summary.
    }
  }
  try {
    const dirs = await readdir(loopDir, { withFileTypes: true });
    const cycles = dirs
      .filter((entry) => entry.isDirectory() && /^cycle_\d+$/u.test(entry.name))
      .map((entry) => entry.name)
      .sort((a, b) => Number(a.replace("cycle_", "")) - Number(b.replace("cycle_", "")));
    if (cycles.length === 0) return null;
    const latest = cycles[cycles.length - 1];
    const path = resolve(loopDir, latest, "summary.json");
    const raw = await readFile(path, "utf8");
    return { cycle: latest, path, payload: JSON.parse(raw) };
  } catch {
    return null;
  }
};

const loadFailureCases = async () => {
  try {
    const payload = JSON.parse(await readFile(failureCasesPath, "utf8"));
    return payload;
  } catch {
    return null;
  }
};

const loadLoopMcpSnapshot = async (cycle) => {
  if (!cycle) return null;
  try {
    const path = resolve(loopDir, cycle, "mcp_eval_snapshot.json");
    const payload = JSON.parse(await readFile(path, "utf8"));
    return { path, payload };
  } catch {
    return null;
  }
};

const main = async () => {
  const summary = JSON.parse(await readFile(summaryPath, "utf8"));
  const loopSummary = await loadLatestLoopSummary();
  const failureCases = await loadFailureCases();
  const loopMcpSnapshot = await loadLoopMcpSnapshot(loopSummary?.cycle);
  const latest = summary.latest_by_mode || {};

  const rule = latest.rule_baseline?.metrics || {};
  const prompt = latest.prompt_baseline?.metrics || {};
  const ft = latest.fine_tuned?.metrics || {};
  const topDimRows = Array.isArray(failureCases?.aggregate_dim_abs_error)
    ? failureCases.aggregate_dim_abs_error.slice(0, 2)
    : [];
  const topFailureRows = Array.isArray(failureCases?.failure_cases) ? failureCases.failure_cases.slice(0, 3) : [];
  const recommendations = Array.isArray(failureCases?.recommendations) ? failureCases.recommendations.slice(0, 3) : [];
  const mcpTopTraceUrls = topFailureRows.map((row) => row?.trace_url).filter(Boolean);
  const traceProject = failureCases?.source_metadata?.trace_project_selected?.project || wandbProject;
  const normalizedMcpTopTraceUrls = normalizeTraceUrls(mcpTopTraceUrls, wandbEntity, traceProject);
  const normalizedFineTraceUrls = normalizeTraceUrls(
    (summary.latest_samples_by_mode?.fine_tuned?.top_trace_urls ?? []).slice(0, 3),
    wandbEntity,
    traceProject,
  );
  const normalizedPromptTraceUrls = normalizeTraceUrls(
    (summary.latest_samples_by_mode?.prompt_baseline?.top_trace_urls ?? []).slice(0, 3),
    wandbEntity,
    traceProject,
  );
  const wandbProjectUrl = `https://wandb.ai/${wandbEntity}/${wandbProject}`;
  const nearProdRun = Array.isArray(loopMcpSnapshot?.payload?.recent_runs)
    ? loopMcpSnapshot.payload.recent_runs[0] || null
    : null;

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
    `- latest_cycle: ${loopSummary?.cycle ?? "-"}`,
    `- latest_cycle_summary: ${loopSummary?.path ?? "-"}`,
    `- latest_cycle_weak_dims: ${(loopSummary?.payload?.weak_dims ?? []).join(", ") || "-"}`,
    `- latest_cycle_added_rows: ${fmt(loopSummary?.payload?.total_added_count)}`,
    "",
    "## Near-Prod Run Snapshot",
    "",
    `- mcp_snapshot_path: ${loopMcpSnapshot?.path ?? "-"}`,
    `- run_id: ${nearProdRun?.run_id ?? "-"}`,
    `- run_name: ${nearProdRun?.name ?? "-"}`,
    `- run_url: ${nearProdRun?.url ?? "-"}`,
    `- run_state: ${nearProdRun?.state ?? "-"}`,
    `- eval/mae_raw: ${fmt(nearProdRun?.summary?.["eval/mae_raw"])}`,
    `- eval/mse_norm: ${fmt(nearProdRun?.summary?.["eval/mse_norm"])}`,
    `- objective/train_loss: ${fmt(nearProdRun?.summary?.["objective/train_loss"])}`,
    `- learning_rate: ${fmt(nearProdRun?.config?.learning_rate)}`,
    `- epochs: ${fmt(nearProdRun?.config?.epochs)}`,
    "",
    "## MCP Failure Analysis",
    "",
    `- failure_source: ${failureCases?.source ?? "-"}`,
    `- failure_source_path: ${failureCases?.source_path ?? "-"}`,
    `- failure_mode: ${failureCases?.mode ?? "-"}`,
    `- top_error_dims: ${topDimRows.map((row) => `${row.dim}(${fmt(row.mean_abs_error)})`).join(", ") || "-"}`,
    `- top_failure_traces: ${topFailureRows.map((row) => row?.trace_url).filter(Boolean).join(" | ") || "-"}`,
    `- proposed_actions: ${recommendations.map((item) => item?.title).filter(Boolean).join(" | ") || "-"}`,
    "",
    "## Trace Hints",
    "",
    `- fine_tuned top traces: ${normalizedMcpTopTraceUrls.join(" | ") || normalizedFineTraceUrls.join(" | ") || "-"}`,
    `- prompt_baseline top traces: ${normalizedPromptTraceUrls.join(" | ") || "-"}`,
    "",
    "## Links",
    "",
    `- W&B Project URL: ${wandbProjectUrl}`,
    `- W&B Runs URL: ${wandbProjectUrl}/runs`,
    `- W&B Sweep URL: ${wandbProjectUrl}/sweeps`,
    `- W&B Report URL: ${wandbProjectUrl}/reports`,
    `- Weave Trace URL: https://wandb.ai/${wandbEntity}/${traceProject}/weave/traces`,
    `- HF Model URL: ${hfModelUrl}`,
    `- HF Dataset URL: ${hfDatasetUrl}`,
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
