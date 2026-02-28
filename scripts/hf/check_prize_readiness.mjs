import { access, readFile } from "node:fs/promises";
import { constants } from "node:fs";
import { execSync } from "node:child_process";
import { resolve } from "node:path";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const summaryPath = resolve(root, "artifacts/eval/summary/latest_summary.json");
const reportDraftPath = resolve(root, "artifacts/wandb/report_draft.md");

const REQUIRED_ENV = [
  "HF_TOKEN",
  "WANDB_API_KEY",
  "HF_NAMESPACE",
  "HF_FT_DATASET_REPO_ID",
  "HF_FT_OUTPUT_MODEL_ID",
  "WANDB_PROJECT",
  "WEAVE_PROJECT"
];

const OPTIONAL_ENV = [
  "WANDB_ENTITY",
  "WANDB_RUN_GROUP",
  "WANDB_MCP_ENABLED",
  "WANDB_MCP_BASE_URL",
  "WANDB_MCP_USE_FALLBACK_API"
];

const MODES = ["rule_baseline", "prompt_baseline", "fine_tuned"];

const REQUIRED_METRICS = [
  "json_valid_rate",
  "vector_mae",
  "mse_raw",
  "mse_norm",
  "constraint_match_rate",
  "slot_exact_match",
  "intent_score_mean",
  "output_sanity_score",
  "p95_inference_latency_ms",
  "cost_per_100_requests_usd",
];

const REQUIRED_AUTO_DELTA = ["intent_score_mean", "vector_mae", "json_valid_rate"];

const REQUIRED_LINK_LABELS = [
  "W&B Runs URL:",
  "W&B Sweep URL:",
  "W&B Report URL:",
  "Weave Trace URL:",
  "HF Model URL:",
  "HF Dataset URL:",
];

const checkFileExists = async (path) => {
  try {
    await access(path, constants.F_OK);
    return true;
  } catch {
    return false;
  }
};

const checkHfAuth = () => {
  try {
    const out = execSync("hf auth whoami", { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    return { ok: !out.includes("Not logged in"), raw: out.trim() };
  } catch (error) {
    const stderr = error?.stderr?.toString?.() ?? String(error.message ?? error);
    return { ok: false, raw: stderr.trim() };
  }
};

const main = async () => {
  const missing = [];
  const warnings = [];

  const envStatus = Object.fromEntries(
    REQUIRED_ENV.map((key) => [key, Boolean(process.env[key] && process.env[key].trim() !== "")]),
  );
  const optionalEnvStatus = Object.fromEntries(
    OPTIONAL_ENV.map((key) => [key, Boolean(process.env[key] && process.env[key].trim() !== "")]),
  );

  for (const [key, isSet] of Object.entries(envStatus)) {
    if (!isSet) {
      missing.push(`env:${key}`);
    }
  }

  const hfAuth = checkHfAuth();
  if (!hfAuth.ok) {
    missing.push("auth:hf_cli");
  }

  const hasSummary = await checkFileExists(summaryPath);
  const hasReportDraft = await checkFileExists(reportDraftPath);

  if (!hasSummary) {
    missing.push("artifact:artifacts/eval/summary/latest_summary.json");
  }

  if (!hasReportDraft) {
    missing.push("artifact:artifacts/wandb/report_draft.md");
  }

  let summary = null;
  if (hasSummary) {
    summary = JSON.parse(await readFile(summaryPath, "utf8"));
  }

  const metricCoverage = {};
  if (summary) {
    const latestByMode = summary.latest_by_mode ?? {};
    for (const mode of MODES) {
      const modeMetrics = latestByMode?.[mode]?.metrics ?? {};
      metricCoverage[mode] = {};
      for (const key of REQUIRED_METRICS) {
        const ok = typeof modeMetrics[key] === "number" && Number.isFinite(modeMetrics[key]);
        metricCoverage[mode][key] = ok;
        if (!ok) {
          missing.push(`metric:${mode}.${key}`);
        }
      }
    }

    if (!(typeof summary.loop_completion_rate === "number" && Number.isFinite(summary.loop_completion_rate))) {
      missing.push("metric:loop_completion_rate");
    }

    for (const key of REQUIRED_AUTO_DELTA) {
      const value = summary.auto_improvement_delta?.[key];
      if (!(typeof value === "number" && Number.isFinite(value))) {
        missing.push(`metric:auto_improvement_delta.${key}`);
      }
    }

    const jsonValid = summary.latest_by_mode?.fine_tuned?.metrics?.json_valid_rate;
    if (typeof jsonValid === "number" && jsonValid < 0.98) {
      warnings.push(`fine_tuned json_valid_rate ${jsonValid.toFixed(4)} is below target 0.98`);
    }
  }

  const reportLinks = {};
  if (hasReportDraft) {
    const reportText = await readFile(reportDraftPath, "utf8");
    for (const label of REQUIRED_LINK_LABELS) {
      const found = reportText.includes(label);
      reportLinks[label] = found;
      if (!found) {
        missing.push(`report:${label}`);
      }
    }
  }

  const result = {
    checked_at: new Date().toISOString(),
    ready_for_real_job_and_report: missing.length === 0,
    hf_cli_auth: hfAuth,
    env_status: envStatus,
    optional_env_status: optionalEnvStatus,
    metric_coverage: metricCoverage,
    report_link_placeholders: reportLinks,
    warnings,
    missing,
  };

  console.log(JSON.stringify(result, null, 2));

  if (missing.length > 0) {
    process.exit(1);
  }
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
