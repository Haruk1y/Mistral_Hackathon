import { mkdir, readFile, writeFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const summaryPath = resolve(root, "artifacts/eval/summary/latest_summary.json");
const samplesDir = resolve(root, "artifacts/eval/samples");
const loopDir = resolve(root, "artifacts/loop");
const outDir = resolve(root, "artifacts/wandb");

const mode = process.env.FAILURE_MODE || "fine_tuned";
const topK = Math.max(1, Number.parseInt(process.env.FAILURE_TOP_K || "12", 10));
const cycleId = process.env.LOOP_CYCLE_ID || "";

const asNum = (value) => (typeof value === "number" && Number.isFinite(value) ? value : null);
const fmt = (value, digits = 4) => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(digits);
};

const mean = (values) => {
  if (!values.length) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
};

const failureScore = (row) => {
  const mae = asNum(row?.mae_raw) ?? 0;
  const slotExact = asNum(row?.slot_exact_match);
  const sanity = asNum(row?.output_sanity_score);
  const intent = asNum(row?.intent_score);

  let score = mae;
  if (row?.json_valid === false || row?.parse_error) score += 1000;
  if (row?.constraint_match === false) score += 50;
  if (slotExact !== null) score += (1 - Math.max(0, Math.min(1, slotExact))) * 30;
  if (sanity !== null) score += Math.max(0, 90 - sanity);
  if (intent !== null) score += Math.max(0, 60 - intent) * 0.5;
  return score;
};

const normalizeAbsErrorByDim = (value) => {
  if (!value || typeof value !== "object") return {};
  const out = {};
  for (const [key, raw] of Object.entries(value)) {
    const num = Number(raw);
    if (Number.isFinite(num)) out[key] = num;
  }
  return out;
};

const topDims = (row, k = 2) => {
  const byDim = normalizeAbsErrorByDim(row?.abs_error_by_dim);
  const entries = Object.entries(byDim)
    .sort((a, b) => b[1] - a[1])
    .slice(0, k);
  return entries.map(([dim, error]) => ({ dim, abs_error: error }));
};

const aggregateDims = (rows) => {
  const sums = new Map();
  const counts = new Map();
  for (const row of rows) {
    const byDim = normalizeAbsErrorByDim(row?.abs_error_by_dim);
    for (const [dim, value] of Object.entries(byDim)) {
      sums.set(dim, (sums.get(dim) || 0) + value);
      counts.set(dim, (counts.get(dim) || 0) + 1);
    }
  }
  return [...sums.entries()]
    .map(([dim, sum]) => ({ dim, mean_abs_error: sum / (counts.get(dim) || 1), count: counts.get(dim) || 0 }))
    .sort((a, b) => b.mean_abs_error - a.mean_abs_error);
};

const buildRecommendations = (dimRank) => {
  const focus = dimRank.slice(0, 2).map((x) => x.dim);
  const focusLabel = focus.length > 0 ? focus.join(", ") : "top_error_dims";
  return [
    {
      title: "Focused Hard-Case Augmentation",
      why: `Highest absolute errors are concentrated on: ${focusLabel}.`,
      action:
        "Generate paired requests that stress those dimensions with high and low target buckets, then replay top hard cases from latest validation.",
      config_hint: {
        focus_dims: focus,
        add_ratio: 0.2,
        hard_case_replay_ratio: 0.2,
      },
    },
    {
      title: "Dimension-Weighted Regression Loss",
      why: "Current MAE is skewed by a few dimensions.",
      action:
        "Apply per-dimension loss weights proportional to recent hard-case MAE (cap at 2.0x), then compare with unweighted baseline in A/B runs.",
      config_hint: {
        weight_source: "artifacts/wandb/weave_failure_cases.json:aggregate_dim_abs_error",
      },
    },
    {
      title: "MCP-Backed Promotion Gate",
      why: "Need reproducible pass/fail criteria before promoting a retrain.",
      action:
        "Block promotion unless focus dimensions improve and include at least 3 MCP trace-linked failures with remediation notes in the W&B report.",
      config_hint: {
        gate: {
          max_mae_raw: 22.5,
          max_focus_dim_mae_raw: 25.0,
          require_mcp_trace_examples: 3,
        },
      },
    },
  ];
};

const latestCycleDir = async () => {
  if (cycleId && /^cycle_\d+$/u.test(cycleId)) {
    return resolve(loopDir, cycleId);
  }
  try {
    const entries = await readdir(loopDir, { withFileTypes: true });
    const cycles = entries
      .filter((entry) => entry.isDirectory() && /^cycle_\d+$/u.test(entry.name))
      .map((entry) => entry.name)
      .sort((a, b) => Number(a.replace("cycle_", "")) - Number(b.replace("cycle_", "")));
    if (cycles.length === 0) return null;
    return resolve(loopDir, cycles[cycles.length - 1]);
  } catch {
    return null;
  }
};

const loadMcpFailureRows = async () => {
  const cyclePath = await latestCycleDir();
  if (!cyclePath) return null;

  const snapshotPath = resolve(cyclePath, "mcp_eval_snapshot.json");
  try {
    const raw = JSON.parse(await readFile(snapshotPath, "utf8"));
    const failures = Array.isArray(raw?.failures_top_k) ? raw.failures_top_k : [];
    if (failures.length === 0) return null;
    return {
      source: "mcp_snapshot",
      source_path: snapshotPath,
      mode,
      rows: failures
        .filter((row) => !mode || row?.mode === mode)
        .map((row) => ({
          ...row,
          abs_error_by_dim: normalizeAbsErrorByDim(row?.abs_error_by_dim),
          mae_raw:
            asNum(row?.mae_raw) ??
            (Object.keys(normalizeAbsErrorByDim(row?.abs_error_by_dim)).length > 0
              ? mean(Object.values(normalizeAbsErrorByDim(row?.abs_error_by_dim)))
              : null),
        })),
      metadata: {
        snapshot_source: raw?.source ?? "unknown",
        trace_project_selected: raw?.trace_project_selected ?? null,
        failures_len: raw?.failures_len ?? failures.length,
        selected_mode: raw?.failures_selected_mode ?? null,
      },
    };
  } catch {
    return null;
  }
};

const loadEvalSampleRows = async () => {
  const summary = JSON.parse(await readFile(summaryPath, "utf8"));
  const sampleFile = summary?.latest_samples_by_mode?.[mode]?.sample_file;
  if (!sampleFile) {
    throw new Error(`latest sample file not found for mode=${mode}`);
  }
  const samplePath = resolve(samplesDir, sampleFile);
  const sample = JSON.parse(await readFile(samplePath, "utf8"));
  const rows = Array.isArray(sample?.rows) ? sample.rows : [];
  return {
    source: "eval_samples",
    source_path: samplePath,
    mode,
    rows: rows.map((row) => ({
      ...row,
      abs_error_by_dim: normalizeAbsErrorByDim(row?.abs_error_by_dim),
    })),
    metadata: {
      sample_file: sampleFile,
    },
  };
};

const main = async () => {
  const mcpPayload = await loadMcpFailureRows();
  const sourcePayload = mcpPayload ?? (await loadEvalSampleRows());

  const scoredRows = sourcePayload.rows.map((row) => ({
    row,
    failure_score: failureScore(row),
  }));
  const withDimErrors = scoredRows.filter(
    ({ row }) => row?.json_valid === true && Object.keys(normalizeAbsErrorByDim(row?.abs_error_by_dim)).length > 0,
  );
  const withoutDimErrors = scoredRows.filter(
    ({ row }) => !(row?.json_valid === true && Object.keys(normalizeAbsErrorByDim(row?.abs_error_by_dim)).length > 0),
  );
  const dimQuota = Math.min(topK, Math.max(5, Math.ceil(topK * 0.7)));
  const selected = [
    ...withDimErrors.sort((a, b) => b.failure_score - a.failure_score).slice(0, dimQuota),
    ...withoutDimErrors
      .sort((a, b) => b.failure_score - a.failure_score)
      .slice(0, Math.max(0, topK - Math.min(withDimErrors.length, dimQuota))),
  ].slice(0, topK);

  const ranked = selected.map(({ row, failure_score }) => ({
      id: row?.id || row?.trace_id || null,
      scenario: row?.scenario || null,
      request_text: row?.request_text || "",
      failure_score,
      mae_raw: asNum(row?.mae_raw),
      json_valid: row?.json_valid ?? null,
      parse_error: row?.parse_error ?? null,
      constraint_match: row?.constraint_match ?? null,
      slot_exact_match: asNum(row?.slot_exact_match),
      intent_score: asNum(row?.intent_score),
      output_sanity_score: asNum(row?.output_sanity_score),
      trace_id: row?.trace_id || null,
      trace_url: row?.trace_url || null,
      top_error_dims: topDims(row, 2),
      abs_error_by_dim: normalizeAbsErrorByDim(row?.abs_error_by_dim),
      mode: row?.mode || null,
    }));

  const dimRank = aggregateDims(ranked);
  const recommendations = buildRecommendations(dimRank);

  const payload = {
    generated_at: new Date().toISOString(),
    mode,
    top_k: topK,
    source: sourcePayload.source,
    source_path: sourcePayload.source_path,
    source_metadata: sourcePayload.metadata,
    aggregate_dim_abs_error: dimRank,
    failure_cases: ranked,
    recommendations,
  };

  const md = [
    "# Weave Failure Playbook",
    "",
    `Generated at: ${payload.generated_at}`,
    `Mode: ${mode}`,
    `Source: ${payload.source}`,
    `Source path: ${payload.source_path}`,
    `Top K: ${topK}`,
    "",
    "## Source Metadata",
    "",
    "```json",
    JSON.stringify(payload.source_metadata, null, 2),
    "```",
    "",
    "## Aggregate Error Dimensions (from failure set)",
    "",
    "| Dim | Mean Abs Error | Count |",
    "|---|---:|---:|",
    ...dimRank.map((row) => `| ${row.dim} | ${fmt(row.mean_abs_error, 3)} | ${row.count} |`),
    "",
    "## Failure Cases (Trace-linked)",
    "",
    "| ID | Scenario | mae_raw | json_valid | top_error_dims | trace |",
    "|---|---|---:|---|---|---|",
    ...ranked.map((row) => {
      const dims = row.top_error_dims.map((x) => `${x.dim}:${fmt(x.abs_error, 1)}`).join(", ");
      const trace = row.trace_url ? `[link](${row.trace_url})` : "-";
      return `| ${row.id || "-"} | ${row.scenario || "-"} | ${fmt(row.mae_raw, 3)} | ${String(row.json_valid)} | ${dims || "-"} | ${trace} |`;
    }),
    "",
    "## Improvement Plan",
    "",
    ...recommendations.flatMap((item, idx) => [
      `${idx + 1}. ${item.title}`,
      `- Why: ${item.why}`,
      `- Action: ${item.action}`,
      `- Config hint: \`${JSON.stringify(item.config_hint)}\``,
    ]),
    "",
  ];

  await mkdir(outDir, { recursive: true });
  const jsonOut = resolve(outDir, "weave_failure_cases.json");
  const mdOut = resolve(outDir, "weave_failure_playbook.md");
  await writeFile(jsonOut, `${JSON.stringify(payload, null, 2)}\n`);
  await writeFile(mdOut, `${md.join("\n")}\n`);

  console.log(`Wrote ${jsonOut}`);
  console.log(`Wrote ${mdOut}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
