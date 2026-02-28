import { mkdir, readFile, writeFile } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const summaryPath = resolve(root, "artifacts/eval/summary/latest_summary.json");
const samplesDir = resolve(root, "artifacts/eval/samples");
const loopRoot = resolve(root, "artifacts/loop");

const now = new Date().toISOString();
const cycle = process.env.LOOP_CYCLE_ID || "cycle_1";
const cycleDir = resolve(loopRoot, cycle);
const baseDatasetPath = process.env.LOOP_BASE_DATA_PATH || resolve(root, "data/ft/teacher_pairs.filtered.jsonl");
const mergedDatasetPath = process.env.LOOP_MERGED_OUTPUT_PATH || resolve(root, `data/ft/teacher_pairs.${cycle}.jsonl`);

const KEYS = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];

const asNumber = (value, fallback) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const hashString = (input) => {
  let hash = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
  }
  return Math.abs(hash >>> 0);
};

const lcg = (seed) => {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
};

const parseJsonl = (raw) =>
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));

const writeJsonl = async (path, rows) => {
  await writeFile(path, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
};

const toYaml = (obj) => {
  const lines = [];
  for (const [key, value] of Object.entries(obj)) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      lines.push(`${key}:`);
      for (const [nestedKey, nestedValue] of Object.entries(value)) {
        lines.push(`  ${nestedKey}: ${nestedValue}`);
      }
      continue;
    }
    lines.push(`${key}: ${value}`);
  }
  return `${lines.join("\n")}\n`;
};

const asBool = (value, fallback = false) => {
  if (value == null || value === "") return fallback;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
};

const parseForcedWeakDims = () => {
  const raw = String(process.env.LOOP_FORCE_WEAK_DIMS || "").trim();
  if (!raw) return [];
  const asSet = new Set(KEYS);
  return raw
    .split(",")
    .map((x) => x.trim())
    .filter((x) => x.length > 0 && asSet.has(x));
};

const loadSummary = async () => JSON.parse(await readFile(summaryPath, "utf8"));
const loadJsonlFile = async (path) => parseJsonl(await readFile(path, "utf8"));

const runMcpFetcher = async () => {
  const enabled = asBool(process.env.WANDB_MCP_ENABLED, true);
  if (!enabled) return false;

  const scriptPath = resolve(root, "scripts/wandb/fetch_mcp_eval_context.mjs");
  const run = spawnSync("node", [scriptPath], {
    cwd: root,
    env: { ...process.env, LOOP_CYCLE_ID: cycle },
    encoding: "utf8",
  });
  if (run.stdout) process.stdout.write(run.stdout);
  if (run.stderr) process.stderr.write(run.stderr);
  return run.status === 0;
};

const loadMcpDecisionInput = async () => {
  const path = resolve(cycleDir, "mcp_decision_input.json");
  try {
    const payload = JSON.parse(await readFile(path, "utf8"));
    return { payload, path };
  } catch {
    return { payload: null, path: null };
  }
};

const loadMcpSnapshot = async () => {
  const path = resolve(cycleDir, "mcp_eval_snapshot.json");
  try {
    const payload = JSON.parse(await readFile(path, "utf8"));
    return { payload, path };
  } catch {
    return { payload: null, path: null };
  }
};

const loadFineTunedSampleRows = async (summary) => {
  const runFile = summary.latest_by_mode?.fine_tuned?.file;
  if (!runFile) {
    return { rows: [], samplePath: null, runId: null };
  }

  const runId = runFile.replace(/\.json$/u, "");
  const samplePath = resolve(samplesDir, `${runId}.json`);

  try {
    const payload = JSON.parse(await readFile(samplePath, "utf8"));
    const failures = Array.isArray(payload.failures_top_k) ? payload.failures_top_k : [];
    return { rows: Array.isArray(payload.rows) ? payload.rows : [], failures, samplePath, runId };
  } catch {
    return { rows: [], failures: [], samplePath, runId };
  }
};

const computeWeakDims = (sampleRows) => {
  const totals = Object.fromEntries(KEYS.map((key) => [key, 0]));
  const counts = Object.fromEntries(KEYS.map((key) => [key, 0]));

  for (const row of sampleRows) {
    if (!row?.json_valid) continue;
    const byDim = row?.abs_error_by_dim ?? row?.abs_error_raw ?? null;
    if (!byDim) continue;
    for (const key of KEYS) {
      const value = Number(byDim?.[key]);
      if (Number.isFinite(value)) {
        totals[key] += value;
        counts[key] += 1;
      }
    }
  }

  const byDim = KEYS.map((key) => ({
    key,
    mae_raw: counts[key] > 0 ? totals[key] / counts[key] : 0,
    count: counts[key],
  })).sort((a, b) => b.mae_raw - a.mae_raw);

  return {
    by_dim: byDim,
    focus_dims: byDim.filter((x) => x.count > 0).slice(0, 2).map((x) => x.key),
  };
};

const deriveNextHparams = (summary, weakDims, recentRuns = []) => {
  const prompt = summary.latest_by_mode?.prompt_baseline?.metrics ?? {};
  const ft = summary.latest_by_mode?.fine_tuned?.metrics ?? {};
  const recent = recentRuns.find((row) => row?.config) ?? null;

  const current = {
    learning_rate: asNumber(Number(recent?.config?.learning_rate ?? process.env.HF_FT_LR), 2e-5),
    lora_r: asNumber(Number(recent?.config?.lora_r ?? process.env.HF_FT_LORA_R), 16),
    lora_alpha: asNumber(Number(recent?.config?.lora_alpha ?? process.env.HF_FT_LORA_ALPHA), 32),
    lora_dropout: asNumber(Number(recent?.config?.lora_dropout ?? process.env.HF_FT_LORA_DROPOUT), 0.05),
    epochs: asNumber(Number(recent?.config?.epochs ?? process.env.HF_FT_EPOCHS), 2),
  };

  const next = { ...current };
  const reasons = [];

  const jsonValid = asNumber(ft.json_valid_rate, 0);
  if (jsonValid < 0.98) {
    next.learning_rate = Math.max(1e-6, current.learning_rate * 0.6);
    next.epochs = Math.min(4, current.epochs + 1);
    reasons.push("json_valid_rate below target -> lower LR and increase epochs");
  }

  const mseNorm = asNumber(ft.mse_norm, 0);
  const promptMseNorm = asNumber(prompt.mse_norm, 0);
  if (mseNorm >= promptMseNorm && promptMseNorm > 0) {
    next.lora_r = current.lora_r >= 32 ? 32 : current.lora_r === 16 ? 32 : 16;
    next.lora_alpha = next.lora_r * 2;
    reasons.push("mse_norm not improved vs prompt baseline -> increase LoRA rank");
  }

  if (weakDims.length > 0) {
    next.epochs = Math.min(4, Math.max(next.epochs, 2));
    reasons.push(`weak dims detected (${weakDims.join(", ")}) -> keep stronger adaptation capacity`);
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

const deriveAugmentationSpec = (summary, weak, runId, samplePath, snapshotSource) => {
  const delta = summary.auto_improvement_delta ?? {};
  const weakDims = weak.focus_dims.length > 0 ? weak.focus_dims : ["nostalgia", "brightness"];

  const addRatio = clamp(Number(process.env.LOOP_ADD_RATIO || 0.2), 0.05, 0.5);
  const replayRatio = clamp(Number(process.env.LOOP_HARD_CASE_REPLAY_RATIO || 0.15), 0, 0.5);

  const defaultBuckets = {
    nostalgia: [70, 95],
    brightness: [0, 30],
    energy: [70, 95],
    complexity: [0, 30],
    warmth: [65, 95],
    acousticness: [65, 95],
  };

  return {
    dataset_version_from: process.env.FT_DATASET_VERSION_FROM || "v1",
    dataset_version_to: process.env.FT_DATASET_VERSION_TO || `${cycle}`,
    source_run_id: runId,
    source_sample_path: samplePath,
    source_snapshot: snapshotSource,
    focus_dims: weakDims,
    weak_dim_stats: weak.by_dim,
    focus_buckets: Object.fromEntries(weakDims.map((dim) => [dim, defaultBuckets[dim] ?? [20, 80]])),
    add_ratio: addRatio,
    hard_case_replay_ratio: replayRatio,
    teacher_model: process.env.MISTRAL_TEACHER_MODEL || "mistral-large-latest",
    filters: {
      deny_numeric_leak: true,
      deduplicate: true,
      language_allow: ["en"],
    },
    rationale: {
      auto_improvement_delta: delta,
      note: "focus_dims are selected from highest per-dimension MAE in latest fine_tuned eval samples",
    },
  };
};

const makeTags = (vector) => {
  const tags = [];
  if (vector.nostalgia > 65) tags.push("nostalgic");
  if (vector.acousticness > 60) tags.push("acoustic");
  if (vector.warmth > 60) tags.push("cozy");
  if (vector.energy > 60) tags.push("upbeat");
  if (vector.brightness < 35) tags.push("rain");
  if (!tags.length) tags.push("balanced");
  return [...new Set(tags)];
};

const makeConstraints = (vector) => ({
  preferredStyleTags: vector.nostalgia > 60 ? ["citypop_80s"] : vector.energy > 60 ? ["pop_2000s"] : ["hiphop_90s"],
  preferredGimmickTags: vector.energy > 58 ? ["filter_rise"] : ["beat_mute"],
  avoidPartIds: vector.brightness < 20 ? ["style_2000s_pop"] : [],
});

const describeDim = (dim, value) => {
  const v = Number(value);
  if (dim === "nostalgia") return v >= 50 ? "with stronger old-memory atmosphere" : "with less retro feeling";
  if (dim === "brightness") return v >= 50 ? "with brighter air and sparkle" : "with dimmer, softer brightness";
  if (dim === "energy") return v >= 50 ? "with more forward drive" : "with calmer pacing";
  if (dim === "complexity") return v >= 50 ? "with richer layered detail" : "with simpler structure";
  if (dim === "warmth") return v >= 50 ? "with warmer tone color" : "with cooler tone";
  if (dim === "acousticness") return v >= 50 ? "with more acoustic texture" : "with less acoustic emphasis";
  return "";
};

const mutateVector = (vector, focusDims, rng) => {
  const out = {};
  for (const key of KEYS) {
    const base = asNumber(Number(vector?.[key]), 50);
    const jitter = Math.round((rng() * 2 - 1) * 6);
    out[key] = clamp(base + jitter, 0, 100);
  }
  for (const dim of focusDims) {
    const direction = out[dim] >= 50 ? 1 : -1;
    const magnitude = 12 + Math.round(rng() * 16);
    out[dim] = clamp(out[dim] + direction * magnitude, 0, 100);
  }
  return out;
};

const VARIANT_WORDS = [
  "breeze",
  "harbor",
  "lantern",
  "velvet",
  "echo",
  "dawn",
  "twilight",
  "petal",
  "meadow",
  "drizzle",
  "cobblestone",
  "amber",
  "ripple",
  "willow",
  "sunbeam",
  "sparrow",
];

const generateAugmentedRows = (baseRows, sampleRows, spec, runId) => {
  const rng = lcg(hashString(`${cycle}:${runId ?? "none"}`));
  const desired = Math.max(12, Math.round(baseRows.length * spec.add_ratio));
  const replayCount = Math.min(desired, Math.round(desired * spec.hard_case_replay_ratio));
  const seenTexts = new Set(baseRows.map((row) => String(row.request_text || "").trim().toLowerCase()));

  const validSamples = sampleRows
    .filter((row) => row?.json_valid && row?.target_vector && typeof row.request_text === "string")
    .sort((a, b) => asNumber(b.mae_raw, 0) - asNumber(a.mae_raw, 0));

  const fallbackSamples =
    validSamples.length > 0
      ? validSamples
      : baseRows.map((row, idx) => ({
          id: row.id || `base_${idx}`,
          request_text: row.request_text,
          target_vector: row.target_hidden_params?.vector ?? {},
          mae_raw: 0,
        }));

  if (fallbackSamples.length === 0) {
    return { generatedRows: [], replayRows: [] };
  }

  const focusDims = spec.focus_dims.length > 0 ? spec.focus_dims : ["nostalgia", "brightness"];
  const generatedRows = [];
  let serial = 0;
  let attempts = 0;
  const maxAttempts = Math.max(500, desired * 60);

  while (generatedRows.length < desired && attempts < maxAttempts) {
    attempts += 1;
    const src = fallbackSamples[serial % fallbackSamples.length];
    serial += 1;

    const vector = mutateVector(src.target_vector ?? {}, focusDims, rng);
    const phrases = focusDims.map((dim) => describeDim(dim, vector[dim])).filter(Boolean);
    const variant = VARIANT_WORDS[Math.floor(rng() * VARIANT_WORDS.length) % VARIANT_WORDS.length];
    const requestText =
      `Please compose a variation of this request: ${src.request_text}. ` +
      `${phrases.join(", ")}, keeping a ${variant} mood.`.replace(/\s+/gu, " ").trim();

    const dedupeKey = requestText.toLowerCase();
    if (seenTexts.has(dedupeKey)) continue;
    seenTexts.add(dedupeKey);

    generatedRows.push({
      id: `${cycle}_aug_${String(generatedRows.length + 1).padStart(5, "0")}`,
      request_text: requestText,
      target_hidden_params: {
        vector,
        tags: makeTags(vector),
        constraints: makeConstraints(vector),
      },
      meta: {
        generation_source: "dynamic_hard_case_augmentation",
        loop_cycle: cycle,
        source_eval_run_id: runId,
        source_case_id: src.id ?? null,
        focus_dims: focusDims,
        weak_case_mae_raw: asNumber(src.mae_raw, 0),
        generated_at: now,
      },
    });
  }

  const replayRows = [];
  for (let i = 0; i < replayCount && i < fallbackSamples.length; i += 1) {
    const src = fallbackSamples[i];
    const variant = VARIANT_WORDS[(i + 3) % VARIANT_WORDS.length];
    const replayText = `Please reinterpret this request with similar intent and ${variant} mood: ${src.request_text}`;
    const dedupeKey = replayText.toLowerCase();
    if (seenTexts.has(dedupeKey)) continue;
    seenTexts.add(dedupeKey);

    const vector = {};
    for (const key of KEYS) {
      vector[key] = clamp(Math.round(asNumber(Number(src.target_vector?.[key]), 50)), 0, 100);
    }

    replayRows.push({
      id: `${cycle}_replay_${String(replayRows.length + 1).padStart(5, "0")}`,
      request_text: replayText,
      target_hidden_params: {
        vector,
        tags: makeTags(vector),
        constraints: makeConstraints(vector),
      },
      meta: {
        generation_source: "hard_case_replay",
        loop_cycle: cycle,
        source_eval_run_id: runId,
        source_case_id: src.id ?? null,
        focus_dims: spec.focus_dims,
        weak_case_mae_raw: asNumber(src.mae_raw, 0),
        generated_at: now,
      },
    });
  }

  return { generatedRows, replayRows };
};

const buildHparamPatch = (current, next) => {
  const patch = {};
  for (const key of Object.keys(next)) {
    if (next[key] !== current[key]) {
      patch[key] = {
        from: current[key],
        to: next[key],
      };
    }
  }
  return patch;
};

const writeBeforeAfterCsv = async (summary) => {
  const prompt = summary.latest_by_mode?.prompt_baseline?.metrics ?? {};
  const ft = summary.latest_by_mode?.fine_tuned?.metrics ?? {};
  const delta = summary.auto_improvement_delta ?? {};
  const rows = [
    "metric,prompt_baseline,fine_tuned,delta",
    `json_valid_rate,${asNumber(prompt.json_valid_rate, 0)},${asNumber(ft.json_valid_rate, 0)},${asNumber(delta.json_valid_rate, 0)}`,
    `vector_mae,${asNumber(prompt.vector_mae, 0)},${asNumber(ft.vector_mae, 0)},${asNumber(delta.vector_mae, 0)}`,
    `intent_score_mean,${asNumber(prompt.intent_score_mean, 0)},${asNumber(ft.intent_score_mean, 0)},${asNumber(delta.intent_score_mean, 0)}`,
    `mse_norm,${asNumber(prompt.mse_norm, 0)},${asNumber(ft.mse_norm, 0)},${asNumber(ft.mse_norm, 0) - asNumber(prompt.mse_norm, 0)}`,
  ];
  await writeFile(resolve(cycleDir, "before_after_metrics.csv"), `${rows.join("\n")}\n`);
};

const main = async () => {
  await mkdir(cycleDir, { recursive: true });
  const mcpEnabled = asBool(process.env.WANDB_MCP_ENABLED, true);
  const mcpFetchOk = await runMcpFetcher();

  const mcpSnapshot = await loadMcpSnapshot();
  const mcpDecision = await loadMcpDecisionInput();
  const hasMcpArtifacts = Boolean(mcpSnapshot.payload && mcpDecision.payload);
  const allowStaleSnapshot = asBool(process.env.WANDB_MCP_ALLOW_STALE_SNAPSHOT, false);

  if (mcpEnabled && !mcpFetchOk && !(allowStaleSnapshot && hasMcpArtifacts)) {
    throw new Error(
      "WANDB_MCP_ENABLED=true but MCP fetch failed. Check WANDB_MCP_BASE_URL / DNS / WANDB_API_KEY. " +
        "If you intentionally want to reuse existing artifacts, set WANDB_MCP_ALLOW_STALE_SNAPSHOT=true.",
    );
  }

  if (mcpEnabled && !hasMcpArtifacts) {
    throw new Error(
      "WANDB_MCP_ENABLED=true but MCP artifacts are missing. Expected mcp_eval_snapshot.json and mcp_decision_input.json.",
    );
  }

  if (mcpEnabled && !mcpFetchOk && allowStaleSnapshot && hasMcpArtifacts) {
    console.warn("MCP fetch failed; continuing with existing MCP snapshot artifacts (WANDB_MCP_ALLOW_STALE_SNAPSHOT=true).");
  }

  const summary = mcpDecision.payload?.eval_summary?.latest_by_mode
    ? {
        ...((mcpSnapshot.payload?.eval_summary ?? {})),
        latest_by_mode: mcpDecision.payload.eval_summary.latest_by_mode,
        auto_improvement_delta: mcpDecision.payload.eval_summary.auto_improvement_delta,
        loop_completion_rate: mcpDecision.payload.eval_summary.loop_completion_rate,
      }
    : await loadSummary();

  const samplePack = await loadFineTunedSampleRows(summary);
  const mcpFailures = Array.isArray(mcpDecision.payload?.failures_top_k) ? mcpDecision.payload.failures_top_k : [];
  const weakSourceRows = mcpFailures.length > 0 ? mcpFailures : samplePack.rows;
  const computedWeak = computeWeakDims(weakSourceRows);
  const forcedWeakDims = parseForcedWeakDims();
  const weak = forcedWeakDims.length > 0 ? { ...computedWeak, focus_dims: forcedWeakDims } : computedWeak;
  const weakDimsSource = forcedWeakDims.length > 0 ? "forced_from_training_validation" : mcpFailures.length > 0 ? "mcp_failures_top_k" : "eval_sample_rows";

  const recentRuns = Array.isArray(mcpDecision.payload?.recent_runs) ? mcpDecision.payload.recent_runs : [];
  const hparams = deriveNextHparams(summary, weak.focus_dims, recentRuns);
  const augmentation = deriveAugmentationSpec(
    summary,
    weak,
    samplePack.runId,
    samplePack.samplePath,
    mcpSnapshot.payload?.source ?? "local",
  );

  const baseRows = await loadJsonlFile(baseDatasetPath);
  const sourceForAug = mcpFailures.length > 0 ? mcpFailures : samplePack.rows;
  const { generatedRows, replayRows } = generateAugmentedRows(baseRows, sourceForAug, augmentation, samplePack.runId);
  const combinedRows = [...generatedRows, ...replayRows];
  const mergedRows = [...baseRows, ...combinedRows];

  const hparamPatch = buildHparamPatch(hparams.current, hparams.next);

  await writeFile(resolve(cycleDir, "next_hparams.yaml"), toYaml(hparams.next));
  await writeFile(resolve(cycleDir, "hparam_patch.yaml"), toYaml(hparamPatch));
  await writeFile(resolve(cycleDir, "augmentation_spec.json"), JSON.stringify(augmentation, null, 2));
  await writeJsonl(resolve(cycleDir, "generated_augmented_pairs.jsonl"), combinedRows);
  await writeJsonl(mergedDatasetPath, mergedRows);
  await writeBeforeAfterCsv(summary);

  const decisionLog = [
    `# Loop Decision Log: ${cycle}`,
    "",
    `Generated at: ${now}`,
    "",
    "## Source",
    `- mcp_snapshot_path: ${mcpSnapshot.path ?? "none"}`,
    `- mcp_decision_input_path: ${mcpDecision.path ?? "none"}`,
    `- mcp_source: ${mcpSnapshot.payload?.source ?? "none"}`,
    `- fine_tuned_eval_run_id: ${samplePack.runId ?? "none"}`,
    `- fine_tuned_sample_path: ${samplePack.samplePath ?? "none"}`,
    `- base_dataset_path: ${baseDatasetPath}`,
    "",
    "## Weak Dimensions (from sample errors)",
    `- source: ${weakDimsSource}`,
    forcedWeakDims.length > 0 ? `- forced_dims: ${forcedWeakDims.join(", ")}` : "- forced_dims: none",
    "```json",
    JSON.stringify(weak.by_dim, null, 2),
    "```",
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
    "## HParam Patch",
    "```json",
    JSON.stringify(hparamPatch, null, 2),
    "```",
    "",
    "## Reasons",
    ...hparams.reasons.map((reason) => `- ${reason}`),
    "",
    "## Augmentation Spec",
    "```json",
    JSON.stringify(augmentation, null, 2),
    "```",
    "",
    "## Generated Rows",
    `- generated_augmented: ${generatedRows.length}`,
    `- hard_case_replay: ${replayRows.length}`,
    `- total_added: ${combinedRows.length}`,
    `- merged_total: ${mergedRows.length}`,
  ].join("\n");

  await writeFile(resolve(cycleDir, "decision_log.md"), decisionLog);

  const summaryPayload = {
    cycle,
    completed: true,
    generated_at: now,
    loop_completion_rate_hint: 1.0,
    auto_improvement_delta: summary.auto_improvement_delta ?? null,
    source_eval_run_id: samplePack.runId,
    weak_dims: augmentation.focus_dims,
    weak_dims_source: weakDimsSource,
    generated_augmented_count: generatedRows.length,
    replay_count: replayRows.length,
    total_added_count: combinedRows.length,
    base_dataset_count: baseRows.length,
    merged_dataset_count: mergedRows.length,
    generated_pairs_path: resolve(cycleDir, "generated_augmented_pairs.jsonl"),
    merged_dataset_path: mergedDatasetPath,
    mcp_snapshot_path: mcpSnapshot.path,
    mcp_decision_input_path: mcpDecision.path,
  };

  await writeFile(resolve(cycleDir, "summary.json"), JSON.stringify(summaryPayload, null, 2));

  console.log(`Generated loop artifacts in ${cycleDir}`);
  console.log(
    JSON.stringify(
      {
        cycle,
        source_eval_run_id: samplePack.runId,
        weak_dims: augmentation.focus_dims,
        generated_augmented_count: generatedRows.length,
        replay_count: replayRows.length,
        merged_dataset_path: mergedDatasetPath,
      },
      null,
      2,
    ),
  );
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
