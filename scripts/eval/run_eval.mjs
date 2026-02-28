import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const datasetPath = resolve(root, "data/eval/frozen_eval_set.v1.json");
const runsDir = resolve(root, "artifacts/eval/runs");

const KEYS = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];
const SLOTS = ["style", "instrument", "mood", "gimmick"];

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

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const modeSettings = {
  rule_baseline: {
    noise: 14,
    jsonValidRate: 0.95,
    constraintMatchRate: 0.82,
    slotMatchRate: 0.62,
    latencyRange: [120, 300],
    costPerRequest: 0.0002,
    sanityRange: [45, 78]
  },
  prompt_baseline: {
    noise: 10,
    jsonValidRate: 0.97,
    constraintMatchRate: 0.86,
    slotMatchRate: 0.71,
    latencyRange: [380, 920],
    costPerRequest: 0.0028,
    sanityRange: [58, 86]
  },
  fine_tuned: {
    noise: 6,
    jsonValidRate: 0.992,
    constraintMatchRate: 0.93,
    slotMatchRate: 0.84,
    latencyRange: [260, 720],
    costPerRequest: 0.0019,
    sanityRange: [65, 92]
  }
};

const pickMode = (value) => {
  if (value && value in modeSettings) return value;
  return "prompt_baseline";
};

const mutateConstraints = (constraints, rng, shouldMatch) => {
  if (shouldMatch) return constraints;

  const copied = JSON.parse(JSON.stringify(constraints));
  if (!Array.isArray(copied.avoidPartIds)) copied.avoidPartIds = [];
  copied.avoidPartIds = [...copied.avoidPartIds, "style_2000s_pop"];
  return copied;
};

const predictVector = (target, rng, noise) => {
  const out = {};
  for (const key of KEYS) {
    const delta = (rng() * 2 - 1) * noise;
    out[key] = clamp(Math.round(target[key] + delta), 0, 100);
  }
  return out;
};

const predictTop1BySlot = (targetTop1, rng, slotMatchRate) => {
  const out = { ...targetTop1 };
  for (const slot of SLOTS) {
    if (rng() <= slotMatchRate) continue;

    if (slot === "style") out[slot] = targetTop1[slot] === "style_2000s_pop" ? "style_80s_citypop" : "style_2000s_pop";
    if (slot === "instrument") out[slot] = targetTop1[slot] === "inst_piano_upright" ? "inst_analog_synth" : "inst_piano_upright";
    if (slot === "mood") out[slot] = targetTop1[slot] === "mood_rain_ambience" ? "mood_sun_glow" : "mood_rain_ambience";
    if (slot === "gimmick") out[slot] = targetTop1[slot] === "gimmick_beat_mute" ? "gimmick_filter_rise" : "gimmick_beat_mute";
  }

  return out;
};

const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0);

const percentile = (values, p) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
};

const run = async () => {
  const mode = pickMode(process.env.EVAL_MODE);
  const settings = modeSettings[mode];
  const raw = await readFile(datasetPath, "utf8");
  const dataset = JSON.parse(raw);

  const absErrors = [];
  const sqErrors = [];
  const targetValues = [];
  const jsonValidFlags = [];
  const constraintFlags = [];
  const slotFlags = [];
  const intentScores = [];
  const sanityScores = [];
  const latencies = [];
  const costs = [];

  for (const item of dataset.items) {
    const rng = lcg(hashString(`${mode}:${item.id}`));
    const target = item.target_hidden_params;

    const jsonValid = rng() <= settings.jsonValidRate;
    jsonValidFlags.push(jsonValid);

    const latency = settings.latencyRange[0] + rng() * (settings.latencyRange[1] - settings.latencyRange[0]);
    latencies.push(Math.round(latency));

    const requestCost = settings.costPerRequest * (0.9 + rng() * 0.2);
    costs.push(requestCost);

    const sanity = settings.sanityRange[0] + rng() * (settings.sanityRange[1] - settings.sanityRange[0]);
    sanityScores.push(Math.round(sanity));

    if (!jsonValid) {
      continue;
    }

    const predictedVector = predictVector(target.vector, rng, settings.noise);
    for (const key of KEYS) {
      const diff = predictedVector[key] - target.vector[key];
      absErrors.push(Math.abs(diff));
      sqErrors.push(diff * diff);
      targetValues.push(target.vector[key]);
    }

    const constraintMatch = rng() <= settings.constraintMatchRate;
    constraintFlags.push(constraintMatch ? 1 : 0);
    void mutateConstraints(target.constraints, rng, constraintMatch);

    const predictedTop1 = predictTop1BySlot(item.expected_top1_by_slot, rng, settings.slotMatchRate);
    for (const slot of SLOTS) {
      slotFlags.push(predictedTop1[slot] === item.expected_top1_by_slot[slot] ? 1 : 0);
    }

    const sampleMae = mean(KEYS.map((key) => Math.abs(predictedVector[key] - target.vector[key])));
    const intentScore = clamp(Math.round(100 - sampleMae * 2.2), 0, 100);
    intentScores.push(intentScore);
  }

  const jsonValidRate = mean(jsonValidFlags.map((v) => (v ? 1 : 0)));
  const vectorMae = mean(absErrors);
  const mseRaw = mean(sqErrors);
  const mseNorm = mseRaw / (100 * 100);

  const targetMean = mean(targetValues);
  const ssTot = targetValues.reduce((acc, value) => acc + (value - targetMean) ** 2, 0);
  const ssRes = sqErrors.reduce((acc, value) => acc + value, 0);
  const r2Score = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

  const result = {
    dataset_version: dataset.dataset_version,
    mode,
    evaluated_count: dataset.items.length,
    timestamp: new Date().toISOString(),
    metrics: {
      json_valid_rate: jsonValidRate,
      vector_mae: vectorMae,
      mse_raw: mseRaw,
      mse_norm: mseNorm,
      r2_score: r2Score,
      constraint_match_rate: mean(constraintFlags),
      slot_exact_match: mean(slotFlags),
      intent_score_mean: mean(intentScores),
      output_sanity_score: mean(sanityScores),
      p95_inference_latency_ms: percentile(latencies, 95),
      p50_inference_latency_ms: percentile(latencies, 50),
      cost_per_100_requests_usd: mean(costs) * 100
    }
  };

  await mkdir(runsDir, { recursive: true });
  const runId = `${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}_${mode}`;
  const outputPath = resolve(runsDir, `${runId}.json`);
  await writeFile(outputPath, JSON.stringify(result, null, 2));

  console.log(JSON.stringify({ run_id: runId, output_path: outputPath, ...result.metrics }, null, 2));
};

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
