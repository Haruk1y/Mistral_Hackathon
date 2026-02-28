import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { nanoid } from "nanoid";

const root = resolve(new URL("../../", import.meta.url).pathname);
const datasetPath = resolve(root, process.env.EVAL_DATASET_PATH || "data/eval/frozen_eval_set.v1.json");
const runsDir = resolve(root, "artifacts/eval/runs");
const samplesDir = resolve(root, "artifacts/eval/samples");

const KEYS = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];
const SLOTS = ["style", "instrument", "mood", "gimmick"];
const evalScaleEnv = Number(process.env.EVAL_TARGET_SCALE || process.env.HF_FT_TARGET_SCALE || process.env.FT_TARGET_SCALE || "10");
if (evalScaleEnv !== 10) {
  throw new Error(`EVAL_TARGET_SCALE must be 10 for this pipeline. got=${evalScaleEnv}`);
}
const EVAL_TARGET_SCALE = 10;
const THRESHOLD_SCALE = 100 / EVAL_TARGET_SCALE;

const asNumber = (value, fallback = 0) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const mean = (values) => (values.length ? values.reduce((acc, value) => acc + value, 0) / values.length : 0);
const scaleFrom100 = (value) => Math.round(clamp((Number(value) / 100) * EVAL_TARGET_SCALE, 0, EVAL_TARGET_SCALE));
const threshold = (value100) => (Number(value100) / 100) * EVAL_TARGET_SCALE;
const inferScale = (vector) => {
  const values = KEYS.map((key) => Number(vector?.[key])).filter((value) => Number.isFinite(value));
  const maxValue = values.length ? Math.max(...values) : EVAL_TARGET_SCALE;
  if (maxValue <= 10) return 10;
  if (maxValue <= 100) return 100;
  return EVAL_TARGET_SCALE;
};
const normalizeToEvalScale = (vector, sourceScale) => {
  const src = Math.max(1, Number(sourceScale) || EVAL_TARGET_SCALE);
  const out = {};
  for (const key of KEYS) {
    const value = Number(vector?.[key]);
    const bounded = Number.isFinite(value) ? clamp(value, 0, src) : 0;
    out[key] = Math.round(clamp((bounded / src) * EVAL_TARGET_SCALE, 0, EVAL_TARGET_SCALE));
  }
  return out;
};
const percentile = (values, p) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
};

const mode = (() => {
  const raw = process.env.EVAL_MODE || "prompt_baseline";
  if (["rule_baseline", "prompt_baseline", "fine_tuned"].includes(raw)) return raw;
  return "prompt_baseline";
})();

const promptModelId = process.env.EVAL_PROMPT_BASELINE_MODEL_ID || process.env.HF_BASE_MODEL_ID || "mistralai/Ministral-3-3B-Instruct-2512";
const fineTunedModelId = process.env.EVAL_FINE_TUNED_MODEL_ID || process.env.HF_FT_OUTPUT_MODEL_ID || promptModelId;
const modelId = mode === "fine_tuned" ? fineTunedModelId : mode === "prompt_baseline" ? promptModelId : "rule_based_keyword_v1";
const hfToken = process.env.HF_TOKEN || process.env.HF_API_TOKEN || "";
const hfInferenceBaseUrl = (process.env.HF_INFERENCE_BASE_URL || "https://router.huggingface.co/hf-inference/models").replace(/\/$/, "");
const mistralApiKey = process.env.MISTRAL_API_KEY || "";
const mistralBaseUrl = (process.env.MISTRAL_BASE_URL || "https://api.mistral.ai/v1").replace(/\/$/, "");
const mistralPromptModelId = process.env.EVAL_MISTRAL_PROMPT_MODEL_ID || process.env.MISTRAL_BASE_MODEL || "mistral-small-latest";
const mistralFineTunedModelId =
  process.env.EVAL_MISTRAL_FINE_TUNED_MODEL_ID || process.env.MISTRAL_FINE_TUNED_MODEL_ID || process.env.MISTRAL_FT_MODEL_ID || "";
const allowMistralFallback = !["0", "false", "no"].includes((process.env.EVAL_MISTRAL_FALLBACK_ENABLED || "true").toLowerCase());
const traceProject = process.env.WEAVE_PROJECT || "atelier-kotone-weave";
const traceEntity = process.env.WANDB_ENTITY || "";
const topFailures = Math.max(1, Number(process.env.EVAL_TOP_FAILURES || 20));
const costs = {
  rule_baseline: Number(process.env.EVAL_COST_PER_REQUEST_RULE_BASELINE || 0.0002),
  prompt_baseline: Number(process.env.EVAL_COST_PER_REQUEST_PROMPT_BASELINE || 0.0028),
  fine_tuned: Number(process.env.EVAL_COST_PER_REQUEST_FINE_TUNED || 0.0019),
};

const extractJsonBlock = (text) => {
  const trimmed = text.trim();
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return trimmed;
  } catch {}
  const fenced = trimmed.match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) return fenced[1].trim();
  const first = trimmed.indexOf("{");
  const last = trimmed.lastIndexOf("}");
  if (first >= 0 && last > first) return trimmed.slice(first, last + 1);
  return null;
};

const extractVectorPayload = (payload) => {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error("parsed_payload_not_object");
  }

  const candidates = [
    payload,
    payload.hidden_params,
    payload.target_hidden_params,
    payload.targetHiddenParams,
    payload.target_vector,
    payload.vector,
  ];

  for (const candidate of candidates) {
    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      if (KEYS.every((key) => key in candidate)) return candidate;
      if (candidate.vector && typeof candidate.vector === "object" && KEYS.every((key) => key in candidate.vector)) {
        return candidate.vector;
      }
    }
  }

  throw new Error("vector_payload_not_found");
};

const sanitizeVector = (payload) => {
  const vectorPayload = extractVectorPayload(payload);
  const out = {};
  for (const key of KEYS) {
    out[key] = Math.round(clamp(Number(vectorPayload[key]), 0, EVAL_TARGET_SCALE));
  }
  return out;
};

const deriveConstraints = (vector) => ({
  preferredStyleTags: vector.nostalgia > threshold(60) ? ["citypop_80s"] : vector.brightness > threshold(65) ? ["pop_2000s"] : ["hiphop_90s"],
  preferredGimmickTags: vector.energy > threshold(60) ? ["filter_rise"] : ["beat_mute"],
  avoidPartIds: vector.brightness < threshold(22) ? ["style_2000s_pop"] : [],
});

const deriveTop1 = (vector) => ({
  style: vector.brightness > threshold(68) ? "style_2000s_pop" : vector.nostalgia > threshold(65) ? "style_80s_citypop" : "style_90s_hiphop",
  instrument:
    vector.acousticness > threshold(70)
      ? "inst_piano_upright"
      : vector.warmth > threshold(64)
        ? "inst_soft_strings"
        : "inst_analog_synth",
  mood: vector.brightness < threshold(35) ? "mood_rain_ambience" : vector.energy > threshold(62) ? "mood_sun_glow" : "mood_night_drive",
  gimmick:
    vector.complexity > threshold(55)
      ? "gimmick_harmony_stack"
      : vector.energy > threshold(62)
        ? "gimmick_filter_rise"
        : "gimmick_beat_mute",
});

const rulePredict = (requestText) => {
  const lower = requestText.toLowerCase();
  if (/(rain|quiet|evening|night)/.test(lower)) {
    return {
      energy: scaleFrom100(22),
      warmth: scaleFrom100(60),
      brightness: scaleFrom100(24),
      acousticness: scaleFrom100(72),
      complexity: scaleFrom100(32),
      nostalgia: scaleFrom100(72),
    };
  }
  if (/(smile|bright|market|sun)/.test(lower)) {
    return {
      energy: scaleFrom100(75),
      warmth: scaleFrom100(58),
      brightness: scaleFrom100(82),
      acousticness: scaleFrom100(38),
      complexity: scaleFrom100(48),
      nostalgia: scaleFrom100(42),
    };
  }
  if (/(focus|study|reading|cafe)/.test(lower)) {
    return {
      energy: scaleFrom100(34),
      warmth: scaleFrom100(54),
      brightness: scaleFrom100(44),
      acousticness: scaleFrom100(68),
      complexity: scaleFrom100(28),
      nostalgia: scaleFrom100(58),
    };
  }
  if (/(memory|old|nostalgia|retro)/.test(lower)) {
    return {
      energy: scaleFrom100(40),
      warmth: scaleFrom100(74),
      brightness: scaleFrom100(50),
      acousticness: scaleFrom100(76),
      complexity: scaleFrom100(40),
      nostalgia: scaleFrom100(88),
    };
  }
  return {
    energy: scaleFrom100(45),
    warmth: scaleFrom100(55),
    brightness: scaleFrom100(50),
    acousticness: scaleFrom100(65),
    complexity: scaleFrom100(38),
    nostalgia: scaleFrom100(60),
  };
};

const hfPredict = async (requestText, weather) => {
  if (!hfToken) return { vector: null, parseError: "HF_TOKEN missing" };
  try {
    const prompt = [
      "You estimate 6 hidden music parameters.",
      "Return strict JSON only with schema:",
      '{"request_text":"...", "hidden_params":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
      "Copy request_text exactly from input.",
      `Each hidden_params value must be integer between 0 and ${EVAL_TARGET_SCALE}.`,
      `weather=${weather}`,
      `request_text=${requestText}`,
    ].join("\n");

    const response = await fetch(`${hfInferenceBaseUrl}/${modelId}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${hfToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens: 96,
          temperature: 0.1,
          return_full_text: false,
        },
      }),
    });
    if (!response.ok) {
      const body = await response.text();
      return { vector: null, parseError: `http_${response.status}:${String(body || "").slice(0, 240)}` };
    }
    const payload = await response.json();
    const text = Array.isArray(payload) ? payload[0]?.generated_text || "" : payload?.generated_text || "";
    const block = extractJsonBlock(String(text || ""));
    if (!block) return { vector: null, parseError: "json_block_not_found" };
    const parsed = JSON.parse(block);
    return { vector: sanitizeVector(parsed), parseError: null };
  } catch (error) {
    return { vector: null, parseError: error instanceof Error ? error.message : String(error) };
  }
};

const shouldTryMistralFallback = (parseError) => {
  if (!parseError) return false;
  const lowered = parseError.toLowerCase();
  return (
    lowered.startsWith("http_404") ||
    lowered.startsWith("http_410") ||
    lowered.includes("no longer supported") ||
    lowered.includes("model_not_supported")
  );
};

const mistralPredict = async (requestText, weather, mistralModelId) => {
  if (!mistralApiKey) return { vector: null, parseError: "MISTRAL_API_KEY missing" };
  if (!mistralModelId) return { vector: null, parseError: "Mistral model_id missing" };

  try {
    const prompt = [
      "You estimate 6 hidden music parameters.",
      "Return strict JSON only with schema:",
      '{"request_text":"...", "hidden_params":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
      "Copy request_text exactly from input.",
      `Each hidden_params value must be integer between 0 and ${EVAL_TARGET_SCALE}.`,
      `weather=${weather}`,
      `request_text=${requestText}`,
    ].join("\n");
    const response = await fetch(`${mistralBaseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${mistralApiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: mistralModelId,
        temperature: 0.1,
        max_tokens: 96,
        messages: [
          {
            role: "system",
            content: "You output only strict JSON with six integer fields.",
          },
          {
            role: "user",
            content: prompt,
          },
        ],
      }),
    });
    if (!response.ok) {
      const body = await response.text();
      return { vector: null, parseError: `mistral_http_${response.status}:${String(body || "").slice(0, 240)}` };
    }

    const payload = await response.json();
    let content = payload?.choices?.[0]?.message?.content ?? "";
    if (Array.isArray(content)) {
      content = content
        .map((part) => {
          if (!part || typeof part !== "object") return "";
          if (part.type === "text" || part.type === "output_text") return String(part.text || "");
          if ("content" in part) return String(part.content || "");
          return "";
        })
        .join("");
    }
    const block = extractJsonBlock(String(content || ""));
    if (!block) return { vector: null, parseError: "mistral_json_block_not_found" };
    const parsed = JSON.parse(block);
    return { vector: sanitizeVector(parsed), parseError: null };
  } catch (error) {
    return { vector: null, parseError: error instanceof Error ? error.message : String(error) };
  }
};

const computeSanity = (jsonValid, vector) => {
  if (!jsonValid) return 12;
  const spread = Math.max(...KEYS.map((key) => vector[key])) - Math.min(...KEYS.map((key) => vector[key]));
  const spread100 = spread * THRESHOLD_SCALE;
  return Math.round(clamp(70 + spread100 * 0.3, 0, 100));
};

const traceUrl = (traceId) => {
  if (!traceProject) return null;
  if (traceEntity) return `https://wandb.ai/${traceEntity}/${traceProject}/weave/traces?query=${traceId}`;
  return `https://wandb.ai/${traceProject}/weave/traces?query=${traceId}`;
};

const run = async () => {
  const dataset = JSON.parse(await readFile(datasetPath, "utf8"));
  const rows = [];
  const absErrors = [];
  const sqErrors = [];
  const targetValues = [];
  const jsonFlags = [];
  const constraintFlags = [];
  const slotFlags = [];
  const intentScores = [];
  const sanityScores = [];
  const latencies = [];
  const costList = [];

  for (const item of dataset.items || []) {
    const startedAt = performance.now();
    const traceId = `eval_${mode}_${nanoid(10)}`;
    let vector = null;
    let parseError = null;
    let effectiveModelId = modelId;
    let inferenceBackend = "rule_based";
    if (mode === "rule_baseline") {
      vector = rulePredict(item.request_text || "");
    } else {
      const predicted = await hfPredict(item.request_text || "", item.weather || "sunny");
      vector = predicted.vector;
      parseError = predicted.parseError;
      inferenceBackend = "hf_router_hf_inference";
      if (!vector && allowMistralFallback && shouldTryMistralFallback(parseError)) {
        const fallbackModelId = mode === "fine_tuned" ? mistralFineTunedModelId || mistralPromptModelId : mistralPromptModelId;
        const fallback = await mistralPredict(item.request_text || "", item.weather || "sunny", fallbackModelId);
        if (fallback.vector) {
          vector = fallback.vector;
          parseError = null;
          effectiveModelId = fallbackModelId;
          inferenceBackend = "mistral_chat_fallback";
        } else if (fallback.parseError) {
          parseError = parseError ? `${parseError};${fallback.parseError}` : fallback.parseError;
        }
      }
    }

    const latencyMs = Math.max(0, performance.now() - startedAt);
    const jsonValid = Boolean(vector);
    jsonFlags.push(jsonValid ? 1 : 0);
    latencies.push(latencyMs);
    costList.push(costs[mode]);
    const rawTargetVector = item.target_hidden_params?.vector || {};
    const sourceScale = Number(item.target_scale || item.target_hidden_params?.target_scale || inferScale(rawTargetVector));
    const targetVector = normalizeToEvalScale(rawTargetVector, sourceScale);

    const row = {
      id: item.id,
      mode,
      scenario: item.scenario,
      source_type: item.source_type || item.scenario || "unknown",
      request_text: item.request_text,
      weather: item.weather,
      model_source: mode,
      model_id: modelId,
      effective_model_id: effectiveModelId,
      inference_backend: inferenceBackend,
      trace_id: traceId,
      trace_url: traceUrl(traceId),
      json_valid: jsonValid,
      parse_error: parseError,
      latency_ms: Math.round(latencyMs),
      cost_usd: costs[mode],
      output_sanity_score: 0,
      target_vector: targetVector,
    };

    if (!jsonValid) {
      row.output_sanity_score = computeSanity(false, null);
      sanityScores.push(row.output_sanity_score);
      rows.push(row);
      continue;
    }

    const absErrorByDim = {};
    for (const key of KEYS) {
      const target = asNumber(targetVector[key], 0);
      const pred = asNumber(vector[key], 0);
      const diff = pred - target;
      absErrors.push(Math.abs(diff));
      sqErrors.push(diff * diff);
      targetValues.push(target);
      absErrorByDim[key] = Math.abs(diff);
    }

    const predictedConstraints = deriveConstraints(vector);
    const targetConstraints = deriveConstraints(targetVector);
    const constraintMatch = JSON.stringify(predictedConstraints) === JSON.stringify(targetConstraints);
    constraintFlags.push(constraintMatch ? 1 : 0);

    const predictedTop1 = deriveTop1(vector);
    const expectedTop1 = deriveTop1(targetVector);
    let slotMatches = 0;
    for (const slot of SLOTS) {
      const hit = predictedTop1[slot] === expectedTop1[slot];
      slotFlags.push(hit ? 1 : 0);
      if (hit) slotMatches += 1;
    }

    const mae = mean(Object.values(absErrorByDim));
    const intent = clamp(100 - mae * THRESHOLD_SCALE * 2.1, 0, 100);
    const sanity = computeSanity(true, vector);
    intentScores.push(intent);
    sanityScores.push(sanity);

    row.predicted_vector = vector;
    row.abs_error_by_dim = absErrorByDim;
    row.mae_raw = mae;
    row.constraint_match = constraintMatch;
    row.slot_exact_match = slotMatches / SLOTS.length;
    row.intent_score = intent;
    row.output_sanity_score = sanity;
    rows.push(row);
  }

  const targetMean = mean(targetValues);
  const ssTot = targetValues.reduce((acc, value) => acc + (value - targetMean) ** 2, 0);
  const ssRes = sqErrors.reduce((acc, value) => acc + value, 0);
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

  const metrics = {
    target_scale: EVAL_TARGET_SCALE,
    json_valid_rate: mean(jsonFlags),
    vector_mae: mean(absErrors),
    mse_raw: mean(sqErrors),
    mse_norm: mean(sqErrors) / (EVAL_TARGET_SCALE * EVAL_TARGET_SCALE),
    r2_score: r2,
    constraint_match_rate: mean(constraintFlags),
    slot_exact_match: mean(slotFlags),
    intent_score_mean: mean(intentScores),
    output_sanity_score: mean(sanityScores),
    p95_inference_latency_ms: percentile(latencies, 95),
    p50_inference_latency_ms: percentile(latencies, 50),
    cost_per_100_requests_usd: mean(costList) * 100,
  };

  const failuresTopK = rows
    .filter((row) => row.json_valid && typeof row.mae_raw === "number")
    .sort((a, b) => b.mae_raw - a.mae_raw)
    .slice(0, topFailures);

  await mkdir(runsDir, { recursive: true });
  await mkdir(samplesDir, { recursive: true });
  const runId = `${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}_${mode}`;
  const runPath = resolve(runsDir, `${runId}.json`);
  const samplePath = resolve(samplesDir, `${runId}.json`);

  const runPayload = {
    dataset_version: dataset.dataset_version || "unknown",
    target_scale: EVAL_TARGET_SCALE,
    mode,
    model_source: mode,
    model_id: modelId,
    evaluated_count: rows.length,
    timestamp: new Date().toISOString(),
    metrics,
  };
  const samplePayload = {
    run_id: runId,
    mode,
    target_scale: EVAL_TARGET_SCALE,
    dataset_version: dataset.dataset_version || "unknown",
    model_source: mode,
    model_id: modelId,
    rows,
    failures_top_k: failuresTopK,
  };

  await writeFile(runPath, JSON.stringify(runPayload, null, 2));
  await writeFile(samplePath, JSON.stringify(samplePayload, null, 2));

  console.log(
    JSON.stringify(
      {
        run_id: runId,
        output_path: runPath,
        sample_path: samplePath,
        mode,
        model_id: modelId,
        ...metrics,
      },
      null,
      2,
    ),
  );
};

run().catch((error) => {
  console.error(error);
  process.exit(1);
});
