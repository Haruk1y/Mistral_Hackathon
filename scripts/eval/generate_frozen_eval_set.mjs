import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { setTimeout as wait } from "node:timers/promises";

const root = resolve(new URL("../../", import.meta.url).pathname);
const outputPath = resolve(root, "data/eval/frozen_eval_set.v1.json");
const generationMode = process.env.EVAL_REQUEST_GENERATOR || "synthetic";
const useMistralTeacher = generationMode === "mistral_teacher";
const mistralTeacherModel = process.env.MISTRAL_TEACHER_MODEL || "mistral-large-latest";
const mistralBaseUrl = process.env.MISTRAL_BASE_URL || "https://api.mistral.ai/v1";
const mistralApiKey = process.env.MISTRAL_API_KEY || "";
const teacherAllowFallback = process.env.EVAL_TEACHER_ALLOW_FALLBACK === "true";
const teacherLang = process.env.EVAL_TEACHER_LANG || "en";
const teacherRetries = Math.max(0, Number(process.env.EVAL_TEACHER_MAX_RETRIES || 2));
const teacherTemperature = Number(process.env.EVAL_TEACHER_TEMPERATURE || 0.2);

const PARTS_BY_SLOT = {
  style: ["style_80s_citypop", "style_90s_hiphop", "style_2000s_pop"],
  instrument: ["inst_piano_upright", "inst_soft_strings", "inst_analog_synth"],
  mood: ["mood_rain_ambience", "mood_sun_glow", "mood_night_drive"],
  gimmick: ["gimmick_beat_mute", "gimmick_filter_rise", "gimmick_harmony_stack"]
};

const BASE_REQUESTS = [
  "A quiet evening after rain in a small town.",
  "A bright and smiling market morning tune.",
  "Calm focus music for reading in a cafe.",
  "A memory of old streets and warm lights.",
  "Soft night drive with nostalgic city air.",
  "Gentle handmade mood for twilight.",
  "A playful walk by the river on sunny day.",
  "A cozy indoor rainy day composition.",
  "A retro yet uplifting track for friends.",
  "A simple but emotional sketch for sunset."
];

const PROFILE_SCENARIOS = [
  "high_nostalgia",
  "low_brightness",
  "high_energy",
  "acoustic_focus",
  "rainy_quiet"
];

const WEATHER = ["sunny", "cloudy", "rainy"];
const VECTOR_KEYS = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];

const lcg = (seed) => {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
};

const toInt = (value, min, max) => Math.max(min, Math.min(max, Math.round(value)));

const makeVector = (rng, profile = "balanced") => {
  const base = {
    energy: toInt(25 + rng() * 55, 0, 100),
    warmth: toInt(30 + rng() * 50, 0, 100),
    brightness: toInt(20 + rng() * 60, 0, 100),
    acousticness: toInt(25 + rng() * 60, 0, 100),
    complexity: toInt(15 + rng() * 50, 0, 100),
    nostalgia: toInt(25 + rng() * 65, 0, 100)
  };

  if (profile === "high_nostalgia") base.nostalgia = toInt(72 + rng() * 24, 0, 100);
  if (profile === "low_brightness") base.brightness = toInt(4 + rng() * 22, 0, 100);
  if (profile === "high_energy") base.energy = toInt(72 + rng() * 24, 0, 100);
  if (profile === "acoustic_focus") base.acousticness = toInt(78 + rng() * 20, 0, 100);
  if (profile === "rainy_quiet") {
    base.energy = toInt(8 + rng() * 24, 0, 100);
    base.brightness = toInt(5 + rng() * 26, 0, 100);
    base.nostalgia = toInt(70 + rng() * 24, 0, 100);
  }

  return base;
};

const selectTop1 = (vector) => {
  const style =
    vector.brightness > 68 ? "style_2000s_pop" : vector.nostalgia > 65 ? "style_80s_citypop" : "style_90s_hiphop";
  const instrument =
    vector.acousticness > 70
      ? "inst_piano_upright"
      : vector.warmth > 64
        ? "inst_soft_strings"
        : "inst_analog_synth";
  const mood =
    vector.brightness < 35 ? "mood_rain_ambience" : vector.energy > 62 ? "mood_sun_glow" : "mood_night_drive";
  const gimmick =
    vector.complexity > 55
      ? "gimmick_harmony_stack"
      : vector.energy > 62
        ? "gimmick_filter_rise"
        : "gimmick_beat_mute";

  return { style, instrument, mood, gimmick };
};

const makeConstraints = (vector) => ({
  preferredStyleTags: vector.nostalgia > 65 ? ["citypop_80s"] : vector.brightness > 68 ? ["pop_2000s"] : ["hiphop_90s"],
  preferredGimmickTags: vector.energy > 62 ? ["filter_rise"] : ["beat_mute"],
  avoidPartIds: vector.brightness < 22 ? ["style_2000s_pop"] : []
});

const makeTags = (vector) => {
  const tags = [];
  if (vector.nostalgia > 65) tags.push("nostalgic");
  if (vector.energy > 62) tags.push("upbeat");
  if (vector.acousticness > 70) tags.push("acoustic");
  if (vector.brightness < 35) tags.push("rain");
  if (vector.warmth > 62) tags.push("cozy");
  return tags.length ? tags : ["balanced"];
};

const toBand = (value) => {
  if (value <= 25) return "very_low";
  if (value <= 45) return "low";
  if (value <= 65) return "mid";
  if (value <= 85) return "high";
  return "very_high";
};

const hasLeak = (text) =>
  /\b(?:energy|warmth|brightness|acousticness|complexity|nostalgia)\b/i.test(text) ||
  /["{[]\s*(?:energy|warmth|brightness|acousticness|complexity|nostalgia)\s*["}:]/i.test(text);

const teacherPromptFor = (item) => {
  const bands = VECTOR_KEYS.map((key) => `${key}:${toBand(item.target_hidden_params.vector[key])}`).join(", ");
  const constraints = JSON.stringify(item.target_hidden_params.constraints);

  return [
    `Language: ${teacherLang}`,
    "Write one natural user request sentence for a music composition game.",
    "Do not output JSON. Do not mention explicit numbers. Do not mention parameter names.",
    `Scenario=${item.scenario}`,
    `Weather=${item.weather}`,
    `VectorBands=${bands}`,
    `Tags=${item.target_hidden_params.tags.join(",")}`,
    `Constraints=${constraints}`,
    `SeedText=${item.request_text}`
  ].join("\n");
};

const callTeacher = async (item) => {
  if (!mistralApiKey) {
    throw new Error("MISTRAL_API_KEY is required when EVAL_REQUEST_GENERATOR=mistral_teacher");
  }

  const body = {
    model: mistralTeacherModel,
    messages: [
      {
        role: "system",
        content:
          "You create high-quality player request text for music commissions. Output exactly one short sentence only."
      },
      {
        role: "user",
        content: teacherPromptFor(item)
      }
    ],
    temperature: teacherTemperature,
    max_tokens: 80
  };

  const response = await fetch(`${mistralBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${mistralApiKey}`
    },
    body: JSON.stringify(body)
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`Teacher call failed: HTTP ${response.status} ${JSON.stringify(payload)}`);
  }

  const text = payload?.choices?.[0]?.message?.content;
  if (typeof text !== "string" || text.trim().length < 8) {
    throw new Error(`Teacher response invalid: ${JSON.stringify(payload)}`);
  }

  return text.replaceAll("\n", " ").trim();
};

const applyTeacherRequests = async (items) => {
  const out = [];

  for (let i = 0; i < items.length; i += 1) {
    const current = items[i];
    let text = current.request_text;
    let ok = false;

    for (let attempt = 0; attempt <= teacherRetries; attempt += 1) {
      try {
        const generated = await callTeacher(current);
        if (hasLeak(generated)) {
          throw new Error("Generated text has numeric/param leak");
        }

        text = generated;
        ok = true;
        break;
      } catch (error) {
        if (!teacherAllowFallback && attempt >= teacherRetries) {
          throw new Error(`Teacher generation failed for ${current.id}: ${error instanceof Error ? error.message : String(error)}`);
        }

        if (attempt < teacherRetries) {
          await wait(250 * (attempt + 1));
        }
      }
    }

    out.push({
      ...current,
      request_text: text,
      teacher_generated: ok
    });

    if ((i + 1) % 10 === 0) {
      console.log(`Teacher generation progress: ${i + 1}/${items.length}`);
    }
  }

  return out;
};

const buildFixed50 = () => {
  const rows = [];
  let id = 1;

  for (let i = 0; i < 50; i += 1) {
    const rng = lcg(1000 + i);
    const requestBase = BASE_REQUESTS[i % BASE_REQUESTS.length];
    const variation = i % 5;
    const requestText = `${requestBase} Variant-${variation + 1}`;
    const weather = WEATHER[i % WEATHER.length];
    const vector = makeVector(rng);
    const expectedTop1BySlot = selectTop1(vector);

    rows.push({
      id: `fixed_${String(id).padStart(3, "0")}`,
      split: "frozen_eval",
      scenario: "fixed_request",
      request_text: requestText,
      weather,
      inventory_part_ids: Object.values(PARTS_BY_SLOT).flat(),
      expected_top1_by_slot: expectedTop1BySlot,
      target_hidden_params: {
        vector,
        tags: makeTags(vector),
        constraints: makeConstraints(vector)
      }
    });

    id += 1;
  }

  return rows;
};

const buildProfile20 = () => {
  const rows = [];
  for (let i = 0; i < 20; i += 1) {
    const scenario = PROFILE_SCENARIOS[i % PROFILE_SCENARIOS.length];
    const rng = lcg(5000 + i);
    const vector = makeVector(rng, scenario);

    rows.push({
      id: `profile_${String(i + 1).padStart(3, "0")}`,
      split: "frozen_eval",
      scenario,
      request_text: `Profile scenario ${scenario.replaceAll("_", " ")} sample ${i + 1}`,
      weather: WEATHER[(i + 1) % WEATHER.length],
      inventory_part_ids: Object.values(PARTS_BY_SLOT).flat(),
      expected_top1_by_slot: selectTop1(vector),
      target_hidden_params: {
        vector,
        tags: makeTags(vector),
        constraints: makeConstraints(vector)
      }
    });
  }

  return rows;
};

const main = async () => {
  if (useMistralTeacher && !mistralApiKey && !teacherAllowFallback) {
    throw new Error(
      "MISTRAL_API_KEY is missing. Set it, or set EVAL_TEACHER_ALLOW_FALLBACK=true if you explicitly want synthetic fallback."
    );
  }

  const baseData = [...buildFixed50(), ...buildProfile20()];
  const data = useMistralTeacher ? await applyTeacherRequests(baseData) : baseData;

  if (data.length !== 70) {
    throw new Error(`Frozen eval set must have 70 rows, got ${data.length}`);
  }

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(
    outputPath,
    JSON.stringify(
      {
        dataset_version: useMistralTeacher ? "frozen_eval_set.v1_teacher_mistral" : "frozen_eval_set.v1",
        created_at: new Date().toISOString(),
        request_generation_mode: generationMode,
        teacher_model: useMistralTeacher ? mistralTeacherModel : null,
        count: data.length,
        items: data
      },
      null,
      2
    )
  );

  console.log(
    `Wrote ${data.length} rows to ${outputPath} (mode=${generationMode}${useMistralTeacher ? `, teacher=${mistralTeacherModel}` : ""})`
  );
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
