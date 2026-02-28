import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { setTimeout as wait } from "node:timers/promises";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const outputPath = resolve(root, "data/ft/teacher_pairs.raw.jsonl");

const count = Math.max(50, Number(process.env.TEACHER_PAIR_COUNT || 500));
const seed = Number(process.env.TEACHER_SEED || 20260228);

const generationMode = process.env.TEACHER_TARGET_GENERATOR || process.env.TEACHER_REQUEST_GENERATOR || "mistral_large";
const useMistralTeacher = generationMode !== "synthetic";
const mistralTeacherModel = process.env.MISTRAL_TEACHER_MODEL || "mistral-large-latest";
const mistralBaseUrl = process.env.MISTRAL_BASE_URL || "https://api.mistral.ai/v1";
const mistralApiKey = process.env.MISTRAL_API_KEY || "";

const teacherLang = process.env.TEACHER_LANG || "en";
const teacherBatchSize = Math.max(1, Number(process.env.TEACHER_BATCH_SIZE || 12));
const teacherRetries = Math.max(0, Number(process.env.TEACHER_MAX_RETRIES || 2));
const teacherTemperature = Number(process.env.TEACHER_TEMPERATURE || 0.2);
const teacherAllowFallback = process.env.TEACHER_ALLOW_FALLBACK === "true";

const VECTOR_KEYS = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];
const SLOT_KEYS = ["style", "instrument", "mood", "gimmick"];

const STYLE_PARTS = [
  {
    id: "style_80s_citypop",
    slot: "style",
    name: "80s City Pop",
    styleTag: "citypop_80s",
    vector: { energy: 52, warmth: 74, brightness: 62, complexity: 40, nostalgia: 78 }
  },
  {
    id: "style_90s_hiphop",
    slot: "style",
    name: "90s Hip-Hop",
    styleTag: "hiphop_90s",
    vector: { energy: 58, warmth: 52, brightness: 44, complexity: 46, nostalgia: 66 }
  },
  {
    id: "style_2000s_pop",
    slot: "style",
    name: "2000s Pop",
    styleTag: "pop_2000s",
    vector: { energy: 76, warmth: 55, brightness: 82, complexity: 38, nostalgia: 48 }
  },
  {
    id: "style_70s_folk",
    slot: "style",
    name: "70s Folk",
    styleTag: "folk_70s",
    vector: { energy: 44, warmth: 78, brightness: 46, complexity: 34, nostalgia: 86 }
  },
  {
    id: "style_2010s_edm",
    slot: "style",
    name: "2010s EDM",
    styleTag: "edm_2010s",
    vector: { energy: 86, warmth: 38, brightness: 90, complexity: 58, nostalgia: 34 }
  },
  {
    id: "style_60s_parade_jazz",
    slot: "style",
    name: "60s Parade Jazz",
    styleTag: "jazz_60s",
    vector: { energy: 66, warmth: 63, brightness: 60, complexity: 56, nostalgia: 74 }
  },
  {
    id: "style_2010s_dance_pop",
    slot: "style",
    name: "2010s Dance Pop",
    styleTag: "dancepop_2010s",
    vector: { energy: 82, warmth: 57, brightness: 78, complexity: 46, nostalgia: 40 }
  }
];

const INSTRUMENT_PARTS = [
  {
    id: "inst_piano_upright",
    slot: "instrument",
    name: "Upright Piano",
    vector: { acousticness: 88, warmth: 70, nostalgia: 78, complexity: 44 }
  },
  {
    id: "inst_soft_strings",
    slot: "instrument",
    name: "Fairy Harp",
    vector: { warmth: 75, acousticness: 72, complexity: 50, nostalgia: 64 }
  },
  {
    id: "inst_analog_synth",
    slot: "instrument",
    name: "Snake Music Box",
    vector: { energy: 52, brightness: 58, acousticness: 36, complexity: 54, nostalgia: 70 }
  },
  {
    id: "inst_cello_warm",
    slot: "instrument",
    name: "Warm Cello",
    vector: { energy: 42, warmth: 86, brightness: 34, acousticness: 88, complexity: 52, nostalgia: 72 }
  },
  {
    id: "inst_guitar_street",
    slot: "instrument",
    name: "Street Guitar",
    vector: { energy: 65, warmth: 69, brightness: 58, acousticness: 84, complexity: 50, nostalgia: 66 }
  },
  {
    id: "inst_musicbox_garden",
    slot: "instrument",
    name: "Garden Music Box",
    vector: { energy: 38, warmth: 62, brightness: 66, acousticness: 42, complexity: 40, nostalgia: 84 }
  },
  {
    id: "inst_violin_vintage",
    slot: "instrument",
    name: "Vintage Violin",
    vector: { energy: 50, warmth: 71, brightness: 60, acousticness: 84, complexity: 59, nostalgia: 80 }
  }
];

const MOOD_PARTS = [
  {
    id: "mood_rain_ambience",
    slot: "mood",
    name: "Rain Ambience",
    vector: { warmth: 50, brightness: 28, nostalgia: 84, complexity: 20 }
  },
  {
    id: "mood_sun_glow",
    slot: "mood",
    name: "Sun Glow",
    vector: { warmth: 84, brightness: 74, nostalgia: 58 }
  },
  {
    id: "mood_night_drive",
    slot: "mood",
    name: "Night Drive",
    vector: { energy: 55, warmth: 42, brightness: 36, complexity: 46, nostalgia: 50 }
  },
  {
    id: "mood_cozy_hearth",
    slot: "mood",
    name: "Cozy Hearth",
    vector: { energy: 34, warmth: 90, brightness: 48, complexity: 20, nostalgia: 72 }
  },
  {
    id: "mood_hope_star",
    slot: "mood",
    name: "Hope Star",
    vector: { energy: 62, warmth: 68, brightness: 80, complexity: 24, nostalgia: 46 }
  },
  {
    id: "mood_pocket_memory",
    slot: "mood",
    name: "Pocket Memory",
    vector: { energy: 36, warmth: 72, brightness: 50, complexity: 26, nostalgia: 90 }
  }
];

const GIMMICK_PARTS = [
  {
    id: "gimmick_beat_mute",
    slot: "gimmick",
    name: "Beat Mute",
    gimmickTag: "beat_mute",
    vector: { complexity: 52, energy: 60, nostalgia: 44 }
  },
  {
    id: "gimmick_filter_rise",
    slot: "gimmick",
    name: "Filter Rise",
    gimmickTag: "filter_rise",
    vector: { complexity: 56, energy: 72, brightness: 68 }
  },
  {
    id: "gimmick_harmony_stack",
    slot: "gimmick",
    name: "Harmony Stack",
    gimmickTag: "harmony_stack",
    vector: { warmth: 70, complexity: 58, nostalgia: 62 }
  },
  {
    id: "gimmick_campfire_crackle",
    slot: "gimmick",
    name: "Campfire Crackle",
    gimmickTag: "campfire_crackle",
    vector: { energy: 36, warmth: 82, brightness: 34, complexity: 42, nostalgia: 78 }
  },
  {
    id: "gimmick_rainfall",
    slot: "gimmick",
    name: "Rainfall Layer",
    gimmickTag: "rainfall",
    vector: { energy: 28, warmth: 46, brightness: 32, complexity: 40, nostalgia: 72 }
  },
  {
    id: "gimmick_river_flow",
    slot: "gimmick",
    name: "River Flow",
    gimmickTag: "river_flow",
    vector: { energy: 34, warmth: 57, brightness: 44, complexity: 44, nostalgia: 63 }
  }
];

const REQUEST_OPENERS_EN = ["I want", "I need", "Could you make", "Can you make", "Please compose", "I'd love"];
const REQUEST_SCENES_EN = [
  "a lantern-lit corner after rain",
  "a quiet market street at dusk",
  "an old town alley with soft echoes",
  "a small square in evening light",
  "a calm riverside walk at night",
  "a cozy game scene in a tiny town",
  "a sunset path with gentle footsteps",
  "a nostalgic backstreet with warm air"
];
const REQUEST_FEELS_EN = [
  "keeping it calm and emotional",
  "with a handcrafted and intimate feel",
  "with a mellow and reflective mood",
  "with a soft and memory-like tone",
  "while staying simple and loop-friendly",
  "without becoming too heavy or dramatic",
  "with gentle motion and steady flow",
  "with cozy retro color"
];
const REQUEST_FLOWS_EN = [
  "for a short instrumental loop",
  "for a compact game background track",
  "for a brief scene transition cue",
  "for a subtle background layer",
  "for a loop that can repeat naturally",
  "for a small but expressive cue"
];

const REQUEST_OPENERS_JA = ["〜な曲にしてください", "〜な雰囲気でお願いします", "〜っぽい曲がほしいです"];
const REQUEST_SCENES_JA = [
  "雨上がりのランタン通り",
  "夕方の小さな商店街",
  "古い街角の路地",
  "夜の川沿いの散歩道",
  "夕焼けの石畳",
  "小さな町の広場",
  "静かな住宅街の曲がり角",
  "記憶の残る細い道"
];
const REQUEST_FEELS_JA = [
  "落ち着いていて感情がにじむ",
  "手作り感があって親密な",
  "やわらかくて余韻が残る",
  "懐かしさを感じる",
  "重すぎず自然に流れる",
  "シンプルでループしやすい",
  "静けさと温度感がある",
  "少しレトロな色味を持つ"
];

const ALL_PARTS = [...STYLE_PARTS, ...INSTRUMENT_PARTS, ...MOOD_PARTS, ...GIMMICK_PARTS];
const PART_BY_ID = new Map(ALL_PARTS.map((part) => [part.id, part]));

const lcg = (seedValue) => {
  let value = seedValue >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
};

const rng = lcg(seed);

const clamp = (value, min, max) => Math.max(min, Math.min(max, Math.round(value)));
const pick = (array) => array[Math.floor(rng() * array.length) % array.length];

const cleanText = (value) =>
  String(value || "")
    .replaceAll("\n", " ")
    .replace(/\s+/g, " ")
    .trim();

const toVector = (input) => {
  const out = {};
  for (const key of VECTOR_KEYS) {
    out[key] = clamp(Number(input?.[key] ?? 0), 0, 100);
  }
  return out;
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

const extractJsonArray = (content) => {
  const text = cleanText(content);

  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed;
  } catch {
    // no-op
  }

  const start = text.indexOf("[");
  const end = text.lastIndexOf("]");
  if (start < 0 || end <= start) return null;

  try {
    const parsed = JSON.parse(text.slice(start, end + 1));
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
};

const extractMessageContent = (payload) => {
  const raw = payload?.choices?.[0]?.message?.content;
  if (typeof raw === "string") return raw;
  if (Array.isArray(raw)) {
    return raw
      .map((part) => {
        if (!part || typeof part !== "object") return "";
        if (typeof part.text === "string") return part.text;
        if (typeof part.content === "string") return part.content;
        return "";
      })
      .join(" ");
  }
  return "";
};

const weatherFromVector = (vector) => {
  if (vector.brightness < 35) return "rainy";
  if (vector.brightness < 60) return "cloudy";
  return "sunny";
};

const buildRulePrompt = (selectedPartsBySlot) => {
  const lines = [
    "Compose nostalgic retro pixel-town background music.",
    "Return instrumental music suitable for a game scene.",
    "This is a rule-based prompt generated from selected Kotone parts.",
    "Selected Kotone combination:",
    ...SLOT_KEYS.map((slot) => {
      const partId = selectedPartsBySlot[slot];
      const name = PART_BY_ID.get(partId)?.name || partId;
      return `- ${slot}: ${name} (${partId})`;
    }),
    "Style: warm, cozy, handcrafted, street evening, non-vocal, emotional but simple."
  ];
  return lines.join("\n");
};

const synthesizeVectorFromParts = (selectedPartsBySlot) => {
  const sums = Object.fromEntries(VECTOR_KEYS.map((key) => [key, 0]));
  const counts = Object.fromEntries(VECTOR_KEYS.map((key) => [key, 0]));

  for (const slot of SLOT_KEYS) {
    const part = PART_BY_ID.get(selectedPartsBySlot[slot]);
    if (!part) continue;

    for (const key of VECTOR_KEYS) {
      const value = Number(part.vector?.[key]);
      if (!Number.isFinite(value)) continue;
      sums[key] += value;
      counts[key] += 1;
    }
  }

  const vector = {};
  for (const key of VECTOR_KEYS) {
    const base = counts[key] > 0 ? sums[key] / counts[key] : 50;
    vector[key] = clamp(base + (rng() * 12 - 6), 0, 100);
  }
  return vector;
};

const buildFallbackVector = (record) => synthesizeVectorFromParts(record.selected_parts_by_slot);

const normalizeVectorCandidate = (candidate) => {
  if (!candidate || typeof candidate !== "object") return null;

  if (VECTOR_KEYS.every((key) => key in candidate)) {
    return toVector(candidate);
  }

  if (candidate.vector && typeof candidate.vector === "object" && VECTOR_KEYS.every((key) => key in candidate.vector)) {
    return toVector(candidate.vector);
  }

  return null;
};

const callMistralBatch = async (batch) => {
  if (!mistralApiKey) {
    throw new Error("MISTRAL_API_KEY is required for Mistral teacher inference mode.");
  }

  const items = batch
    .map((item) => [`IDX=${item.meta.index}`, item.rule_prompt].join("\n"))
    .join("\n\n---\n\n");

  const response = await fetch(`${mistralBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${mistralApiKey}`
    },
    body: JSON.stringify({
      model: mistralTeacherModel,
      temperature: teacherTemperature,
      max_tokens: Math.max(320, batch.length * 120),
      messages: [
        {
          role: "system",
          content:
            "You infer hidden music parameters from rule prompts. Return strict JSON array only, no prose."
        },
        {
          role: "user",
          content: [
            "For each item, estimate six hidden params from the prompt semantics.",
            "Scale: integer 0..100.",
            "Output format exactly:",
            '[{"idx":123,"vector":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}]',
            "Items:",
            items
          ].join("\n")
        }
      ]
    })
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`Mistral batch call failed: HTTP ${response.status} ${JSON.stringify(payload)}`);
  }

  const content = extractMessageContent(payload);
  const rows = extractJsonArray(content);
  if (!rows) {
    throw new Error(`Mistral batch output is not JSON array: ${cleanText(content).slice(0, 280)}`);
  }

  const byIdx = new Map();
  for (const row of rows) {
    const idx = Number(row?.idx);
    if (!Number.isFinite(idx)) continue;
    const vector = normalizeVectorCandidate(row);
    if (!vector) continue;
    byIdx.set(idx, vector);
  }

  return byIdx;
};

const callMistralSingle = async (item) => {
  if (!mistralApiKey) {
    throw new Error("MISTRAL_API_KEY is required for Mistral teacher inference mode.");
  }

  const response = await fetch(`${mistralBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${mistralApiKey}`
    },
    body: JSON.stringify({
      model: mistralTeacherModel,
      temperature: teacherTemperature,
      max_tokens: 160,
      messages: [
        {
          role: "system",
          content: "Return strict JSON only with vector keys: energy,warmth,brightness,acousticness,complexity,nostalgia"
        },
        {
          role: "user",
          content: [
            "Estimate hidden params from this rule prompt.",
            "Scale: integer 0..100.",
            "Output exactly JSON object with key vector:",
            '{"vector":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0}}',
            item.rule_prompt
          ].join("\n")
        }
      ]
    })
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`Mistral single call failed: HTTP ${response.status} ${JSON.stringify(payload)}`);
  }

  const content = extractMessageContent(payload);
  const vector = normalizeVectorCandidate(JSON.parse(content));
  if (!vector) {
    throw new Error(`Mistral single output has no vector: ${cleanText(content).slice(0, 220)}`);
  }

  return vector;
};

const inferVectorsWithTeacher = async (records) => {
  const out = [];

  for (let i = 0; i < records.length; i += teacherBatchSize) {
    const batch = records.slice(i, i + teacherBatchSize);
    let byIdx = null;
    let lastError = null;

    for (let attempt = 0; attempt <= teacherRetries; attempt += 1) {
      try {
        byIdx = await callMistralBatch(batch);
        break;
      } catch (error) {
        lastError = error;
        if (attempt < teacherRetries) {
          await wait(300 * (attempt + 1));
        }
      }
    }

    if (!byIdx) {
      if (!teacherAllowFallback) {
        throw new Error(
          `Teacher batch failed at ${i}-${i + batch.length - 1}: ${lastError instanceof Error ? lastError.message : String(lastError)}`
        );
      }

      for (const row of batch) {
        const vector = buildFallbackVector(row);
        out.push({ ...row, inferred_vector: vector, teacher_generated: false, weather: weatherFromVector(vector) });
      }
      continue;
    }

    for (const row of batch) {
      const idx = row.meta.index;
      const fromBatch = byIdx.get(idx);
      if (fromBatch) {
        out.push({ ...row, inferred_vector: fromBatch, teacher_generated: true, weather: weatherFromVector(fromBatch) });
        continue;
      }

      let rescued = null;
      for (let attempt = 0; attempt <= teacherRetries; attempt += 1) {
        try {
          rescued = await callMistralSingle(row);
          break;
        } catch {
          if (attempt < teacherRetries) {
            await wait(220 * (attempt + 1));
          }
        }
      }

      if (rescued) {
        out.push({ ...row, inferred_vector: rescued, teacher_generated: true, weather: weatherFromVector(rescued) });
        continue;
      }

      if (!teacherAllowFallback) {
        throw new Error(`Missing teacher vector at index=${idx}`);
      }

      const fallbackVector = buildFallbackVector(row);
      out.push({ ...row, inferred_vector: fallbackVector, teacher_generated: false, weather: weatherFromVector(fallbackVector) });
    }

    console.log(`Teacher inference progress: ${Math.min(i + teacherBatchSize, records.length)}/${records.length}`);
    await wait(120);
  }

  return out;
};

const sampleSelectedParts = () => ({
  style: pick(STYLE_PARTS).id,
  instrument: pick(INSTRUMENT_PARTS).id,
  mood: pick(MOOD_PARTS).id,
  gimmick: pick(GIMMICK_PARTS).id
});

const makePlayerRequestText = () => {
  if (teacherLang === "ja") {
    const scene = pick(REQUEST_SCENES_JA);
    const feel = pick(REQUEST_FEELS_JA);
    const ending = pick(REQUEST_OPENERS_JA);
    return `${scene}に合う、${feel}、${ending}。`;
  }

  const opener = pick(REQUEST_OPENERS_EN);
  const scene = pick(REQUEST_SCENES_EN);
  const feel = pick(REQUEST_FEELS_EN);
  const flow = pick(REQUEST_FLOWS_EN);
  return `${opener} music for ${scene}, ${feel}, ${flow}.`;
};

const buildSeedRecords = () => {
  const records = [];
  const seenCombo = new Set();
  const maxUnique = STYLE_PARTS.length * INSTRUMENT_PARTS.length * MOOD_PARTS.length * GIMMICK_PARTS.length;

  let index = 0;
  while (records.length < count) {
    const selected = sampleSelectedParts();
    const comboKey = `${selected.style}|${selected.instrument}|${selected.mood}|${selected.gimmick}`;
    if (seenCombo.has(comboKey) && seenCombo.size < maxUnique) {
      continue;
    }
    seenCombo.add(comboKey);

    const requestText = makePlayerRequestText();
    const rulePrompt = buildRulePrompt(selected);

    records.push({
      id: `teacher_${String(records.length + 1).padStart(6, "0")}`,
      request_text: requestText,
      rule_prompt: rulePrompt,
      selected_parts_by_slot: selected,
      meta: {
        seed,
        index,
        teacher_model: mistralTeacherModel,
        generation_source: useMistralTeacher ? "mistral_large3_rule_prompt_infer" : "synthetic_rule_prompt_infer",
        input_format: "rule_prompt_api"
      }
    });

    index += 1;
  }

  return records;
};

const buildConstraints = (selectedPartsBySlot, vector) => {
  const style = PART_BY_ID.get(selectedPartsBySlot.style);
  const gimmick = PART_BY_ID.get(selectedPartsBySlot.gimmick);

  const preferredStyleTags = style?.styleTag ? [style.styleTag] : [];
  const preferredGimmickTags = gimmick?.gimmickTag ? [gimmick.gimmickTag] : [];
  const avoidPartIds = vector.brightness < 20 ? ["style_2000s_pop"] : [];

  return {
    preferredStyleTags,
    preferredGimmickTags,
    avoidPartIds
  };
};

const main = async () => {
  if (useMistralTeacher && !mistralApiKey && !teacherAllowFallback) {
    throw new Error(
      "MISTRAL_API_KEY is missing. Set it, or set TEACHER_ALLOW_FALLBACK=true if you explicitly want fallback vectors."
    );
  }

  const seedRecords = buildSeedRecords();
  const withVectors = useMistralTeacher
    ? await inferVectorsWithTeacher(seedRecords)
    : seedRecords.map((row) => {
        const vector = buildFallbackVector(row);
        return {
          ...row,
          inferred_vector: vector,
          teacher_generated: false,
          weather: weatherFromVector(vector)
        };
      });

  const outputRows = withVectors.map((row) => {
    const vector = toVector(row.inferred_vector);
    return {
      id: row.id,
      request_text: row.request_text,
      rule_prompt: row.rule_prompt,
      weather: row.weather,
      selected_parts_by_slot: row.selected_parts_by_slot,
      target_hidden_params: {
        vector,
        tags: makeTags(vector),
        constraints: buildConstraints(row.selected_parts_by_slot, vector)
      },
      meta: {
        ...row.meta,
        teacher_generated: row.teacher_generated,
        generated_at: new Date().toISOString()
      }
    };
  });

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${outputRows.map((row) => JSON.stringify(row)).join("\n")}\n`);
  console.log(`Generated ${outputRows.length} teacher pairs -> ${outputPath}`);
  console.log(
    JSON.stringify(
      {
        mode: generationMode,
        teacher_model: mistralTeacherModel,
        teacher_generated_count: outputRows.filter((row) => row.meta.teacher_generated).length,
        fallback_count: outputRows.filter((row) => !row.meta.teacher_generated).length
      },
      null,
      2
    )
  );
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
