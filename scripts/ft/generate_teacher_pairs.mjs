import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { setTimeout as wait } from "node:timers/promises";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const outputPath = resolve(root, "data/ft/teacher_pairs.raw.jsonl");

const count = Math.max(50, Number(process.env.TEACHER_PAIR_COUNT || 500));
const seed = Number(process.env.TEACHER_SEED || 20260228);

const generationMode = process.env.TEACHER_REQUEST_GENERATOR || "mistral_large";
const useMistralTeacher = generationMode === "mistral_large";
const mistralTeacherModel = process.env.MISTRAL_TEACHER_MODEL || "mistral-large-latest";
const mistralBaseUrl = process.env.MISTRAL_BASE_URL || "https://api.mistral.ai/v1";
const mistralApiKey = process.env.MISTRAL_API_KEY || "";

const teacherLang = process.env.TEACHER_LANG || "ja";
const teacherBatchSize = Math.max(1, Number(process.env.TEACHER_BATCH_SIZE || 20));
const teacherRetries = Math.max(0, Number(process.env.TEACHER_MAX_RETRIES || 2));
const teacherTemperature = Number(process.env.TEACHER_TEMPERATURE || 0.45);
const teacherAllowFallback = process.env.TEACHER_ALLOW_FALLBACK === "true";
const teacherTargetWordsMin = Math.max(6, Number(process.env.TEACHER_TARGET_WORDS_MIN || 9));
const teacherTargetWordsMax = Math.max(teacherTargetWordsMin, Number(process.env.TEACHER_TARGET_WORDS_MAX || 15));
const teacherTargetCharsJaMin = Math.max(12, Number(process.env.TEACHER_TARGET_CHARS_JA_MIN || 24));
const teacherTargetCharsJaMax = Math.max(teacherTargetCharsJaMin, Number(process.env.TEACHER_TARGET_CHARS_JA_MAX || 48));

const sceneWordsJa = [
  "雨上がりの路地",
  "夕暮れの商店街",
  "古い街灯の下",
  "川沿いの遊歩道",
  "夜の駅前",
  "朝の市場",
  "小さな喫茶店",
  "石畳の坂道"
];

const sceneWordsEn = [
  "a rainy back alley",
  "a sunset shopping street",
  "an old town lantern corner",
  "a riverside walk",
  "a quiet station square at night",
  "a morning market",
  "a tiny cafe",
  "a cobblestone hill road"
];

const requestOpenersEn = [
  "I want",
  "I need",
  "Could you make",
  "Can you make",
  "I'd love",
  "Please compose",
  "Please make",
  "Make",
  "Give me"
];

const requestOpenersJa = [
  "〜な曲がほしいです。",
  "〜な雰囲気でお願いします。",
  "〜っぽい曲にしてほしいです。",
  "〜な余韻が残る感じがほしいです。"
];

const lcg = (seedValue) => {
  let value = seedValue >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, Math.round(value)));
const rng = lcg(seed);

const pick = (array) => array[Math.floor(rng() * array.length) % array.length];

const toBand = (value) => {
  if (value <= 20) return "very_low";
  if (value <= 40) return "low";
  if (value <= 60) return "mid";
  if (value <= 80) return "high";
  return "very_high";
};

const makeVector = () => ({
  energy: clamp(rng() * 100, 0, 100),
  warmth: clamp(rng() * 100, 0, 100),
  brightness: clamp(rng() * 100, 0, 100),
  acousticness: clamp(rng() * 100, 0, 100),
  complexity: clamp(rng() * 100, 0, 100),
  nostalgia: clamp(rng() * 100, 0, 100)
});

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
  avoidPartIds: vector.brightness < 20 ? ["style_2000s_pop"] : []
});

const weatherFromVector = (vector) => {
  if (vector.brightness < 35) return "rainy";
  if (vector.brightness < 60) return "cloudy";
  return "sunny";
};

const summarizeItem = (item) => {
  const vector = item.target_hidden_params.vector;
  const bands = Object.entries(vector)
    .map(([k, v]) => `${k}:${toBand(v)}`)
    .join(", ");
  const tags = item.target_hidden_params.tags.join(",");
  const scene = item.scene;
  return `idx=${item.meta.index}; weather=${item.weather}; scene=${scene}; bands=${bands}; tags=${tags}`;
};

const hasLeak = (text) => {
  const leakPattern =
    /\b(?:energy|warmth|brightness|acousticness|complexity|nostalgia|style|instrument|mood|gimmick)\b|[0-9０-９]{1,3}|city\s*pop|hip[-\s]*hop|edm|piano|guitar|violin|cello|flute|clarinet|accordion|mandolin|harp|music box|フィルター|ビートミュート|シティポップ|ヒップホップ|ピアノ|ギター|バイオリン|チェロ|フルート|クラリネット|アコーディオン|マンドリン|ハープ|オルゴール/i;
  return leakPattern.test(text);
};

const cleanText = (value) =>
  String(value || "")
    .replaceAll("\n", " ")
    .replace(/\s+/g, " ")
    .trim();

const wordCountEn = (text) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9'\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean).length;

const openingKeyEn = (text) =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9'\s]/g, " ")
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .join(" ");

const isReasonableLength = (text) => {
  if (teacherLang === "en") {
    const words = wordCountEn(text);
    return words >= teacherTargetWordsMin && words <= teacherTargetWordsMax;
  }
  const chars = text.length;
  return chars >= teacherTargetCharsJaMin && chars <= teacherTargetCharsJaMax;
};

const isValidRequestText = (text) => {
  if (!text || text.length < 10 || hasLeak(text)) return false;
  if (!isReasonableLength(text)) return false;
  return true;
};

const extractJsonArray = (content) => {
  const trimmed = cleanText(content);
  const direct = (() => {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) return parsed;
      return null;
    } catch {
      return null;
    }
  })();
  if (direct) return direct;

  const start = trimmed.indexOf("[");
  const end = trimmed.lastIndexOf("]");
  if (start < 0 || end <= start) return null;

  try {
    const parsed = JSON.parse(trimmed.slice(start, end + 1));
    return Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
};

const buildFallbackRequest = (item) => {
  const scene = item.scene;
  const weather = item.weather;
  if (teacherLang === "en") {
    const opener = pick(requestOpenersEn);
    return `${opener} a short cue for ${scene}, with ${weather} lingering warmth.`;
  }
  const ending = pick(requestOpenersJa);
  return `${weather === "rainy" ? "雨上がり" : weather === "cloudy" ? "曇り空" : "陽だまり"}の${scene}に似合う、${ending}`;
};

const callMistralBatch = async (batch) => {
  if (!mistralApiKey) {
    throw new Error("MISTRAL_API_KEY is required for mistral_large generation mode.");
  }

  const specs = batch.map((item) => summarizeItem(item)).join("\n");

  const system =
    teacherLang === "en"
      ? "You write abstract customer music requests. Return strict JSON array only."
      : "あなたは抽象的な作曲依頼文を作るアシスタントです。厳密なJSON配列のみ返してください。";

  const user =
    teacherLang === "en"
      ? [
          "Create one abstract request sentence for each spec.",
          "Output format: [{\"idx\": number, \"request_text\": string}]",
          "Rules:",
          "- One sentence per item.",
          `- Keep each sentence around ${teacherTargetWordsMin}-${teacherTargetWordsMax} words.`,
          "- Keep it abstract and emotional.",
          `- Start with a natural opener and vary it across items: ${requestOpenersEn.join(", ")}.`,
          "- No numbers and no parameter names.",
          "- Do not mention style/instrument/mood/gimmick names.",
          "- Do not mention concrete instrument names.",
          "Specs:",
          specs
        ].join("\n")
      : [
          "各specに対して抽象的な依頼文を1文ずつ作ってください。",
          "出力形式: [{\"idx\": number, \"request_text\": string}]",
          "制約:",
          "- 各itemは1文のみ。",
          `- 1文はおよそ${teacherTargetCharsJaMin}-${teacherTargetCharsJaMax}文字。`,
          "- 抽象的・情景的にする。",
          "- 数字やパラメータ名を出さない。",
          "- style/instrument/mood/gimmickや具体的な楽器名を出さない。",
          "Specs:",
          specs
        ].join("\n");

  const response = await fetch(`${mistralBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${mistralApiKey}`
    },
    body: JSON.stringify({
      model: mistralTeacherModel,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ],
      temperature: teacherTemperature,
      max_tokens: Math.max(300, batch.length * 90)
    })
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`Mistral teacher call failed: HTTP ${response.status} ${JSON.stringify(payload)}`);
  }

  const content = payload?.choices?.[0]?.message?.content;
  const rows = extractJsonArray(content);
  if (!rows) {
    throw new Error(`Teacher output is not JSON array: ${cleanText(content).slice(0, 240)}`);
  }

  const byIdx = new Map();
  for (const row of rows) {
    const idx = Number(row?.idx);
    const requestText = cleanText(row?.request_text);
    if (!Number.isFinite(idx) || !requestText) continue;
    byIdx.set(idx, requestText);
  }

  return byIdx;
};

const callMistralSingle = async (item, preferredOpening = null) => {
  if (!mistralApiKey) {
    throw new Error("MISTRAL_API_KEY is required for mistral_large generation mode.");
  }

  const system =
    teacherLang === "en"
      ? "You write abstract customer music requests. Return plain text single sentence only."
      : "あなたは抽象的な作曲依頼文を作るアシスタントです。プレーンテキストの1文のみ返してください。";

  const user =
    teacherLang === "en"
      ? [
          "Create one abstract customer request sentence.",
          `Keep it around ${teacherTargetWordsMin}-${teacherTargetWordsMax} words.`,
          preferredOpening ? `Start with: ${preferredOpening}` : `Use one of these openers: ${requestOpenersEn.join(", ")}`,
          "No numbers and no parameter names.",
          "Do not mention style/instrument/mood/gimmick names.",
          "Do not mention concrete instrument names.",
          `Spec: ${summarizeItem(item)}`
        ].join("\n")
      : [
          "抽象的な作曲依頼文を1文だけ作ってください。",
          `文字数はおよそ${teacherTargetCharsJaMin}-${teacherTargetCharsJaMax}文字。`,
          "数字を使わない。",
          "パラメータ名を使わない。",
          "style/instrument/mood/gimmickや具体的な楽器名を出さない。",
          `Spec: ${summarizeItem(item)}`
        ].join("\n");

  const response = await fetch(`${mistralBaseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${mistralApiKey}`
    },
    body: JSON.stringify({
      model: mistralTeacherModel,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ],
      temperature: teacherTemperature,
      max_tokens: 120
    })
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`Mistral single call failed: HTTP ${response.status} ${JSON.stringify(payload)}`);
  }

  const text = cleanText(payload?.choices?.[0]?.message?.content);
  if (!text) {
    throw new Error("Mistral single call returned empty text");
  }
  return text;
};

const rescueSingleItem = async (item, preferredOpening = null) => {
  for (let attempt = 0; attempt <= teacherRetries; attempt += 1) {
    try {
      const text = await callMistralSingle(item, preferredOpening);
      if (isValidRequestText(text)) {
        return text;
      }
    } catch {
      // retry below
    }
    if (attempt < teacherRetries) {
      await wait(250 * (attempt + 1));
    }
  }
  return null;
};

const generateAbstractRequestsWithTeacher = async (items) => {
  const generated = [];

  for (let i = 0; i < items.length; i += teacherBatchSize) {
    const batch = items.slice(i, i + teacherBatchSize);
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
      for (const item of batch) {
        generated.push({ ...item, request_text: buildFallbackRequest(item), teacher_generated: false });
      }
      continue;
    }

    for (const item of batch) {
      const text = byIdx.get(item.meta.index);
      const preferredOpening =
        teacherLang === "en" ? requestOpenersEn[item.meta.index % requestOpenersEn.length] : null;
      if (!isValidRequestText(text)) {
        const rescued = await rescueSingleItem(item, preferredOpening);
        if (rescued) {
          generated.push({ ...item, request_text: rescued, teacher_generated: true });
          continue;
        }
        if (teacherAllowFallback) {
          generated.push({ ...item, request_text: buildFallbackRequest(item), teacher_generated: false });
          continue;
        }
        throw new Error(`Invalid teacher output at index=${item.meta.index}`);
      }
      generated.push({ ...item, request_text: text, teacher_generated: true });
    }

    if (teacherLang === "en") {
      const start = generated.length - batch.length;
      const recent = generated.slice(start);
      const counts = new Map();
      for (const row of recent) {
        const key = openingKeyEn(row.request_text);
        counts.set(key, (counts.get(key) || 0) + 1);
      }

      const maxPerOpening = Math.max(2, Math.ceil(batch.length / 3));
      for (const [key, currentCount] of counts.entries()) {
        if (currentCount <= maxPerOpening) continue;
        let overflow = currentCount - maxPerOpening;
        for (let idx = 0; idx < recent.length && overflow > 0; idx += 1) {
          const row = recent[idx];
          if (openingKeyEn(row.request_text) !== key) continue;
          const preferredOpening = requestOpenersEn[(row.meta.index + 3) % requestOpenersEn.length];
          const rescued = await rescueSingleItem(row, preferredOpening);
          if (rescued) {
            row.request_text = rescued;
            row.teacher_generated = true;
            overflow -= 1;
            continue;
          }
          if (!teacherAllowFallback) {
            throw new Error(`Failed to diversify opening at index=${row.meta.index}`);
          }
          row.request_text = buildFallbackRequest(row);
          row.teacher_generated = false;
          overflow -= 1;
        }
      }
    }

    console.log(`Teacher generation progress: ${Math.min(i + teacherBatchSize, items.length)}/${items.length}`);
    await wait(120);
  }

  return generated;
};

const makeSyntheticAbstractRequest = (item) => {
  const scene = item.scene;
  const weather = item.weather;
  if (teacherLang === "en") {
    const opener = pick(requestOpenersEn);
    return `${opener} a brief tune for ${scene}, with ${weather} afterglow.`;
  }
  const ending = pick(requestOpenersJa);
  return `${weather === "rainy" ? "雨上がり" : weather === "cloudy" ? "曇り空" : "陽だまり"}の${scene}に似合う、${ending}`;
};

const records = [];
for (let i = 0; i < count; i += 1) {
  const vector = makeVector();
  const tags = makeTags(vector);
  const scene = teacherLang === "en" ? pick(sceneWordsEn) : pick(sceneWordsJa);
  const weather = weatherFromVector(vector);

  records.push({
    id: `teacher_${String(i + 1).padStart(6, "0")}`,
    scene,
    weather,
    request_text: "",
    target_hidden_params: {
      vector,
      tags,
      constraints: makeConstraints(vector)
    },
    meta: {
      seed,
      index: i,
      teacher_model: mistralTeacherModel,
      generation_source: useMistralTeacher ? "mistral_large_teacher" : "synthetic_teacher_stub"
    }
  });
}

const main = async () => {
  const finalized = useMistralTeacher
    ? await generateAbstractRequestsWithTeacher(records)
    : records.map((row) => ({ ...row, request_text: makeSyntheticAbstractRequest(row), teacher_generated: false }));

  const outputRows = finalized.map((row) => ({
    id: row.id,
    request_text: row.request_text,
    target_hidden_params: row.target_hidden_params,
    meta: {
      ...row.meta,
      teacher_generated: row.teacher_generated,
      generated_at: new Date().toISOString()
    }
  }));

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
