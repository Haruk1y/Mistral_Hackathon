import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const inputPath = process.env.FT_SOURCE_PATH
  ? resolve(root, process.env.FT_SOURCE_PATH)
  : resolve(root, "data/ft/teacher_pairs.filtered.jsonl");
const outputDir = resolve(root, "data/ft");
const outputPrefix = process.env.FT_OUTPUT_PREFIX || "ft_request_param";

const SYSTEM_PROMPT =
  "You are a request interpreter for Atelier kotone. Return strict JSON only with keys: energy, warmth, brightness, acousticness, complexity, nostalgia.";

const hashString = (value) => {
  let hash = 2166136261;
  const text = String(value || "");
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
};

const pickByHash = (array, hash, offset = 0) => array[(hash + offset) % array.length];

const pickStyleHint = (row) => {
  const preferred = row.target_hidden_params?.constraints?.preferredStyleTags?.[0];
  if (preferred === "citypop_80s") return "80s City Pop";
  if (preferred === "hiphop_90s") return "90s Hip-Hop";
  if (preferred === "pop_2000s") return "2000s Pop";
  return "70s Folk";
};

const pickGimmickHint = (row) => {
  const preferred = row.target_hidden_params?.constraints?.preferredGimmickTags?.[0];
  if (preferred === "beat_mute") return "Beat Mute";
  if (preferred === "filter_rise") return "Filter Rise";
  if (preferred === "harmony_stack") return "Harmony Stack";
  return "Temple Bell";
};

const pickInstrumentHint = (row) => {
  const tags = row.target_hidden_params?.tags || [];
  if (tags.includes("acoustic")) return "Upright Piano";
  if (tags.includes("nostalgic")) return "Vintage Violin";
  if (tags.includes("upbeat")) return "Travel Accordion";
  if (tags.includes("night")) return "Snake Music Box";
  return "Street Guitar";
};

const pickMoodHint = (row) => {
  const tags = row.target_hidden_params?.tags || [];
  if (tags.includes("rain")) return "Rain Ambience";
  if (tags.includes("night")) return "Night Drive";
  if (tags.includes("cozy")) return "Cozy Hearth";
  if (tags.includes("nostalgic")) return "Pocket Memory";
  if (tags.includes("upbeat")) return "Sun Laugh";
  return "Sun Glow";
};

const pickEnergyMoodLine = (row) => {
  const vec = row.target_hidden_params?.vector || {};
  const energy = Number(vec.energy || 0);
  const nostalgia = Number(vec.nostalgia || 0);
  const complexity = Number(vec.complexity || 0);

  if (energy >= 70 && complexity >= 60) return "Keep the momentum lively with layered motion.";
  if (energy <= 30 && nostalgia >= 60) return "Let it stay soft, reflective, and memory-like.";
  if (energy <= 35) return "Keep the pace gentle and uncluttered.";
  if (nostalgia >= 70) return "Leave a warm nostalgic tail after each phrase.";
  return "Balance groove and calm in a compact arrangement.";
};

const buildRulePrompt = (row) => {
  const tags = row.target_hidden_params?.tags || [];
  const style = pickStyleHint(row);
  const instrument = pickInstrumentHint(row);
  const mood = pickMoodHint(row);
  const gimmick = pickGimmickHint(row);
  const flavor = tags.slice(0, 3).join(", ") || "balanced, cozy";
  const hash = hashString(row.id || JSON.stringify(row.target_hidden_params || {}));

  const opener = pickByHash(
    [
      "Create a short instrumental cue for a retro town moment.",
      "I want a compact instrumental piece for a small city scene.",
      "Could you craft an instrumental snippet with a nostalgic street vibe?",
      "Please compose a concise instrumental track for game play."
    ],
    hash,
    0
  );
  const tail = pickByHash(
    [
      "No vocals. Keep it concise and game-friendly.",
      "Instrumental only. Keep it brief and loop-friendly.",
      "No singing. Keep the structure tight and playable in-game.",
      "Instrumental only. Short, clear, and suitable for gameplay."
    ],
    hash,
    3
  );

  return [
    opener,
    `Style: ${style}.`,
    `Instrument: ${instrument}.`,
    `Mood: ${mood}.`,
    `Gimmick: ${gimmick}.`,
    `Flavor tags: ${flavor}.`,
    pickEnergyMoodLine(row),
    tail
  ].join(" ");
};

const toMessagesRow = (row, sourceType, inputText) => ({
  source_type: sourceType,
  request_text: inputText,
  target_hidden_params: row.target_hidden_params,
  messages: [
    {
      role: "system",
      content: SYSTEM_PROMPT
    },
    {
      role: "user",
      content: `request_text=${inputText}`
    },
    {
      role: "assistant",
      content: JSON.stringify(row.target_hidden_params.vector)
    }
  ]
});

const parseJsonl = (raw) =>
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));

const splitRows = (rows) => {
  const total = rows.length;
  const trainCount = Math.floor(total * 0.8);
  const validCount = Math.floor(total * 0.1);
  const testCount = total - trainCount - validCount;

  return {
    train: rows.slice(0, trainCount),
    valid: rows.slice(trainCount, trainCount + validCount),
    test: rows.slice(trainCount + validCount, trainCount + validCount + testCount)
  };
};

const expandRows = (rows) =>
  rows.flatMap((row) => {
    const requestRow = toMessagesRow(row, "request_text", row.request_text);
    const rulePrompt = buildRulePrompt(row);
    const promptRow = toMessagesRow(row, "rule_prompt", rulePrompt);
    return [requestRow, promptRow];
  });

const writeJsonl = async (path, rows) => {
  await writeFile(path, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
};

const statsFor = (rows) => {
  const vecKeys = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];
  const stats = {};

  for (const key of vecKeys) {
    const values = rows.map((row) => row.target_hidden_params.vector[key]);
    const mean = values.reduce((acc, v) => acc + v, 0) / Math.max(1, values.length);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 0;
    stats[key] = { mean, min, max };
  }

  return stats;
};

const main = async () => {
  const raw = await readFile(inputPath, "utf8");
  const sourceRows = parseJsonl(raw);

  if (sourceRows.length < 10) {
    throw new Error(`Need at least 10 rows for train/valid/test split. current=${sourceRows.length}`);
  }

  const { train, valid, test } = splitRows(sourceRows);

  const trainRows = expandRows(train);
  const validRows = expandRows(valid);
  const testRows = expandRows(test);

  await mkdir(outputDir, { recursive: true });
  await writeJsonl(resolve(outputDir, `${outputPrefix}_train.jsonl`), trainRows);
  await writeJsonl(resolve(outputDir, `${outputPrefix}_valid.jsonl`), validRows);
  await writeJsonl(resolve(outputDir, `${outputPrefix}_test.jsonl`), testRows);

  const stats = {
    generated_at: new Date().toISOString(),
    split: {
      train: trainRows.length,
      valid: validRows.length,
      test: testRows.length,
      total: trainRows.length + validRows.length + testRows.length,
      source_rows_total: sourceRows.length
    },
    source_path: inputPath,
    output_prefix: outputPrefix,
    expansion: {
      request_text_rows_per_source: 1,
      rule_prompt_rows_per_source: 1
    },
    source_vector_stats: statsFor(sourceRows)
  };

  await writeFile(resolve(outputDir, "ft_split_stats.json"), JSON.stringify(stats, null, 2));
  console.log(JSON.stringify(stats, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
