import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const inputPath = process.env.FT_SOURCE_PATH
  ? resolve(root, process.env.FT_SOURCE_PATH)
  : resolve(root, "data/ft/teacher_pairs.filtered.jsonl");
const outputDir = resolve(root, "data/ft");
const outputPrefix = process.env.FT_OUTPUT_PREFIX || "ft_request_param";
const VECTOR_KEYS = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];
const SLOT_KEYS = ["style", "instrument", "mood", "gimmick"];
const PART_NAME_BY_ID = {
  style_80s_citypop: "80s City Pop",
  style_90s_hiphop: "90s Hip-Hop",
  style_2000s_pop: "2000s Pop",
  inst_piano_upright: "Upright Piano",
  inst_soft_strings: "Fairy Harp",
  inst_analog_synth: "Snake Music Box",
  mood_rain_ambience: "Rain Ambience",
  mood_sun_glow: "Sun Glow",
  mood_night_drive: "Night Drive",
  gimmick_beat_mute: "Beat Mute",
  gimmick_filter_rise: "Filter Rise",
  gimmick_harmony_stack: "Harmony Stack"
};
const STYLE_ID_BY_TAG = {
  citypop_80s: "style_80s_citypop",
  hiphop_90s: "style_90s_hiphop",
  pop_2000s: "style_2000s_pop"
};
const GIMMICK_ID_BY_TAG = {
  beat_mute: "gimmick_beat_mute",
  filter_rise: "gimmick_filter_rise",
  harmony_stack: "gimmick_harmony_stack"
};
const targetScaleEnv = Number(process.env.FT_TARGET_SCALE || process.env.HF_FT_TARGET_SCALE || "10");
if (targetScaleEnv !== 10) {
  throw new Error(`FT_TARGET_SCALE must be 10 for this pipeline. got=${targetScaleEnv}`);
}
const TARGET_SCALE = 10;
const INCLUDE_REQUEST_TEXT_ROWS = process.env.FT_INCLUDE_REQUEST_TEXT_ROWS === "true";

const SYSTEM_PROMPT =
  "You are a request interpreter for Atelier kotone. Return strict JSON only with keys: energy, warmth, brightness, acousticness, complexity, nostalgia.";

const toBoundedInt = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.max(0, Math.min(100, Math.round(numeric)));
};

const toTargetScaleInt = (value) => {
  const bounded100 = toBoundedInt(value);
  return Math.max(0, Math.min(TARGET_SCALE, Math.round((bounded100 / 100) * TARGET_SCALE)));
};

const normalizeHiddenParams = (hiddenParams) => {
  const source = hiddenParams || {};
  const vector = source.vector || {};
  const normalizedVector = Object.fromEntries(VECTOR_KEYS.map((key) => [key, toTargetScaleInt(vector[key])]));

  return {
    vector: normalizedVector
  };
};

const getPartName = (partId) => PART_NAME_BY_ID[partId] || String(partId);

const readKotoneSelection = (row) => {
  const selectedPartsBySlot = row.selected_parts_by_slot;
  if (selectedPartsBySlot && typeof selectedPartsBySlot === "object") {
    const style = String(selectedPartsBySlot.style || "");
    const instrument = String(selectedPartsBySlot.instrument || "");
    const mood = String(selectedPartsBySlot.mood || "");
    const gimmick = String(selectedPartsBySlot.gimmick || "");
    if (style && instrument && mood && gimmick) {
      return {
        selectedPartsBySlot: {
          style,
          instrument,
          mood,
          gimmick
        }
      };
    }
  }

  const vector = row.target_hidden_params?.vector || {};
  const constraints = row.target_hidden_params?.constraints || {};
  const styleTag = Array.isArray(constraints.preferredStyleTags) ? String(constraints.preferredStyleTags[0] || "") : "";
  const gimmickTag = Array.isArray(constraints.preferredGimmickTags) ? String(constraints.preferredGimmickTags[0] || "") : "";

  const style =
    STYLE_ID_BY_TAG[styleTag] || (toBoundedInt(vector.brightness) > 68 ? "style_2000s_pop" : toBoundedInt(vector.nostalgia) > 65 ? "style_80s_citypop" : "style_90s_hiphop");
  const instrument =
    toBoundedInt(vector.acousticness) > 70
      ? "inst_piano_upright"
      : toBoundedInt(vector.warmth) > 64
        ? "inst_soft_strings"
        : "inst_analog_synth";
  const mood =
    toBoundedInt(vector.brightness) < 35
      ? "mood_rain_ambience"
      : toBoundedInt(vector.energy) > 62
        ? "mood_sun_glow"
        : "mood_night_drive";
  const gimmick =
    GIMMICK_ID_BY_TAG[gimmickTag] || (toBoundedInt(vector.complexity) > 55 ? "gimmick_harmony_stack" : toBoundedInt(vector.energy) > 62 ? "gimmick_filter_rise" : "gimmick_beat_mute");

  return {
    selectedPartsBySlot: {
      style,
      instrument,
      mood,
      gimmick
    }
  };
};

const buildRulePrompt = (row) => {
  const existingRulePrompt = String(row.rule_prompt || "").trim();
  if (existingRulePrompt) {
    const originalLines = existingRulePrompt
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    const selectedHeaderIndex = originalLines.findIndex((line) => line === "Selected Kotone combination:");
    if (selectedHeaderIndex >= 0) {
      const selectedSection = originalLines.slice(selectedHeaderIndex);
      const hasStyleTail = selectedSection.some((line) =>
        line.startsWith("Style: warm, cozy, handcrafted, street evening, non-vocal, emotional but simple.")
      );
      const lines = [
        "Compose nostalgic retro pixel-town background music.",
        "Return instrumental music suitable for a game scene.",
        "This is a rule-based prompt generated from selected Kotone parts.",
        ...selectedSection
      ];
      if (!hasStyleTail) {
        lines.push("Style: warm, cozy, handcrafted, street evening, non-vocal, emotional but simple.");
      }
      return lines.join("\n");
    }
  }

  const selected = readKotoneSelection(row);
  const lines = [
    "Compose nostalgic retro pixel-town background music.",
    "Return instrumental music suitable for a game scene.",
    "This is a rule-based prompt generated from selected Kotone parts.",
    "Selected Kotone combination:",
    ...SLOT_KEYS.map((slot) => {
      const partId = selected.selectedPartsBySlot[slot];
      return `- ${slot}: ${getPartName(partId)} (${partId})`;
    }),
    "Style: warm, cozy, handcrafted, street evening, non-vocal, emotional but simple."
  ];
  return lines.join("\n");
};

const toMessagesRow = (row, sourceType, inputText) => {
  const targetHiddenParams = normalizeHiddenParams(row.target_hidden_params);
  return {
    source_type: sourceType,
    request_text: inputText,
    target_hidden_params: targetHiddenParams,
    messages: [
      {
        role: "system",
        content: SYSTEM_PROMPT
      },
      {
        role: "user",
        content: inputText
      },
      {
        role: "assistant",
        content: JSON.stringify(targetHiddenParams.vector)
      }
    ]
  };
};

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
    const rulePrompt = buildRulePrompt(row);
    const promptRow = toMessagesRow(row, "rule_prompt", rulePrompt);
    if (!INCLUDE_REQUEST_TEXT_ROWS) return [promptRow];
    const requestRow = toMessagesRow(row, "request_text", row.request_text);
    return [requestRow, promptRow];
  });

const writeJsonl = async (path, rows) => {
  await writeFile(path, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
};

const statsFor = (rows) => {
  const stats = {};

  for (const key of VECTOR_KEYS) {
    const values = rows.map((row) => toBoundedInt(row?.target_hidden_params?.vector?.[key]));
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
      request_text_rows_per_source: INCLUDE_REQUEST_TEXT_ROWS ? 1 : 0,
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
