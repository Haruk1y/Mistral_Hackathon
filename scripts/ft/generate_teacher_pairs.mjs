import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const outputPath = resolve(root, "data/ft/teacher_pairs.raw.jsonl");

const count = Math.max(50, Number(process.env.TEACHER_PAIR_COUNT || 500));
const seed = Number(process.env.TEACHER_SEED || 20260228);

const toneWords = ["quiet", "warm", "nostalgic", "bright", "gentle", "urban", "cozy", "hopeful", "calm"];
const sceneWords = ["rainy evening", "old town", "market street", "sunset", "night drive", "small cafe"];
const tagsPool = ["nostalgic", "acoustic", "cozy", "rain", "upbeat", "citypop_80s", "night"];

const lcg = (seedValue) => {
  let value = seedValue >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) % 4294967296;
    return value / 4294967296;
  };
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, Math.round(value)));

const rng = lcg(seed);

const makeVector = () => ({
  energy: clamp(rng() * 100, 0, 100),
  warmth: clamp(rng() * 100, 0, 100),
  brightness: clamp(rng() * 100, 0, 100),
  acousticness: clamp(rng() * 100, 0, 100),
  complexity: clamp(rng() * 100, 0, 100),
  nostalgia: clamp(rng() * 100, 0, 100)
});

const pick = (array) => array[Math.floor(rng() * array.length) % array.length];

const makeTags = (vector) => {
  const tags = [];
  if (vector.nostalgia > 65) tags.push("nostalgic");
  if (vector.acousticness > 60) tags.push("acoustic");
  if (vector.warmth > 60) tags.push("cozy");
  if (vector.energy > 60) tags.push("upbeat");
  if (vector.brightness < 35) tags.push("rain");
  if (!tags.length) tags.push(pick(tagsPool));
  return [...new Set(tags)];
};

const makeConstraints = (vector) => ({
  preferredStyleTags: vector.nostalgia > 60 ? ["citypop_80s"] : vector.energy > 60 ? ["pop_2000s"] : ["hiphop_90s"],
  preferredGimmickTags: vector.energy > 58 ? ["filter_rise"] : ["beat_mute"],
  avoidPartIds: vector.brightness < 20 ? ["style_2000s_pop"] : []
});

const makeRequestText = (vector) => {
  const tone = pick(toneWords);
  const scene = pick(sceneWords);
  const phrases = [];

  if (vector.energy > 65) phrases.push("with a bit more drive");
  if (vector.nostalgia > 65) phrases.push("that feels like an old memory");
  if (vector.acousticness > 70) phrases.push("with organic acoustic color");
  if (vector.brightness < 35) phrases.push("in a subdued, darker tone");
  if (vector.warmth > 70) phrases.push("and warm emotional texture");
  if (!phrases.length) phrases.push("with balanced feeling");

  return `Please compose a ${tone} piece for ${scene}, ${phrases.join(", ")}.`;
};

const records = [];
for (let i = 0; i < count; i += 1) {
  const vector = makeVector();
  const tags = makeTags(vector);

  records.push({
    id: `teacher_${String(i + 1).padStart(6, "0")}`,
    request_text: makeRequestText(vector),
    target_hidden_params: {
      vector,
      tags,
      constraints: makeConstraints(vector)
    },
    meta: {
      seed,
      index: i,
      teacher_model: process.env.TEACHER_MODEL_ID || "mistral-large-latest",
      generation_source: "synthetic_teacher_stub"
    }
  });
}

const main = async () => {
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${records.map((row) => JSON.stringify(row)).join("\n")}\n`);
  console.log(`Generated ${records.length} teacher pairs -> ${outputPath}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
