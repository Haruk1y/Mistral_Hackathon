import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const inputPath = process.env.FT_TRAIN_ONLY_SOURCE_PATH
  ? resolve(root, process.env.FT_TRAIN_ONLY_SOURCE_PATH)
  : resolve(root, "artifacts/hf_dataset_source/train.jsonl");
const outputDir = resolve(root, "data/ft");
const outputPrefix = process.env.FT_OUTPUT_PREFIX || "ft_request_param";

const trainRatio = Number(process.env.FT_TRAIN_RATIO || "0.8");
const validRatio = Number(process.env.FT_VALID_RATIO || "0.1");
const testRatio = Number(process.env.FT_TEST_RATIO || "0.1");
const splitSeed = Number(process.env.FT_SPLIT_SEED || "42");

const parseJsonl = (raw) =>
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));

const writeJsonl = async (path, rows) => {
  await writeFile(path, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`);
};

const createPrng = (seed) => {
  let state = Number.isFinite(seed) ? (seed >>> 0) : 42;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const shuffled = (rows, seed) => {
  const result = rows.slice();
  const random = createPrng(seed);
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
};

const validateRatios = () => {
  const ratios = [trainRatio, validRatio, testRatio];
  if (ratios.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error(`Split ratios must be finite positive numbers. got=${JSON.stringify({ trainRatio, validRatio, testRatio })}`);
  }
  const sum = trainRatio + validRatio + testRatio;
  if (Math.abs(sum - 1) > 1e-9) {
    throw new Error(`Split ratios must sum to 1. got=${sum}`);
  }
};

const splitRows = (rows) => {
  const total = rows.length;
  const trainCount = Math.floor(total * trainRatio);
  const validCount = Math.floor(total * validRatio);
  const testCount = total - trainCount - validCount;

  return {
    train: rows.slice(0, trainCount),
    valid: rows.slice(trainCount, trainCount + validCount),
    test: rows.slice(trainCount + validCount, trainCount + validCount + testCount)
  };
};

const sourceTypeCounts = (rows) =>
  rows.reduce((acc, row) => {
    const key = String(row?.source_type || "unknown");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

const main = async () => {
  validateRatios();

  const raw = await readFile(inputPath, "utf8");
  const sourceRows = parseJsonl(raw);
  if (sourceRows.length < 10) {
    throw new Error(`Need at least 10 rows to build train/valid/test. current=${sourceRows.length}`);
  }

  const randomized = shuffled(sourceRows, splitSeed);
  const { train, valid, test } = splitRows(randomized);

  await mkdir(outputDir, { recursive: true });
  await writeJsonl(resolve(outputDir, `${outputPrefix}_train.jsonl`), train);
  await writeJsonl(resolve(outputDir, `${outputPrefix}_valid.jsonl`), valid);
  await writeJsonl(resolve(outputDir, `${outputPrefix}_test.jsonl`), test);

  const stats = {
    generated_at: new Date().toISOString(),
    mode: "resplit_from_train_only",
    source_path: inputPath,
    split_seed: splitSeed,
    split_ratio: {
      train: trainRatio,
      valid: validRatio,
      test: testRatio
    },
    split: {
      train: train.length,
      valid: valid.length,
      test: test.length,
      total: train.length + valid.length + test.length,
      source_rows_total: sourceRows.length
    },
    source_type_counts: sourceTypeCounts(sourceRows),
    output_prefix: outputPrefix
  };

  await writeFile(resolve(outputDir, "ft_split_stats.json"), JSON.stringify(stats, null, 2));
  console.log(JSON.stringify(stats, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
