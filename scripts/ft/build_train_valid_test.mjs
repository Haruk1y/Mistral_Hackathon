import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const inputPath = resolve(root, "data/ft/teacher_pairs.filtered.jsonl");
const outputDir = resolve(root, "data/ft");

const toMessagesRow = (row) => ({
  messages: [
    {
      role: "system",
      content:
        "You are a request interpreter for Atelier kotone. Return strict JSON only with keys: energy, warmth, brightness, acousticness, complexity, nostalgia."
    },
    {
      role: "user",
      content: `request_text=${row.request_text}`
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

  const trainRows = train.map(toMessagesRow);
  const validRows = valid.map(toMessagesRow);
  const testRows = test.map(toMessagesRow);

  await mkdir(outputDir, { recursive: true });
  await writeJsonl(resolve(outputDir, "ft_request_param_train.jsonl"), trainRows);
  await writeJsonl(resolve(outputDir, "ft_request_param_valid.jsonl"), validRows);
  await writeJsonl(resolve(outputDir, "ft_request_param_test.jsonl"), testRows);

  const stats = {
    generated_at: new Date().toISOString(),
    split: {
      train: trainRows.length,
      valid: validRows.length,
      test: testRows.length,
      total: sourceRows.length
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
