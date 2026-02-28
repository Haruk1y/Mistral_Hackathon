import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const inputPath = resolve(root, "data/ft/teacher_pairs.raw.jsonl");
const outputPath = resolve(root, "data/ft/teacher_pairs.filtered.jsonl");
const reportPath = resolve(root, "data/ft/teacher_pairs.filtered.report.json");

const linesToObjects = (raw) =>
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));

const hasNumericLeak = (text) => /\b(?:energy|warmth|brightness|acousticness|complexity|nostalgia)\b|\b\d{1,3}\b/i.test(text);

const isLanguageAllowed = (text) => /[a-zA-Zぁ-んァ-ヶ一-龯]/.test(text);

const main = async () => {
  const raw = await readFile(inputPath, "utf8");
  const rows = linesToObjects(raw);

  const seen = new Set();
  const filtered = [];

  let droppedNumericLeak = 0;
  let droppedDuplicate = 0;
  let droppedLength = 0;
  let droppedLanguage = 0;

  for (const row of rows) {
    const text = String(row.request_text || "").trim();

    if (text.length < 16 || text.length > 260) {
      droppedLength += 1;
      continue;
    }

    if (!isLanguageAllowed(text)) {
      droppedLanguage += 1;
      continue;
    }

    if (hasNumericLeak(text)) {
      droppedNumericLeak += 1;
      continue;
    }

    const key = text.toLowerCase();
    if (seen.has(key)) {
      droppedDuplicate += 1;
      continue;
    }

    seen.add(key);
    filtered.push(row);
  }

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${filtered.map((row) => JSON.stringify(row)).join("\n")}\n`);

  const report = {
    input_count: rows.length,
    output_count: filtered.length,
    dropped: {
      numeric_leak: droppedNumericLeak,
      duplicate: droppedDuplicate,
      length: droppedLength,
      language: droppedLanguage
    },
    generated_at: new Date().toISOString()
  };

  await writeFile(reportPath, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
