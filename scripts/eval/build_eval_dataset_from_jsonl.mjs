import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const inputPath = resolve(root, process.env.EVAL_INPUT_JSONL || "data/ft/ft_request_param_test.jsonl");
const outputPath = resolve(root, process.env.EVAL_OUTPUT_JSON || "data/eval/test_eval_set.from_ft_test.v1.json");

const inferWeather = (text) => {
  const lower = String(text || "").toLowerCase();
  if (/(rain|storm|drizzle|wet)/u.test(lower)) return "rainy";
  if (/(night|evening|dusk|cloud|fog)/u.test(lower)) return "cloudy";
  if (/(sun|morning|bright|daylight)/u.test(lower)) return "sunny";
  return "cloudy";
};

const parseJsonl = (raw) =>
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));

const toItems = (rows) =>
  rows.map((row, idx) => {
    const requestText = String(row?.request_text || "");
    const targetVector = row?.target_hidden_params?.vector || {};
    const targetScale = Number(row?.target_hidden_params?.target_scale || row?.target_scale || 10);
    return {
      id: String(row?.id || `ft_test_${String(idx + 1).padStart(4, "0")}`),
      scenario: String(row?.source_type || "ft_test"),
      source_type: String(row?.source_type || "unknown"),
      weather: String(row?.weather || inferWeather(requestText)),
      request_text: requestText,
      target_scale: Number.isFinite(targetScale) ? targetScale : 10,
      target_hidden_params: {
        vector: targetVector,
      },
    };
  });

const main = async () => {
  const raw = await readFile(inputPath, "utf8");
  const rows = parseJsonl(raw);
  if (rows.length === 0) {
    throw new Error(`No rows found in ${inputPath}`);
  }

  const payload = {
    version: "ft_test_split_v1",
    generated_at: new Date().toISOString(),
    source_path: inputPath,
    count: rows.length,
    items: toItems(rows),
  };

  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(payload, null, 2)}\n`);
  console.log(`Wrote ${outputPath}`);
  console.log(`Rows: ${rows.length}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
