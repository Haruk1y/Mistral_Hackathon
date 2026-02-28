import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const runsDir = resolve(root, "artifacts/eval/runs");
const samplesDir = resolve(root, "artifacts/eval/samples");
const summaryDir = resolve(root, "artifacts/eval/summary");
const loopDir = resolve(root, "artifacts/loop");

const readJson = async (path) => JSON.parse(await readFile(path, "utf8"));

const byModeLatest = (rows) => {
  const map = new Map();
  for (const row of rows) {
    const existing = map.get(row.mode);
    if (!existing || row.timestamp > existing.timestamp) {
      map.set(row.mode, row);
    }
  }
  return Object.fromEntries([...map.entries()]);
};

const computeLoopCompletionRate = async () => {
  try {
    const dirs = await readdir(loopDir, { withFileTypes: true });
    const cycleDirs = dirs.filter((entry) => entry.isDirectory() && /^cycle_\d+$/u.test(entry.name));
    if (cycleDirs.length === 0) {
      return {
        completed: 0,
        planned: Number(process.env.LOOP_PLANNED_CYCLES || 2),
        rate: 0
      };
    }

    let completed = 0;
    for (const cycle of cycleDirs) {
      const summaryPath = resolve(loopDir, cycle.name, "summary.json");
      try {
        const summary = await readJson(summaryPath);
        if (summary.completed === true) {
          completed += 1;
        }
      } catch {
        // noop
      }
    }

    const planned = Math.max(Number(process.env.LOOP_PLANNED_CYCLES || 2), cycleDirs.length);
    return {
      completed,
      planned,
      rate: planned === 0 ? 0 : completed / planned
    };
  } catch {
    return {
      completed: 0,
      planned: Number(process.env.LOOP_PLANNED_CYCLES || 2),
      rate: 0
    };
  }
};

const latestSamplesByMode = async (latestByMode) => {
  const out = {};
  for (const [mode, row] of Object.entries(latestByMode)) {
    const runFile = row?.file;
    if (!runFile) continue;
    const runId = runFile.replace(/\.json$/u, "");
    const samplePath = resolve(samplesDir, `${runId}.json`);
    try {
      const payload = await readJson(samplePath);
      const failures = Array.isArray(payload.failures_top_k) ? payload.failures_top_k : [];
      out[mode] = {
        sample_file: `${runId}.json`,
        failure_count: failures.length,
        top_trace_urls: failures
          .map((item) => item?.trace_url)
          .filter((value) => typeof value === "string" && value.length > 0)
          .slice(0, 5)
      };
    } catch {
      out[mode] = {
        sample_file: `${runId}.json`,
        failure_count: 0,
        top_trace_urls: []
      };
    }
  }
  return out;
};

const main = async () => {
  const files = await readdir(runsDir).catch(() => []);
  const runFiles = files.filter((name) => name.endsWith(".json")).sort();
  if (runFiles.length === 0) {
    throw new Error(`No run files found in ${runsDir}`);
  }

  const rows = [];
  for (const file of runFiles) {
    const fullPath = resolve(runsDir, file);
    const payload = await readJson(fullPath);
    rows.push({ ...payload, file });
  }

  const latestByMode = byModeLatest(rows);
  const promptBaseline = latestByMode.prompt_baseline;
  const fineTuned = latestByMode.fine_tuned;

  const autoImprovementDelta =
    promptBaseline && fineTuned
      ? {
          intent_score_mean: fineTuned.metrics.intent_score_mean - promptBaseline.metrics.intent_score_mean,
          vector_mae: fineTuned.metrics.vector_mae - promptBaseline.metrics.vector_mae,
          json_valid_rate: fineTuned.metrics.json_valid_rate - promptBaseline.metrics.json_valid_rate
        }
      : {
          intent_score_mean: 0,
          vector_mae: 0,
          json_valid_rate: 0
        };

  const loop = await computeLoopCompletionRate();
  const sampleIndex = await latestSamplesByMode(latestByMode);

  const summary = {
    generated_at: new Date().toISOString(),
    runs_count: rows.length,
    latest_by_mode: latestByMode,
    latest_samples_by_mode: sampleIndex,
    auto_improvement_delta: autoImprovementDelta,
    loop_completion_rate: loop.rate,
    loop_detail: loop
  };

  await mkdir(summaryDir, { recursive: true });
  const summaryPath = resolve(summaryDir, "latest_summary.json");
  await writeFile(summaryPath, JSON.stringify(summary, null, 2));

  console.log(`Summary saved: ${summaryPath}`);
  console.log(JSON.stringify({
    loop_completion_rate: summary.loop_completion_rate,
    auto_improvement_delta: summary.auto_improvement_delta,
    modes: Object.keys(latestByMode)
  }, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
