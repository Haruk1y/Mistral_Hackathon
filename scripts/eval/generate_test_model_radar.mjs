import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(new URL("../../", import.meta.url).pathname);
const runsDir = resolve(root, "artifacts/eval/runs");
const samplesDir = resolve(root, "artifacts/eval/samples");
const summaryDir = resolve(root, "artifacts/eval/summary");

const DIM_KEYS = ["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"];
const RADAR_AXES = [...DIM_KEYS.map((k) => `${k}_mae`), "vector_mae", "p95_latency_ms"];
const compareModes = String(process.env.EVAL_COMPARE_MODES || "prompt_baseline,large_baseline")
  .split(",")
  .map((x) => x.trim())
  .filter(Boolean);

const readJson = async (path) => JSON.parse(await readFile(path, "utf8"));

const safeNum = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : 0;
};

const round4 = (value) => Math.round(safeNum(value) * 10000) / 10000;

const latestRunByMode = async (mode) => {
  const files = (await readdir(runsDir)).filter((name) => name.endsWith(".json")).sort();
  let latest = null;
  for (const file of files) {
    const payload = await readJson(resolve(runsDir, file));
    if (payload?.mode !== mode) continue;
    if (!latest || String(payload.timestamp || "") > String(latest.payload.timestamp || "")) {
      latest = { file, payload };
    }
  }
  return latest;
};

const perDimMae = (rows, runMetrics = {}) => {
  const totals = Object.fromEntries(DIM_KEYS.map((key) => [key, 0]));
  const counts = Object.fromEntries(DIM_KEYS.map((key) => [key, 0]));

  for (const row of rows) {
    if (row?.json_valid !== true) continue;
    const byDim = row?.abs_error_by_dim;
    if (!byDim || typeof byDim !== "object") continue;
    for (const key of DIM_KEYS) {
      const value = Number(byDim[key]);
      if (!Number.isFinite(value)) continue;
      totals[key] += value;
      counts[key] += 1;
    }
  }

  const out = {};
  const anyMeasured = DIM_KEYS.some((key) => counts[key] > 0);
  if (!anyMeasured) {
    const vectorMae = safeNum(runMetrics?.vector_mae);
    for (const key of DIM_KEYS) {
      const metricKey = `mae_raw_${key}`;
      const metricValue = safeNum(runMetrics?.[metricKey]);
      out[key] = metricValue > 0 ? metricValue : vectorMae;
    }
    return out;
  }
  for (const key of DIM_KEYS) {
    out[key] = counts[key] > 0 ? totals[key] / counts[key] : 0;
  }
  return out;
};

const toCsv = (rows) => {
  const header = [
    "mode",
    "model_id",
    ...DIM_KEYS.map((k) => `${k}_mae`),
    "vector_mae",
    "p95_latency_ms",
    "run_file",
    "sample_file",
    "run_url",
  ];
  const lines = [header.join(",")];
  for (const row of rows) {
    const fields = [
      row.mode,
      row.model_id,
      ...DIM_KEYS.map((k) => round4(row.mae_by_dim[k])),
      round4(row.vector_mae),
      round4(row.p95_latency_ms),
      row.run_file,
      row.sample_file,
      row.run_url || "",
    ];
    lines.push(fields.map((x) => `"${String(x).replaceAll('"', '""')}"`).join(","));
  }
  return `${lines.join("\n")}\n`;
};

const buildRadarHtml = (rows) => {
  const axes = RADAR_AXES;
  const maxima = Object.fromEntries(axes.map((axis) => [axis, 0]));
  for (const row of rows) {
    const raw = {
      ...Object.fromEntries(DIM_KEYS.map((k) => [`${k}_mae`, safeNum(row.mae_by_dim[k])])),
      vector_mae: safeNum(row.vector_mae),
      p95_latency_ms: safeNum(row.p95_latency_ms),
    };
    for (const axis of axes) {
      maxima[axis] = Math.max(maxima[axis], raw[axis]);
    }
  }

  const datasets = rows.map((row, index) => {
    const raw = {
      ...Object.fromEntries(DIM_KEYS.map((k) => [`${k}_mae`, safeNum(row.mae_by_dim[k])])),
      vector_mae: safeNum(row.vector_mae),
      p95_latency_ms: safeNum(row.p95_latency_ms),
    };
    const normalized = axes.map((axis) => {
      const max = maxima[axis] > 0 ? maxima[axis] : 1;
      return Math.round((raw[axis] / max) * 1000) / 10;
    });
    const color = ["#ef4444", "#0ea5e9", "#22c55e", "#f59e0b"][index % 4];
    return {
      label: `${row.mode} (${row.model_id})`,
      data: normalized,
      fill: true,
      borderColor: color,
      backgroundColor: `${color}55`,
      pointBackgroundColor: color,
    };
  });

  const rawTableRows = rows
    .map((row) => {
      const cells = [
        row.mode,
        row.model_id,
        ...DIM_KEYS.map((k) => round4(row.mae_by_dim[k])),
        round4(row.vector_mae),
        round4(row.p95_latency_ms),
      ];
      return `<tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`;
    })
    .join("\n");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Test Model Radar</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111827; }
    h1 { margin: 0 0 12px; }
    .muted { color: #4b5563; margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: minmax(320px, 720px); gap: 24px; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; }
    th { background: #f9fafb; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>Test Split Model Comparison</h1>
  <div class="muted">Radar values are normalized per-axis (worst value in compared models = 100).</div>
  <div class="grid">
    <canvas id="radarChart" width="720" height="540"></canvas>
    <table>
      <thead>
        <tr>
          <th>mode</th>
          <th>model_id</th>
          <th>energy_mae</th>
          <th>warmth_mae</th>
          <th>brightness_mae</th>
          <th>acousticness_mae</th>
          <th>complexity_mae</th>
          <th>nostalgia_mae</th>
          <th>vector_mae</th>
          <th>p95_latency_ms</th>
        </tr>
      </thead>
      <tbody>
        ${rawTableRows}
      </tbody>
    </table>
  </div>
  <script>
    const labels = ${JSON.stringify(axes)};
    const datasets = ${JSON.stringify(datasets)};
    const ctx = document.getElementById('radarChart');
    new Chart(ctx, {
      type: 'radar',
      data: { labels, datasets },
      options: {
        responsive: false,
        maintainAspectRatio: false,
        scales: {
          r: {
            min: 0,
            max: 100,
            ticks: { stepSize: 20 },
            pointLabels: { font: { size: 12 } }
          }
        },
        plugins: {
          legend: { position: 'bottom' },
          tooltip: { mode: 'nearest' }
        }
      }
    });
  </script>
</body>
</html>
`;
};

const main = async () => {
  const rows = [];
  for (const mode of compareModes) {
    const found = await latestRunByMode(mode);
    if (!found) continue;
    const runFile = found.file;
    const runPayload = found.payload;
    const sampleFile = runFile;
    const samplePath = resolve(samplesDir, sampleFile);
    const samplePayload = await readJson(samplePath);
    const sampleRows = Array.isArray(samplePayload?.rows) ? samplePayload.rows : [];
    rows.push({
      mode,
      model_id: runPayload.model_id || "unknown",
      run_file: runFile,
      sample_file: sampleFile,
      run_url: runPayload.wandb_run_url || "",
      mae_by_dim: perDimMae(sampleRows, runPayload?.metrics || {}),
      vector_mae: safeNum(runPayload?.metrics?.vector_mae),
      p95_latency_ms: safeNum(runPayload?.metrics?.p95_inference_latency_ms),
    });
  }

  if (rows.length === 0) {
    throw new Error(`No run data found for modes: ${compareModes.join(", ")}`);
  }

  await mkdir(summaryDir, { recursive: true });
  const jsonPath = resolve(summaryDir, "test_model_comparison.json");
  const csvPath = resolve(summaryDir, "test_model_comparison.csv");
  const htmlPath = resolve(summaryDir, "test_model_radar.html");

  await writeFile(
    jsonPath,
    `${JSON.stringify({ generated_at: new Date().toISOString(), compare_modes: compareModes, rows }, null, 2)}\n`,
  );
  await writeFile(csvPath, toCsv(rows));
  await writeFile(htmlPath, buildRadarHtml(rows));

  console.log(`Wrote ${jsonPath}`);
  console.log(`Wrote ${csvPath}`);
  console.log(`Wrote ${htmlPath}`);
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
