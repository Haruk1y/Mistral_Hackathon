import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const cycleId = process.env.LOOP_CYCLE_ID || "cycle_1";
const cycleDir = resolve(root, `artifacts/loop/${cycleId}`);
const snapshotPath = resolve(cycleDir, "mcp_eval_snapshot.json");
const decisionPath = resolve(cycleDir, "mcp_decision_input.json");

const asBool = (value, fallback = false) => {
  if (value == null || value === "") return fallback;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
};

const tryMcpProbe = async () => {
  const enabled = asBool(process.env.WANDB_MCP_ENABLED, false);
  const baseUrl = process.env.WANDB_MCP_BASE_URL || "https://mcp.withwandb.com/mcp";
  const apiKey = process.env.WANDB_API_KEY || "";
  if (!enabled || !apiKey) {
    return {
      enabled,
      ok: false,
      reason: enabled ? "WANDB_API_KEY missing" : "WANDB_MCP_ENABLED=false",
      base_url: baseUrl
    };
  }

  try {
    const response = await fetch(baseUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream"
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: `probe_${Date.now()}`,
        method: "tools/list",
        params: {}
      })
    });

    const text = await response.text();
    return {
      enabled: true,
      ok: response.ok,
      status: response.status,
      base_url: baseUrl,
      response_preview: text.slice(0, 300)
    };
  } catch (error) {
    return {
      enabled: true,
      ok: false,
      base_url: baseUrl,
      reason: error instanceof Error ? error.message : String(error)
    };
  }
};

const runFallbackFetcher = () => {
  const scriptPath = resolve(root, "scripts/wandb/fallback_wandb_api_fetch.py");
  const env = {
    ...process.env,
    UV_CACHE_DIR: process.env.UV_CACHE_DIR || resolve(root, ".uv-cache"),
    WANDB_MCP_SNAPSHOT_PATH: snapshotPath,
    WANDB_MCP_TOP_K: process.env.WANDB_MCP_TOP_K || "20"
  };

  const run = spawnSync("uv", ["run", scriptPath], {
    cwd: root,
    env,
    encoding: "utf8"
  });

  if (run.stdout) process.stdout.write(run.stdout);
  if (run.stderr) process.stderr.write(run.stderr);
  if (run.status !== 0) {
    throw new Error(`uv run fallback_wandb_api_fetch.py failed with code ${run.status ?? 1}`);
  }
};

const buildDecisionInput = (snapshot) => {
  const evalSummary = snapshot?.eval_summary ?? {};
  const failures = Array.isArray(snapshot?.failures_top_k) ? snapshot.failures_top_k : [];
  const recentRuns = Array.isArray(snapshot?.recent_runs) ? snapshot.recent_runs : [];
  return {
    generated_at: new Date().toISOString(),
    cycle_id: cycleId,
    source: snapshot?.source ?? "unknown",
    eval_summary: {
      latest_by_mode: evalSummary.latest_by_mode ?? {},
      auto_improvement_delta: evalSummary.auto_improvement_delta ?? {},
      loop_completion_rate: evalSummary.loop_completion_rate ?? 0
    },
    failures_top_k: failures.slice(0, Number(process.env.WANDB_MCP_TOP_K || 20)),
    recent_runs: recentRuns.slice(0, 10)
  };
};

const main = async () => {
  await mkdir(cycleDir, { recursive: true });

  const probe = await tryMcpProbe();
  runFallbackFetcher();

  const snapshot = JSON.parse(await readFile(snapshotPath, "utf8"));
  const merged = {
    ...snapshot,
    mcp_probe: probe,
    source:
      probe.ok && snapshot.source === "wandb_api_fallback"
        ? "wandb_mcp_probe+api_fallback"
        : snapshot.source,
    fetched_at: new Date().toISOString()
  };

  const decisionInput = buildDecisionInput(merged);

  await writeFile(snapshotPath, JSON.stringify(merged, null, 2));
  await writeFile(decisionPath, JSON.stringify(decisionInput, null, 2));

  console.log(
    JSON.stringify(
      {
        cycle_id: cycleId,
        snapshot_path: snapshotPath,
        decision_input_path: decisionPath,
        source: merged.source,
        mcp_probe_ok: Boolean(probe.ok)
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
