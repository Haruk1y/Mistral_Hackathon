import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const cycleId = process.env.LOOP_CYCLE_ID || "cycle_1";
const cycleDir = resolve(root, `artifacts/loop/${cycleId}`);
const snapshotPath = resolve(cycleDir, "mcp_eval_snapshot.json");
const decisionPath = resolve(cycleDir, "mcp_decision_input.json");

const EVAL_METRIC_KEYS = [
  "json_valid_rate",
  "vector_mae",
  "mse_raw",
  "mse_norm",
  "r2_score",
  "constraint_match_rate",
  "slot_exact_match",
  "intent_score_mean",
  "output_sanity_score",
  "p95_inference_latency_ms",
  "p50_inference_latency_ms",
  "cost_per_100_requests_usd",
];

const asBool = (value, fallback = false) => {
  if (value == null || value === "") return fallback;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
};

const asInt = (value, fallback) => {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asNumber = (value, fallback = null) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const mean = (values) => {
  if (!values.length) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
};

const uniq = (values) => {
  const out = [];
  const seen = new Set();
  for (const value of values) {
    if (typeof value !== "string" || value.trim() === "") continue;
    if (seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
};

const splitCsv = (value) =>
  String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

const parseMaybeJson = (value) => {
  if (value == null) return null;
  if (typeof value === "object") return value;
  if (typeof value !== "string") return null;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
};

const unwrapConfigValue = (value) => {
  if (value && typeof value === "object" && "value" in value) {
    return value.value;
  }
  return value;
};

const normalizeConfig = (configRaw) => {
  const parsed = parseMaybeJson(configRaw);
  if (!parsed || typeof parsed !== "object") return {};
  const out = {};
  for (const [key, value] of Object.entries(parsed)) {
    out[key] = unwrapConfigValue(value);
  }
  return out;
};

const extractMetric = (summary, key) => {
  const direct = asNumber(summary?.[`eval/${key}`]);
  if (direct != null) return direct;
  const iter = asNumber(summary?.[`iter_eval/${key}`]);
  if (iter != null) return iter;
  return 0;
};

const normalizeSummary = (summaryRaw) => {
  const parsed = parseMaybeJson(summaryRaw);
  if (!parsed || typeof parsed !== "object") return {};
  return parsed;
};

const parseSseJson = (raw) => {
  const lines = String(raw || "")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("data: "))
    .map((line) => line.slice(6).trim())
    .filter((line) => line.length > 0 && line !== "[DONE]");

  for (let i = lines.length - 1; i >= 0; i -= 1) {
    try {
      return JSON.parse(lines[i]);
    } catch {
      // try previous line
    }
  }

  throw new Error(`Unable to parse MCP response: ${String(raw).slice(0, 500)}`);
};

const toMode = (value) => {
  const raw = String(value || "").trim();
  if (["rule_baseline", "prompt_baseline", "fine_tuned"].includes(raw)) return raw;
  return null;
};

const mcpEnabled = asBool(process.env.WANDB_MCP_ENABLED, true);
const mcpApiKey = process.env.WANDB_API_KEY || "";
const defaultMcpBaseUrl = "https://mcp.withwandb.com/mcp";
const mcpTryLocalHttp = asBool(process.env.WANDB_MCP_TRY_LOCAL_HTTP, true);
const mcpLocalPort = Math.max(1, asInt(process.env.WANDB_MCP_LOCAL_PORT, 8080));
const mcpLocalHttpBaseUrl = `http://127.0.0.1:${mcpLocalPort}/mcp`;
const mcpFallbackUrls = splitCsv(process.env.WANDB_MCP_BASE_URL_FALLBACKS);
const mcpBaseUrls = uniq([
  process.env.WANDB_MCP_BASE_URL,
  ...mcpFallbackUrls,
  defaultMcpBaseUrl,
  ...(mcpTryLocalHttp ? [mcpLocalHttpBaseUrl] : []),
]);
const mcpRetryMax = Math.max(1, asInt(process.env.WANDB_MCP_RETRY_MAX, 3));
const mcpRetryBackoffMs = Math.max(100, asInt(process.env.WANDB_MCP_RETRY_BACKOFF_MS, 700));
let activeMcpBaseUrl = mcpBaseUrls[0] || defaultMcpBaseUrl;
const RETRYABLE_ERROR_CODES = new Set([
  "ENOTFOUND",
  "EAI_AGAIN",
  "ECONNRESET",
  "ETIMEDOUT",
  "ECONNREFUSED",
  "EHOSTUNREACH",
  "ENETUNREACH",
]);

if (!mcpEnabled) {
  throw new Error("WANDB_MCP_ENABLED=false. This script is strict MCP-only and does not support fallback.");
}
if (!mcpApiKey) {
  throw new Error("WANDB_API_KEY is required for MCP-only fetch.");
}
if (mcpBaseUrls.length === 0) {
  throw new Error("No MCP base URL candidates are configured.");
}

const sleep = async (ms) =>
  new Promise((resolve) => {
    setTimeout(resolve, ms);
  });

const toErrorCode = (error) => {
  if (!error || typeof error !== "object") return "";
  const direct = typeof error.code === "string" ? error.code : "";
  if (direct) return direct;
  const cause = error.cause;
  if (cause && typeof cause === "object" && typeof cause.code === "string") {
    return cause.code;
  }
  return "";
};

const isRetryableMcpError = (error, status) => {
  if (status != null) {
    if (status >= 500) return true;
    if (status === 408 || status === 429) return true;
  }
  const code = toErrorCode(error);
  return RETRYABLE_ERROR_CODES.has(code);
};

const mcpRequest = async (method, params, idPrefix) => {
  const errors = [];
  let hadDnsFailure = false;
  for (const baseUrl of mcpBaseUrls) {
    for (let attempt = 1; attempt <= mcpRetryMax; attempt += 1) {
      try {
        const response = await fetch(baseUrl, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${mcpApiKey}`,
            "Content-Type": "application/json",
            Accept: "application/json, text/event-stream",
          },
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: `${idPrefix}_${Date.now()}`,
            method,
            params,
          }),
        });

        const raw = await response.text();
        const payload = parseSseJson(raw);

        if (!response.ok) {
          const error = new Error(`MCP HTTP ${response.status}: ${JSON.stringify(payload).slice(0, 500)}`);
          errors.push(`[${baseUrl}] attempt=${attempt} ${error.message}`);
          if (!isRetryableMcpError(error, response.status) || attempt >= mcpRetryMax) {
            break;
          }
          await sleep(mcpRetryBackoffMs * attempt);
          continue;
        }
        if (payload?.error) {
          const error = new Error(`MCP JSON-RPC error: ${JSON.stringify(payload.error).slice(0, 500)}`);
          errors.push(`[${baseUrl}] attempt=${attempt} ${error.message}`);
          if (!isRetryableMcpError(error, null) || attempt >= mcpRetryMax) {
            break;
          }
          await sleep(mcpRetryBackoffMs * attempt);
          continue;
        }

        activeMcpBaseUrl = baseUrl;
        return payload?.result;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        const code = toErrorCode(error);
        if (code === "ENOTFOUND" || code === "EAI_AGAIN") {
          hadDnsFailure = true;
        }
        errors.push(`[${baseUrl}] attempt=${attempt} ${message}`);
        if (!isRetryableMcpError(error, null) || attempt >= mcpRetryMax) {
          break;
        }
        await sleep(mcpRetryBackoffMs * attempt);
      }
    }
  }
  const localHint = hadDnsFailure
    ? ` DNS lookup failed. If hosted MCP is blocked, run local MCP server and retry: WANDB_MCP_TRANSPORT=http WANDB_MCP_PORT=${mcpLocalPort} uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server`
    : "";
  throw new Error(
    `MCP request failed for method=${method} after retries. candidates=${mcpBaseUrls.join(", ")}. recent_errors=${errors.slice(-6).join(" || ")}.${localHint}`,
  );
};

const callTool = async (name, args) => {
  const result = await mcpRequest("tools/call", { name, arguments: args }, `call_${name}`);
  if (result?.isError) {
    const preview = JSON.stringify(result?.content ?? result).slice(0, 700);
    throw new Error(`MCP tool call failed (${name}): ${preview}`);
  }
  if (result?.structuredContent && "result" in result.structuredContent) {
    return result.structuredContent.result;
  }
  const textContent = Array.isArray(result?.content)
    ? result.content.find((item) => item?.type === "text")?.text
    : null;
  return textContent ?? result;
};

const fetchTools = async () => {
  const list = await mcpRequest("tools/list", {}, "tools_list");
  const tools = Array.isArray(list?.tools) ? list.tools : [];
  const names = tools.map((tool) => tool?.name).filter((name) => typeof name === "string");

  for (const required of ["query_wandb_tool", "query_weave_traces_tool", "count_weave_traces_tool"]) {
    if (!names.includes(required)) {
      throw new Error(`Required MCP tool missing: ${required}`);
    }
  }

  return names;
};

const fetchViewerEntity = async () => {
  const query = `query Viewer { viewer { id username entity } }`;
  const result = await callTool("query_wandb_tool", {
    query,
    variables: {},
    max_items: 5,
    items_per_page: 5,
  });

  const entity =
    process.env.WANDB_ENTITY ||
    process.env.WANDB_MCP_ENTITY ||
    result?.viewer?.entity ||
    "";

  if (!entity) {
    throw new Error("Unable to resolve W&B entity via env or MCP viewer query.");
  }

  return {
    entity,
    viewer: result?.viewer ?? null,
  };
};

const fetchProjectRuns = async ({ entity, project, filters, first = 40 }) => {
  const query = `
query ProjectRuns($entity: String!, $project: String!, $first: Int!, $filters: JSONString) {
  project(name: $project, entityName: $entity) {
    name
    runs(first: $first, order: "-createdAt", filters: $filters) {
      edges {
        node {
          id
          name
          displayName
          state
          jobType
          group
          createdAt
          heartbeatAt
          config
          summaryMetrics
          tags
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
`;

  const result = await callTool("query_wandb_tool", {
    query,
    variables: {
      entity,
      project,
      first,
      filters: JSON.stringify(filters ?? {}),
    },
    max_items: Math.max(20, first),
    items_per_page: Math.min(50, Math.max(10, first)),
  });

  const edges = result?.project?.runs?.edges;
  if (!Array.isArray(edges)) {
    return [];
  }

  return edges
    .map((edge) => edge?.node)
    .filter((node) => node && typeof node === "object")
    .map((node) => {
      const config = normalizeConfig(node.config);
      const summary = normalizeSummary(node.summaryMetrics);
      const tags = Array.isArray(node.tags) ? node.tags : [];
      const modeFromConfig = toMode(config.mode);
      const modeFromTag = tags.map(toMode).find((mode) => mode);
      const modeFromSummary = toMode(summary["eval/model_source"]);

      return {
        id: node.id,
        runId: node.name,
        displayName: node.displayName || node.name,
        state: node.state,
        jobType: node.jobType,
        group: node.group,
        createdAt: node.createdAt,
        heartbeatAt: node.heartbeatAt,
        tags,
        config,
        summary,
        mode: modeFromConfig || modeFromSummary || modeFromTag || null,
        url: `https://wandb.ai/${entity}/${project}/runs/${node.name}`,
      };
    });
};

const toEvalRunPayload = (row, projectForWeave) => {
  const metrics = Object.fromEntries(EVAL_METRIC_KEYS.map((key) => [key, extractMetric(row.summary, key)]));

  return {
    dataset_version: row.summary["dataset/version"] || row.config.dataset_version || "mcp_unknown",
    mode: row.mode,
    model_source: row.summary["eval/model_source"] || row.mode,
    model_id: row.summary["eval/model_id"] || row.config.fine_tuned_model_id || row.config.prompt_model_id || "",
    evaluated_count: asNumber(row.summary["eval/evaluated_count"], 0) ?? 0,
    timestamp: row.createdAt,
    metrics,
    weave_project: projectForWeave,
    file: `wandb-run-${row.runId}.json`,
    wandb_run_id: row.runId,
    wandb_run_url: row.url,
  };
};

const buildEvalSummary = ({ evalRuns, weaveProject }) => {
  const latestByMode = {};

  for (const row of evalRuns) {
    if (!row.mode) continue;
    const existing = latestByMode[row.mode];
    if (!existing || row.createdAt > existing.timestamp) {
      latestByMode[row.mode] = toEvalRunPayload(row, weaveProject);
    }
  }

  const prompt = latestByMode.prompt_baseline?.metrics ?? null;
  const ft = latestByMode.fine_tuned?.metrics ?? null;
  const delta =
    prompt && ft
      ? {
          intent_score_mean: (ft.intent_score_mean ?? 0) - (prompt.intent_score_mean ?? 0),
          vector_mae: (ft.vector_mae ?? 0) - (prompt.vector_mae ?? 0),
          json_valid_rate: (ft.json_valid_rate ?? 0) - (prompt.json_valid_rate ?? 0),
        }
      : {
          intent_score_mean: 0,
          vector_mae: 0,
          json_valid_rate: 0,
        };

  const modeCount = Object.keys(latestByMode).length;
  const loopCompletionRate = modeCount === 0 ? 0 : Math.min(1, modeCount / 3);

  return {
    generated_at: new Date().toISOString(),
    runs_count: evalRuns.length,
    latest_by_mode: latestByMode,
    auto_improvement_delta: delta,
    loop_completion_rate: loopCompletionRate,
    loop_detail: {
      completed_modes: modeCount,
      planned_modes: 3,
      rate: loopCompletionRate,
    },
  };
};

const pickNumber = (...values) => {
  for (const value of values) {
    const parsed = asNumber(value);
    if (parsed != null) return parsed;
  }
  return null;
};

const buildRecentRuns = (rows, maxItems) =>
  rows.slice(0, maxItems).map((row) => ({
    run_id: row.runId,
    name: row.displayName,
    url: row.url,
    state: row.state,
    created_at: row.createdAt,
    job_type: row.jobType,
    group: row.group,
    config: {
      learning_rate: pickNumber(row.config.learning_rate, row.config.HF_FT_LR),
      lora_r: pickNumber(row.config.lora_r, row.config.HF_FT_LORA_R),
      lora_alpha: pickNumber(row.config.lora_alpha, row.config.HF_FT_LORA_ALPHA),
      lora_dropout: pickNumber(row.config.lora_dropout, row.config.HF_FT_LORA_DROPOUT),
      epochs: pickNumber(row.config.epochs, row.config.HF_FT_EPOCHS, row.config.num_train_epochs),
      dataset_version: row.config.dataset_version || row.summary["dataset/version"] || null,
    },
    summary: {
      "eval/mse_norm": pickNumber(row.summary["eval/mse_norm"], row.summary["iter_eval/mse_norm"]),
      "eval/mae_raw": pickNumber(row.summary["eval/mae_raw"], row.summary["iter_eval/mae_raw"]),
      "eval/json_valid_rate": pickNumber(row.summary["eval/json_valid_rate"]),
      "objective/train_loss": pickNumber(
        row.summary["objective/train_loss"],
        row.summary["train/loss"],
        row.summary.train_loss,
      ),
    },
  }));

const toEvalLikeFromTrainRuns = (rows) =>
  rows
    .filter(
      (row) =>
        pickNumber(
          row.summary["iter_eval/mae_raw"],
          row.summary["eval/mae_raw"],
          row.summary["iter_eval/mse_norm"],
          row.summary["eval/mse_norm"],
        ) != null,
    )
    .map((row) => ({
      ...row,
      mode: row.mode || "fine_tuned",
    }));

const countTraces = async ({ entity, project }) => {
  try {
    const result = await callTool("count_weave_traces_tool", {
      entity_name: entity,
      project_name: project,
      filters: {
        trace_roots_only: true,
      },
    });
    const parsed = parseMaybeJson(result);
    return {
      project,
      total_count: asInt(parsed?.total_count, 0),
      root_traces_count: asInt(parsed?.root_traces_count, 0),
      ok: true,
    };
  } catch (error) {
    return {
      project,
      total_count: 0,
      root_traces_count: 0,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
};

const toTraceUrl = ({ entity, project, traceId }) => {
  if (!traceId) return null;
  return `https://wandb.ai/${entity}/${project}/weave/traces?query=${encodeURIComponent(traceId)}`;
};

const normalizeAbsErrorByDim = (value) => {
  if (!value || typeof value !== "object") return {};
  const out = {};
  for (const [key, raw] of Object.entries(value)) {
    const parsed = asNumber(raw);
    if (parsed != null) out[key] = parsed;
  }
  return out;
};

const failureScore = (row) => {
  const mae = asNumber(row.mae_raw, 0) ?? 0;
  const slot = asNumber(row.slot_exact_match, 1) ?? 1;
  const sanity = asNumber(row.output_sanity_score, 100) ?? 100;
  const intent = asNumber(row.intent_score, 100) ?? 100;

  let score = mae;
  if (row.json_valid === false || row.parse_error) score += 1000;
  if (row.constraint_match === false) score += 50;
  score += (1 - Math.max(0, Math.min(1, slot))) * 30;
  score += Math.max(0, 90 - sanity);
  score += Math.max(0, 60 - intent) * 0.5;
  return score;
};

const fetchTraceFailures = async ({ entity, project, limit, topK }) => {
  const result = await callTool("query_weave_traces_tool", {
    entity_name: entity,
    project_name: project,
    filters: {
      trace_roots_only: true,
    },
    columns: ["id", "trace_id", "op_name", "started_at", "inputs", "exception"],
    include_costs: false,
    include_feedback: false,
    limit,
    return_full_data: true,
    metadata_only: false,
  });

  const parsed = parseMaybeJson(result);
  const traces = Array.isArray(parsed?.traces) ? parsed.traces : [];

  const rows = traces
    .map((trace) => {
      const payload = trace?.inputs?.payload;
      if (!payload || typeof payload !== "object") return null;

      const absErrorByDim = normalizeAbsErrorByDim(payload.abs_error_by_dim ?? payload.abs_error_raw);
      const mae =
        asNumber(payload.mae_raw) ??
        (Object.keys(absErrorByDim).length > 0 ? mean(Object.values(absErrorByDim)) : null);
      const traceId =
        (typeof payload.trace_id === "string" && payload.trace_id) ||
        (typeof trace.trace_id === "string" && trace.trace_id) ||
        (typeof trace.id === "string" && trace.id) ||
        null;

      const parseError = typeof payload.parse_error === "string" && payload.parse_error.length > 0 ? payload.parse_error : null;
      const jsonValid = typeof payload.json_valid === "boolean" ? payload.json_valid : !parseError;

      return {
        id: traceId,
        trace_id: traceId,
        trace_url: toTraceUrl({ entity, project, traceId }),
        mode: typeof payload.mode === "string" ? payload.mode : "fine_tuned",
        scenario:
          typeof payload.scenario === "string"
            ? payload.scenario
            : typeof payload.run_name === "string"
              ? payload.run_name
              : typeof trace?.op_name === "string"
                ? trace.op_name
                : null,
        source_type:
          typeof payload.scenario === "string"
            ? payload.scenario
            : typeof trace?.op_name === "string"
              ? trace.op_name
              : "mcp_weave_trace",
        request_text: typeof payload.request_text === "string" ? payload.request_text : "",
        json_valid: jsonValid,
        parse_error: parseError,
        abs_error_by_dim: absErrorByDim,
        mae_raw: mae,
        intent_score: asNumber(payload.intent_score),
        output_sanity_score: asNumber(payload.output_sanity_score),
        constraint_match:
          typeof payload.constraint_match === "boolean" ? payload.constraint_match : null,
        slot_exact_match: asNumber(payload.slot_exact_match),
        target_vector:
          payload.target_vector && typeof payload.target_vector === "object"
            ? payload.target_vector
            : {},
        predicted_vector:
          payload.predicted_vector && typeof payload.predicted_vector === "object"
            ? payload.predicted_vector
            : null,
        run_name: typeof payload.run_name === "string" ? payload.run_name : null,
        dataset_version: typeof payload.dataset_version === "string" ? payload.dataset_version : null,
        started_at: trace?.started_at || null,
      };
    })
    .filter((row) => row !== null);

  const fineTunedRows = rows.filter((row) => row.mode === "fine_tuned");
  const sourceRows = fineTunedRows.length > 0 ? fineTunedRows : rows;
  const withDimErrors = sourceRows.filter(
    (row) => row.json_valid === true && Object.keys(row.abs_error_by_dim || {}).length > 0,
  );
  const withoutDimErrors = sourceRows.filter(
    (row) => !(row.json_valid === true && Object.keys(row.abs_error_by_dim || {}).length > 0),
  );

  const dimQuota = Math.min(topK, Math.max(5, Math.ceil(topK * 0.7)));
  const dimFirst = [...withDimErrors]
    .sort((a, b) => failureScore(b) - failureScore(a))
    .slice(0, dimQuota);

  const used = new Set(
    dimFirst.map((row) => `${row.trace_id || row.id || "unknown"}:${row.started_at || ""}`),
  );
  const rest = [...withoutDimErrors, ...withDimErrors]
    .sort((a, b) => failureScore(b) - failureScore(a))
    .filter((row) => !used.has(`${row.trace_id || row.id || "unknown"}:${row.started_at || ""}`));

  const failuresTopK = [...dimFirst, ...rest.slice(0, Math.max(0, topK - dimFirst.length))];

  return {
    rows,
    failuresTopK,
    total: rows.length,
    selected_mode: fineTunedRows.length > 0 ? "fine_tuned" : "all_modes",
  };
};

const buildDecisionInput = (snapshot, topK) => {
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
      loop_completion_rate: evalSummary.loop_completion_rate ?? 0,
    },
    failures_top_k: failures.slice(0, topK),
    recent_runs: recentRuns.slice(0, 10),
  };
};

const main = async () => {
  await mkdir(cycleDir, { recursive: true });

  const topK = Math.max(1, asInt(process.env.WANDB_MCP_TOP_K, 20));
  const recentRunsLimit = Math.max(1, asInt(process.env.WANDB_MCP_RECENT_RUNS, 10));
  const traceFetchLimit = Math.max(topK * 4, asInt(process.env.WANDB_MCP_TRACE_FETCH_LIMIT, 300));

  const toolNames = await fetchTools();
  const { entity, viewer } = await fetchViewerEntity();
  const modelsProject = process.env.WANDB_PROJECT || process.env.WANDB_MCP_PROJECT || "atelier-kotone-ft";

  const evalRuns = await fetchProjectRuns({
    entity,
    project: modelsProject,
    filters: {
      state: "finished",
      jobType: "eval",
    },
    first: Math.max(12, asInt(process.env.WANDB_MCP_EVAL_RUN_FETCH_LIMIT, 30)),
  });

  const trainRuns = await fetchProjectRuns({
    entity,
    project: modelsProject,
    filters: {
      state: "finished",
      jobType: "train",
    },
    first: Math.max(20, recentRunsLimit * 3),
  });

  const evalLikeRuns = evalRuns.length > 0 ? evalRuns : toEvalLikeFromTrainRuns(trainRuns);
  const evalSource =
    evalRuns.length > 0
      ? "eval_runs"
      : evalLikeRuns.length > 0
        ? "train_iter_eval_runs"
        : "none";

  const traceProjects = uniq([
    process.env.WANDB_MCP_WEAVE_PROJECT,
    process.env.WEAVE_PROJECT,
    modelsProject,
  ]);

  const traceCounts = [];
  for (const project of traceProjects) {
    // Sequential is safer for MCP request volume and easier to debug.
    // eslint-disable-next-line no-await-in-loop
    const count = await countTraces({ entity, project });
    traceCounts.push(count);
  }

  const selectedTraceProject =
    [...traceCounts].sort((a, b) => b.total_count - a.total_count)[0] || {
      project: modelsProject,
      total_count: 0,
      root_traces_count: 0,
      ok: false,
    };

  const traceFailures =
    selectedTraceProject.total_count > 0
      ? await fetchTraceFailures({
          entity,
          project: selectedTraceProject.project,
          limit: traceFetchLimit,
          topK,
        })
      : {
          rows: [],
          failuresTopK: [],
          total: 0,
          selected_mode: "none",
        };

  const evalSummary = buildEvalSummary({
    evalRuns: evalLikeRuns,
    weaveProject: selectedTraceProject.project,
  });

  const recentRuns = buildRecentRuns(trainRuns, recentRunsLimit);

  const snapshot = {
    source: "wandb_mcp_tools_call",
    fetched_at: new Date().toISOString(),
    cycle_id: cycleId,
    mcp: {
      base_url: activeMcpBaseUrl,
      base_url_candidates: mcpBaseUrls,
      tools: toolNames,
      viewer,
      entity,
      project: modelsProject,
      eval_source: evalSource,
    },
    eval_summary: evalSummary,
    failures_top_k: traceFailures.failuresTopK,
    failures_len: traceFailures.total,
    failures_selected_mode: traceFailures.selected_mode,
    recent_runs: recentRuns,
    recent_run_count: recentRuns.length,
    trace_project_selected: selectedTraceProject,
    trace_project_candidates: traceCounts,
    top_k: topK,
  };

  const decisionInput = buildDecisionInput(snapshot, topK);

  await writeFile(snapshotPath, JSON.stringify(snapshot, null, 2));
  await writeFile(decisionPath, JSON.stringify(decisionInput, null, 2));

  console.log(
    JSON.stringify(
      {
        cycle_id: cycleId,
        snapshot_path: snapshotPath,
        decision_input_path: decisionPath,
        source: snapshot.source,
        entity,
        project: modelsProject,
        trace_project: selectedTraceProject.project,
        trace_count: traceFailures.total,
        failures_top_k_count: snapshot.failures_top_k.length,
        recent_run_count: snapshot.recent_run_count,
      },
      null,
      2,
    ),
  );
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
