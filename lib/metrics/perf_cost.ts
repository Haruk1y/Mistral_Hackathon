import type { PerfCostMetrics } from "@/lib/types";

type PerfCostSample = {
  latencyMs: number;
  costUsd: number;
};

const percentile = (values: number[], p: number): number => {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx] ?? 0;
};

export const computePerfCostMetrics = (samples: PerfCostSample[]): PerfCostMetrics => {
  if (samples.length === 0) {
    return {
      p50_inference_latency_ms: 0,
      p95_inference_latency_ms: 0,
      cost_per_100_requests_usd: 0
    };
  }

  const latencies = samples.map((sample) => sample.latencyMs).filter((value) => Number.isFinite(value));
  const costs = samples.map((sample) => sample.costUsd).filter((value) => Number.isFinite(value));

  const averageCost = costs.length
    ? costs.reduce((acc, value) => acc + value, 0) / costs.length
    : Number(process.env.DEFAULT_COST_PER_1K_REQUEST_USD || "0") / 1000;

  return {
    p50_inference_latency_ms: percentile(latencies, 50),
    p95_inference_latency_ms: percentile(latencies, 95),
    cost_per_100_requests_usd: averageCost * 100
  };
};

export type PerfCostEvalSample = PerfCostSample;
