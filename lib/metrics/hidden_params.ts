import type { HiddenParamsEvalMetrics, TargetHiddenParams } from "@/lib/types";

type Sample = {
  predicted: TargetHiddenParams | null;
  target: TargetHiddenParams;
  jsonValid: boolean;
};

const KEYS: Array<keyof TargetHiddenParams["vector"]> = [
  "energy",
  "warmth",
  "brightness",
  "acousticness",
  "complexity",
  "nostalgia"
];

const mean = (values: number[]): number => {
  if (values.length === 0) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
};

export const computeHiddenParamsMetrics = (samples: Sample[]): HiddenParamsEvalMetrics => {
  if (samples.length === 0) {
    return {
      json_valid_rate: 0,
      vector_mae: 0,
      mse_raw: 0,
      mse_norm: 0,
      r2_score: 0
    };
  }

  const validCount = samples.filter((sample) => sample.jsonValid && sample.predicted).length;

  const absErrors: number[] = [];
  const sqErrors: number[] = [];
  const targetValues: number[] = [];

  for (const sample of samples) {
    if (!sample.predicted) continue;

    for (const key of KEYS) {
      const predictedValue = sample.predicted.vector[key];
      const targetValue = sample.target.vector[key];
      const diff = predictedValue - targetValue;

      absErrors.push(Math.abs(diff));
      sqErrors.push(diff * diff);
      targetValues.push(targetValue);
    }
  }

  const vectorMae = mean(absErrors);
  const mseRaw = mean(sqErrors);
  const mseNorm = mseRaw / (100 * 100);

  const targetMean = mean(targetValues);
  const ssTot = targetValues.reduce((acc, value) => acc + Math.pow(value - targetMean, 2), 0);
  const ssRes = sqErrors.reduce((acc, value) => acc + value, 0);
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

  return {
    json_valid_rate: validCount / samples.length,
    vector_mae: vectorMae,
    mse_raw: mseRaw,
    mse_norm: mseNorm,
    r2_score: r2
  };
};

export type HiddenParamsEvalSample = Sample;
