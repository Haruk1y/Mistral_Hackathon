import type { ConstraintEvalMetrics, SlotKey, TargetHiddenParams } from "@/lib/types";

type ConstraintSample = {
  predicted: TargetHiddenParams | null;
  target: TargetHiddenParams;
};

type SlotSample = {
  predicted: Record<SlotKey, string[]>;
  expected: Record<SlotKey, string[]>;
};

const normalizeRecord = (value: Record<string, unknown>): string => {
  const sortedKeys = Object.keys(value).sort();
  const normalized = Object.fromEntries(sortedKeys.map((key) => [key, value[key]]));
  return JSON.stringify(normalized);
};

export const computeConstraintMatchRate = (samples: ConstraintSample[]): number => {
  if (samples.length === 0) return 0;

  let matches = 0;
  for (const sample of samples) {
    if (!sample.predicted) continue;

    const left = normalizeRecord(sample.predicted.constraints);
    const right = normalizeRecord(sample.target.constraints);
    if (left === right) {
      matches += 1;
    }
  }

  return matches / samples.length;
};

export const computeSlotExactMatchRate = (samples: SlotSample[]): number => {
  if (samples.length === 0) return 0;

  let totalSlots = 0;
  let exactSlots = 0;

  for (const sample of samples) {
    for (const slot of Object.keys(sample.expected) as SlotKey[]) {
      totalSlots += 1;
      const predictedTop1 = sample.predicted[slot]?.[0];
      const expectedTop1 = sample.expected[slot]?.[0];
      if (predictedTop1 && expectedTop1 && predictedTop1 === expectedTop1) {
        exactSlots += 1;
      }
    }
  }

  return totalSlots === 0 ? 0 : exactSlots / totalSlots;
};

export const computeConstraintMetrics = (
  constraintSamples: ConstraintSample[],
  slotSamples: SlotSample[]
): ConstraintEvalMetrics => {
  return {
    constraint_match_rate: computeConstraintMatchRate(constraintSamples),
    slot_exact_match: computeSlotExactMatchRate(slotSamples)
  };
};

export type ConstraintEvalSample = ConstraintSample;
export type SlotExactEvalSample = SlotSample;
