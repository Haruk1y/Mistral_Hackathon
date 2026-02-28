import { scoreComposition } from "@/lib/score";
import type { SlotKey, TargetProfile } from "@/lib/types";

type IntentSample = {
  selectedPartsBySlot: Record<SlotKey, string>;
  targetProfile: TargetProfile;
};

export const computeIntentScoreMean = (samples: IntentSample[]): number => {
  if (samples.length === 0) return 0;

  const total = samples.reduce((acc, sample) => {
    const breakdown = scoreComposition(sample.selectedPartsBySlot, sample.targetProfile);
    return acc + breakdown.total;
  }, 0);

  return total / samples.length;
};

export type IntentEvalSample = IntentSample;
