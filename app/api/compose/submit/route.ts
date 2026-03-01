import { NextResponse } from "next/server";
import { z } from "zod";
import {
  buildCoachFeedbackBySlot,
  buildDistanceBreakdown,
  calculateRank,
  ensureSlotCategoryIntegrity,
  scoreComposition
} from "@/lib/score";
import { isCombinationCovered } from "@/lib/ft-test-dataset";
import { submitCompositionSchema, targetProfileSchema } from "@/lib/schemas";

const requestSchema = submitCompositionSchema.extend({
  targetProfile: targetProfileSchema
});

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const parsed = requestSchema.safeParse(body);

    if (!parsed.success) {
      return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
    }

    const slotIntegrityErrors = ensureSlotCategoryIntegrity(parsed.data.selectedPartsBySlot);
    if (slotIntegrityErrors.length > 0) {
      return NextResponse.json({ error: slotIntegrityErrors }, { status: 400 });
    }

    const isCovered = await isCombinationCovered(parsed.data.selectedPartsBySlot);
    if (!isCovered) {
      return NextResponse.json(
        {
          error: "selected_kotone_combination_not_covered_by_reference_dataset",
          selectedPartsBySlot: parsed.data.selectedPartsBySlot
        },
        { status: 400 }
      );
    }

    const breakdown = scoreComposition(parsed.data.selectedPartsBySlot, parsed.data.targetProfile);
    const distanceBreakdown = buildDistanceBreakdown(
      parsed.data.selectedPartsBySlot,
      parsed.data.targetProfile
    );
    const rank = calculateRank(breakdown.total);
    const rankBonusByRank = { S: 24, A: 14, B: 8, C: 3, D: 0 } as const;
    const scoreReward = Math.round(breakdown.total * 1.6);
    const rewardMoney = Math.max(20, scoreReward + rankBonusByRank[rank]);

    const response = {
      score: breakdown.total,
      rank,
      rewardMoney,
      coachFeedbackBySlot: buildCoachFeedbackBySlot(parsed.data.selectedPartsBySlot, parsed.data.targetProfile),
      distanceBreakdown,
      evalSnapshot: {
        intent_score_mean: breakdown.total,
        timestamp: new Date().toISOString()
      }
    };

    return NextResponse.json(response);
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: error.flatten() }, { status: 400 });
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Compose submit error" },
      { status: 500 }
    );
  }
}
