import { NextResponse } from "next/server";
import { z } from "zod";
import { buildCoachFeedbackBySlot, calculateRank, ensureSlotCategoryIntegrity, scoreComposition } from "@/lib/score";
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

    const breakdown = scoreComposition(parsed.data.selectedPartsBySlot, parsed.data.targetProfile);
    const rank = calculateRank(breakdown.total);
    const rewardMoney = 30 + Math.round(breakdown.total * 1.8);

    const response = {
      score: breakdown.total,
      rank,
      rewardMoney,
      coachFeedbackBySlot: buildCoachFeedbackBySlot(parsed.data.selectedPartsBySlot, parsed.data.targetProfile)
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
