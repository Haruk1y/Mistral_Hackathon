import { z } from "zod";
import { SLOT_KEYS } from "@/lib/types";

const slotRecord = <T extends z.ZodTypeAny>(schema: T) =>
  z.object(
    Object.fromEntries(SLOT_KEYS.map((key) => [key, schema])) as Record<(typeof SLOT_KEYS)[number], T>
  );

export const vectorSchema = z.object({
  energy: z.number().min(0).max(100),
  warmth: z.number().min(0).max(100),
  brightness: z.number().min(0).max(100),
  acousticness: z.number().min(0).max(100),
  complexity: z.number().min(0).max(100),
  nostalgia: z.number().min(0).max(100)
});

export const targetProfileSchema = z.object({
  vector: vectorSchema,
  requiredTags: z.array(z.string()),
  optionalTags: z.array(z.string()),
  forbiddenTags: z.array(z.string()),
  constraints: z
    .object({
      preferredStyleTags: z.array(z.string()).optional(),
      preferredGimmickTags: z.array(z.string()).optional(),
      avoidPartIds: z.array(z.string()).optional()
    })
    .default({})
});

export const interpreterRequestSchema = z.object({
  requestText: z.string().min(1),
  weather: z.enum(["sunny", "cloudy", "rainy"]),
  inventoryPartIds: z.array(z.string())
});

export const interpreterResponseSchema = z.object({
  recommended: slotRecord(z.array(z.string())),
  targetProfile: targetProfileSchema,
  hintToPlayer: z.string(),
  rationale: z.array(z.string())
});

export const submitCompositionSchema = z.object({
  commissionId: z.string().min(1),
  selectedPartsBySlot: slotRecord(z.string().min(1))
});

export const createMusicRequestSchema = z.object({
  commissionId: z.string().min(1),
  requestText: z.string().min(1),
  selectedPartsBySlot: slotRecord(z.string().min(1))
});

export const scoreResponseSchema = z.object({
  score: z.number().min(0).max(100),
  rank: z.enum(["S", "A", "B", "C", "D"]),
  coachFeedbackBySlot: slotRecord(z.string()),
  rewardMoney: z.number().int().min(0)
});

export const gameStateSchema = z.object({
  schemaVersion: z.number().int().min(1),
  money: z.number().int(),
  day: z.number().int().min(1),
  weather: z.enum(["sunny", "cloudy", "rainy"]),
  inventoryPartIds: z.array(z.string()),
  commissions: z.record(z.unknown()),
  commissionOrder: z.array(z.string()),
  tracks: z.record(z.unknown()),
  trackOrder: z.array(z.string()),
  jobs: z.record(z.unknown()),
  shopStock: z.array(
    z.object({
      partId: z.string(),
      unlocked: z.boolean()
    })
  ),
  selectedCommissionId: z.string().optional(),
  updatedAt: z.string()
});
