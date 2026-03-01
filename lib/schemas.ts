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

export const targetHiddenParamsSchema = z.object({
  vector: vectorSchema,
  tags: z.array(z.string()),
  constraints: z.record(z.unknown())
});

export const promptHiddenParamEvalSchema = z.object({
  mae_raw_mean: z.number().nonnegative(),
  mse_raw: z.number().nonnegative(),
  mae_raw_by_dim: z.object({
    energy: z.number().nonnegative(),
    warmth: z.number().nonnegative(),
    brightness: z.number().nonnegative(),
    acousticness: z.number().nonnegative(),
    complexity: z.number().nonnegative(),
    nostalgia: z.number().nonnegative()
  }),
  max_error_dim: z.enum(["energy", "warmth", "brightness", "acousticness", "complexity", "nostalgia"]),
  max_error_value: z.number().nonnegative(),
  tags_jaccard: z.number().min(0).max(1),
  summary: z.string(),
  next_actions: z.array(z.string())
});

export const interpreterEvaluationMetaSchema = z.object({
  json_valid: z.boolean(),
  parse_error: z.string().optional(),
  model_source: z.enum([
    "rule_baseline",
    "prompt_baseline",
    "fine_tuned",
    "firebase_callable",
    "unknown"
  ]),
  latency_ms: z.number().nonnegative().optional(),
  trace_id: z.string().optional(),
  cost_usd: z.number().nonnegative().optional()
});

export const distanceBreakdownSchema = z.object({
  vectorDistance: z.number().nonnegative(),
  normalizedDistance: z.number().min(0).max(1),
  vectorScore: z.number().min(0).max(60),
  tagScore: z.number().min(0).max(25),
  preferenceScore: z.number().min(-10).max(10),
  synergyScore: z.number().min(0).max(10),
  antiSynergyPenalty: z.number().min(-10).max(0)
});

export const evalSnapshotSchema = z.object({
  intent_score_mean: z.number().optional(),
  slot_exact_match: z.number().optional(),
  constraint_match_rate: z.number().optional(),
  json_valid_rate: z.number().optional(),
  vector_mae: z.number().optional(),
  mse_raw: z.number().optional(),
  mse_norm: z.number().optional(),
  timestamp: z.string()
});

export const interpreterRequestSchema = z.object({
  commissionId: z.string().optional(),
  requestText: z.string().min(1),
  weather: z.enum(["sunny", "cloudy", "rainy"]),
  inventoryPartIds: z.array(z.string())
});

export const requestGenerationRequestSchema = z.object({
  templateText: z.string().min(1),
  weather: z.enum(["sunny", "cloudy", "rainy"]),
  customerId: z.string().min(1),
  customerName: z.string().optional(),
  customerPersonality: z.string().optional()
});

export const requestGenerationResponseSchema = z.object({
  requestText: z.string().min(1),
  modelSource: z.enum([
    "rule_baseline",
    "prompt_baseline",
    "fine_tuned",
    "firebase_callable",
    "unknown"
  ]),
  latencyMs: z.number().nonnegative(),
  traceId: z.string().optional(),
  parseError: z.string().optional(),
  targetProfile: targetProfileSchema.optional(),
  targetHiddenParams: targetHiddenParamsSchema.optional()
});

export const interpreterResponseSchema = z.object({
  recommended: slotRecord(z.array(z.string())),
  targetProfile: targetProfileSchema,
  targetHiddenParams: targetHiddenParamsSchema,
  hintToPlayer: z.string(),
  rationale: z.array(z.string()),
  evaluationMeta: interpreterEvaluationMetaSchema
});

export const submitCompositionSchema = z.object({
  commissionId: z.string().min(1),
  selectedPartsBySlot: slotRecord(z.string().min(1))
});

export const createMusicRequestSchema = z.object({
  commissionId: z.string().min(1),
  requestText: z.string().min(1),
  weather: z.enum(["sunny", "cloudy", "rainy"]).optional(),
  selectedPartsBySlot: slotRecord(z.string().min(1)),
  targetHiddenParams: targetHiddenParamsSchema.optional()
});

export const scoreResponseSchema = z.object({
  score: z.number().min(0).max(100),
  rank: z.enum(["S", "A", "B", "C", "D"]),
  coachFeedbackBySlot: slotRecord(z.string()),
  rewardMoney: z.number().int().min(0),
  distanceBreakdown: distanceBreakdownSchema.optional(),
  evalSnapshot: evalSnapshotSchema.optional()
});

export const musicJobStatusResponseSchema = z.object({
  status: z.enum(["queued", "running", "done", "failed"]),
  audioUrl: z.string().optional(),
  error: z.string().optional(),
  rulePrompt: z.string().optional(),
  compositionPlan: z.unknown().optional(),
  songMetadata: z.unknown().optional(),
  outputSanityScore: z.number().optional(),
  promptInferenceHiddenParams: targetHiddenParamsSchema.optional(),
  promptInferenceMeta: interpreterEvaluationMetaSchema.optional(),
  promptEval: promptHiddenParamEvalSchema.optional(),
  promptFeedback: z.string().optional(),
  traceId: z.string().optional()
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
