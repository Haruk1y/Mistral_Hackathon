export type SlotKey = "style" | "instrument" | "mood" | "gimmick";

export const SLOT_KEYS: SlotKey[] = ["style", "instrument", "mood", "gimmick"];

export type Weather = "sunny" | "cloudy" | "rainy";

export type Rank = "S" | "A" | "B" | "C" | "D";

export type CommissionStatus = "queued" | "mixing" | "generating" | "delivered";

export type MusicJobStatus = "queued" | "running" | "done" | "failed";

export type FeatureFlags = {
  featureVoiceRequests: boolean;
  featureWandbWeave: boolean;
  featureFineTuneExport: boolean;
  featureShareLinks: boolean;
};

export type ProfileVector = {
  energy: number;
  warmth: number;
  brightness: number;
  acousticness: number;
  complexity: number;
  nostalgia: number;
};

export type TargetProfile = {
  vector: ProfileVector;
  requiredTags: string[];
  optionalTags: string[];
  forbiddenTags: string[];
  constraints: {
    preferredStyleTags?: string[];
    preferredGimmickTags?: string[];
    avoidPartIds?: string[];
  };
};

export type TargetHiddenParams = {
  vector: ProfileVector;
  tags: string[];
  constraints: Record<string, unknown>;
};

export type HiddenParamVectorKey = keyof ProfileVector;

export type PromptHiddenParamEval = {
  mae_raw_mean: number;
  mse_raw: number;
  mae_raw_by_dim: Record<HiddenParamVectorKey, number>;
  max_error_dim: HiddenParamVectorKey;
  max_error_value: number;
  tags_jaccard: number;
  summary: string;
  next_actions: string[];
};

export type InterpreterModelSource =
  | "rule_baseline"
  | "prompt_baseline"
  | "fine_tuned"
  | "firebase_callable"
  | "unknown";

export type InterpreterEvaluationMeta = {
  json_valid: boolean;
  parse_error?: string;
  model_source: InterpreterModelSource;
  latency_ms?: number;
  trace_id?: string;
  cost_usd?: number;
};

export type Part = {
  id: string;
  slot: SlotKey;
  name: string;
  description: string;
  iconAsset?: string;
  price: number;
  tags: string[];
  vector: Partial<ProfileVector>;
};

export type Customer = {
  id: string;
  name: string;
  portraitAsset: string;
  personality: string;
};

export type RequestTemplate = {
  id: string;
  customerId: string;
  text: string;
  weatherBias: Weather[];
  targetProfile: TargetProfile;
};

export type InterpreterRequest = {
  requestText: string;
  weather: Weather;
  inventoryPartIds: string[];
  commissionId?: string;
};

export type RequestGenerationRequest = {
  templateText: string;
  weather: Weather;
  customerId: string;
  customerName?: string;
  customerPersonality?: string;
};

export type RequestGenerationResponse = {
  requestText: string;
  modelSource: InterpreterModelSource;
  latencyMs: number;
  traceId?: string;
  parseError?: string;
};

export type InterpreterResponse = {
  recommended: Record<SlotKey, string[]>;
  targetProfile: TargetProfile;
  targetHiddenParams: TargetHiddenParams;
  hintToPlayer: string;
  rationale: string[];
  evaluationMeta: InterpreterEvaluationMeta;
};

export type SubmitCompositionRequest = {
  commissionId: string;
  selectedPartsBySlot: Record<SlotKey, string>;
};

export type DistanceBreakdown = {
  vectorDistance: number;
  normalizedDistance: number;
  vectorScore: number;
  tagScore: number;
  preferenceScore: number;
  synergyScore: number;
  antiSynergyPenalty: number;
};

export type EvalSnapshot = {
  intent_score_mean?: number;
  slot_exact_match?: number;
  constraint_match_rate?: number;
  json_valid_rate?: number;
  vector_mae?: number;
  mse_raw?: number;
  mse_norm?: number;
  timestamp: string;
};

export type SubmitCompositionResponse = {
  score: number;
  rank: Rank;
  coachFeedbackBySlot: Record<SlotKey, string>;
  rewardMoney: number;
  distanceBreakdown?: DistanceBreakdown;
  evalSnapshot?: EvalSnapshot;
};

export type CreateMusicRequest = {
  commissionId: string;
  requestText: string;
  weather?: Weather;
  selectedPartsBySlot: Record<SlotKey, string>;
  targetHiddenParams?: TargetHiddenParams;
};

export type CreateMusicResponse = {
  jobId: string;
  rulePrompt?: string;
};

export type MusicJobStatusResponse = {
  status: MusicJobStatus;
  audioUrl?: string;
  error?: string;
  rulePrompt?: string;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  outputSanityScore?: number;
  promptInferenceHiddenParams?: TargetHiddenParams;
  promptInferenceMeta?: InterpreterEvaluationMeta;
  promptEval?: PromptHiddenParamEval;
  promptFeedback?: string;
  traceId?: string;
};

export type Commission = {
  id: string;
  customerId: string;
  requestText: string;
  weather: Weather;
  status: CommissionStatus;
  targetProfile?: TargetProfile;
  targetHiddenParams?: TargetHiddenParams;
  interpreterHiddenParams?: TargetHiddenParams;
  interpreterOutput?: InterpreterResponse;
  generationSource?: "baseline" | "prompt_baseline" | "ft_model";
  requestGenerationSource?: "template" | "ft_model" | "fallback";
  requestGenerationTraceId?: string;
  requestGenerationParseError?: string;
  traceId?: string;
  selectedPartsBySlot?: Record<SlotKey, string>;
  score?: number;
  rank?: Rank;
  rewardMoney?: number;
  coachFeedbackBySlot?: Record<SlotKey, string>;
  distanceBreakdown?: DistanceBreakdown;
  evalSnapshot?: EvalSnapshot;
  jobId?: string;
  trackId?: string;
  rulePrompt?: string;
  promptInferenceHiddenParams?: TargetHiddenParams;
  promptInferenceMeta?: InterpreterEvaluationMeta;
  promptEval?: PromptHiddenParamEval;
  promptFeedback?: string;
  createdAt: string;
  updatedAt: string;
};

export type Track = {
  id: string;
  commissionId: string;
  audioUrl: string;
  usedPartsBySlot: Record<SlotKey, string>;
  score: number;
  rank: Rank;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  outputSanityScore?: number;
  rulePrompt?: string;
  promptInferenceHiddenParams?: TargetHiddenParams;
  promptInferenceMeta?: InterpreterEvaluationMeta;
  promptEval?: PromptHiddenParamEval;
  promptFeedback?: string;
  traceId?: string;
  createdAt: string;
};

export type MusicJob = {
  id: string;
  commissionId: string;
  requestText: string;
  selectedPartsBySlot: Record<SlotKey, string>;
  status: MusicJobStatus;
  audioUrl?: string;
  error?: string;
  rulePrompt?: string;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  outputSanityScore?: number;
  promptInferenceHiddenParams?: TargetHiddenParams;
  promptInferenceMeta?: InterpreterEvaluationMeta;
  promptEval?: PromptHiddenParamEval;
  promptFeedback?: string;
  traceId?: string;
  createdAt: string;
  updatedAt: string;
};

export type ShopStockItem = {
  partId: string;
  unlocked: boolean;
};

export type GameState = {
  schemaVersion: number;
  money: number;
  day: number;
  weather: Weather;
  inventoryPartIds: string[];
  commissions: Record<string, Commission>;
  commissionOrder: string[];
  tracks: Record<string, Track>;
  trackOrder: string[];
  jobs: Record<string, MusicJob>;
  shopStock: ShopStockItem[];
  selectedCommissionId?: string;
  updatedAt: string;
};

export type ScoreBreakdown = {
  vectorScore: number;
  tagScore: number;
  preferenceScore: number;
  synergyScore: number;
  antiSynergyPenalty: number;
  total: number;
};

export type HiddenParamsEvalMetrics = {
  json_valid_rate: number;
  vector_mae: number;
  mse_raw: number;
  mse_norm: number;
  r2_score: number;
};

export type ConstraintEvalMetrics = {
  constraint_match_rate: number;
  slot_exact_match: number;
};

export type PerfCostMetrics = {
  p50_inference_latency_ms: number;
  p95_inference_latency_ms: number;
  cost_per_100_requests_usd: number;
};
