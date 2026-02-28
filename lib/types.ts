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
};

export type InterpreterResponse = {
  recommended: Record<SlotKey, string[]>;
  targetProfile: TargetProfile;
  hintToPlayer: string;
  rationale: string[];
};

export type SubmitCompositionRequest = {
  commissionId: string;
  selectedPartsBySlot: Record<SlotKey, string>;
};

export type SubmitCompositionResponse = {
  score: number;
  rank: Rank;
  coachFeedbackBySlot: Record<SlotKey, string>;
  rewardMoney: number;
};

export type CreateMusicRequest = {
  commissionId: string;
  requestText: string;
  selectedPartsBySlot: Record<SlotKey, string>;
};

export type CreateMusicResponse = {
  jobId: string;
};

export type MusicJobStatusResponse = {
  status: MusicJobStatus;
  audioUrl?: string;
  error?: string;
};

export type Commission = {
  id: string;
  customerId: string;
  requestText: string;
  weather: Weather;
  status: CommissionStatus;
  targetProfile?: TargetProfile;
  interpreterOutput?: InterpreterResponse;
  selectedPartsBySlot?: Record<SlotKey, string>;
  score?: number;
  rank?: Rank;
  rewardMoney?: number;
  coachFeedbackBySlot?: Record<SlotKey, string>;
  jobId?: string;
  trackId?: string;
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
