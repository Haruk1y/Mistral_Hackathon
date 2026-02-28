import { CATALOG_PARTS } from "@/lib/catalog";
import type {
  DistanceBreakdown,
  Part,
  Rank,
  ScoreBreakdown,
  SlotKey,
  TargetProfile
} from "@/lib/types";

const MAX_VECTOR_DISTANCE = Math.sqrt(6 * 100 * 100);

const getPart = (partId: string): Part | undefined => CATALOG_PARTS.find((part) => part.id === partId);

export const ensureSlotCategoryIntegrity = (selectedPartsBySlot: Record<SlotKey, string>): string[] => {
  const errors: string[] = [];

  for (const [slot, partId] of Object.entries(selectedPartsBySlot) as Array<[SlotKey, string]>) {
    const part = getPart(partId);
    if (!part) {
      errors.push(`Part not found: ${partId}`);
      continue;
    }

    if (part.slot !== slot) {
      errors.push(`Slot mismatch for ${slot}: received ${part.slot}`);
    }
  }

  return errors;
};

export const composeVectorFromParts = (selectedPartIds: string[]) => {
  const aggregate = {
    energy: 50,
    warmth: 50,
    brightness: 50,
    acousticness: 50,
    complexity: 50,
    nostalgia: 50
  };

  selectedPartIds
    .map((partId) => getPart(partId))
    .filter((part): part is Part => Boolean(part))
    .forEach((part) => {
      for (const [key, value] of Object.entries(part.vector) as Array<[keyof typeof aggregate, number]>) {
        aggregate[key] = Math.max(0, Math.min(100, Math.round((aggregate[key] + value) / 2)));
      }
    });

  return aggregate;
};

export const calculateVectorDistance = (
  source: ReturnType<typeof composeVectorFromParts>,
  target: TargetProfile["vector"]
): number => {
  const components: Array<keyof TargetProfile["vector"]> = [
    "energy",
    "warmth",
    "brightness",
    "acousticness",
    "complexity",
    "nostalgia"
  ];

  const squareSum = components
    .map((key) => Math.pow(source[key] - target[key], 2))
    .reduce((total, value) => total + value, 0);

  return Math.sqrt(squareSum);
};

const buildTagSet = (selectedPartIds: string[]): Set<string> => {
  const tags = selectedPartIds
    .map((partId) => getPart(partId))
    .filter((part): part is Part => Boolean(part))
    .flatMap((part) => part.tags);

  return new Set(tags);
};

const scoreTagAlignment = (targetProfile: TargetProfile, tags: Set<string>): number => {
  const requiredHits = targetProfile.requiredTags.filter((tag) => tags.has(tag)).length;
  const optionalHits = targetProfile.optionalTags.filter((tag) => tags.has(tag)).length;
  const forbiddenHits = targetProfile.forbiddenTags.filter((tag) => tags.has(tag)).length;

  const requiredScore = targetProfile.requiredTags.length
    ? (requiredHits / targetProfile.requiredTags.length) * 18
    : 18;
  const optionalScore = targetProfile.optionalTags.length
    ? (optionalHits / targetProfile.optionalTags.length) * 7
    : 7;
  const penalty = Math.min(10, forbiddenHits * 4);

  return Math.max(0, Math.min(25, Math.round(requiredScore + optionalScore - penalty)));
};

const hasPreferredTag = (part: Part | undefined, preferredTags: string[] | undefined): boolean => {
  if (!part || !preferredTags?.length) return false;
  return preferredTags.some((tag) => part.tags.includes(tag));
};

const scorePreferences = (
  selectedPartsBySlot: Record<SlotKey, string>,
  targetProfile: TargetProfile
): number => {
  let result = 0;

  const stylePart = getPart(selectedPartsBySlot.style);
  const gimmickPart = getPart(selectedPartsBySlot.gimmick);

  if (targetProfile.constraints.preferredStyleTags?.length) {
    result += hasPreferredTag(stylePart, targetProfile.constraints.preferredStyleTags) ? 6 : -4;
  }

  if (targetProfile.constraints.preferredGimmickTags?.length) {
    result += hasPreferredTag(gimmickPart, targetProfile.constraints.preferredGimmickTags) ? 4 : -3;
  }

  if (targetProfile.constraints.avoidPartIds?.length) {
    const avoidHits = Object.values(selectedPartsBySlot).filter((id) =>
      targetProfile.constraints.avoidPartIds?.includes(id)
    ).length;
    result -= Math.min(6, avoidHits * 3);
  }

  return Math.max(-10, Math.min(10, result));
};

const scoreSynergy = (selectedPartIds: string[]): { synergyScore: number; antiSynergyPenalty: number } => {
  const set = new Set(selectedPartIds);
  let synergyScore = 0;
  let antiSynergyPenalty = 0;

  const synergyCombos: string[][] = [
    ["style_80s_citypop", "inst_piano_upright", "mood_rain_ambience", "gimmick_beat_mute"],
    ["style_2000s_pop", "mood_sun_glow", "gimmick_filter_rise"],
    ["style_90s_hiphop", "inst_analog_synth", "mood_night_drive"]
  ];

  for (const combo of synergyCombos) {
    if (combo.every((id) => set.has(id))) {
      synergyScore += 4;
    }
  }

  const antiCombos: string[][] = [
    ["style_2000s_pop", "mood_rain_ambience"],
    ["style_90s_hiphop", "inst_soft_strings", "gimmick_harmony_stack"]
  ];

  for (const combo of antiCombos) {
    if (combo.every((id) => set.has(id))) {
      antiSynergyPenalty -= 4;
    }
  }

  return {
    synergyScore: Math.min(10, synergyScore),
    antiSynergyPenalty: Math.max(-10, antiSynergyPenalty)
  };
};

export const calculateRank = (score: number): Rank => {
  if (score >= 90) return "S";
  if (score >= 75) return "A";
  if (score >= 60) return "B";
  if (score >= 40) return "C";
  return "D";
};

export const scoreComposition = (
  selectedPartsBySlot: Record<SlotKey, string>,
  targetProfile: TargetProfile
): ScoreBreakdown => {
  const selectedPartIds = Object.values(selectedPartsBySlot);
  const vector = composeVectorFromParts(selectedPartIds);
  const tags = buildTagSet(selectedPartIds);

  const distance = calculateVectorDistance(vector, targetProfile.vector);
  const vectorScore = Math.max(0, Math.round((1 - distance / MAX_VECTOR_DISTANCE) * 60));
  const tagScore = scoreTagAlignment(targetProfile, tags);
  const preferenceScore = scorePreferences(selectedPartsBySlot, targetProfile);
  const { synergyScore, antiSynergyPenalty } = scoreSynergy(selectedPartIds);

  const total = Math.max(
    0,
    Math.min(100, vectorScore + tagScore + preferenceScore + synergyScore + antiSynergyPenalty)
  );

  return {
    vectorScore,
    tagScore,
    preferenceScore,
    synergyScore,
    antiSynergyPenalty,
    total
  };
};

export const buildDistanceBreakdown = (
  selectedPartsBySlot: Record<SlotKey, string>,
  targetProfile: TargetProfile
): DistanceBreakdown => {
  const selectedPartIds = Object.values(selectedPartsBySlot);
  const vector = composeVectorFromParts(selectedPartIds);
  const tags = buildTagSet(selectedPartIds);
  const vectorDistance = calculateVectorDistance(vector, targetProfile.vector);
  const normalizedDistance = Math.max(0, Math.min(1, vectorDistance / MAX_VECTOR_DISTANCE));
  const vectorScore = Math.max(0, Math.round((1 - normalizedDistance) * 60));
  const tagScore = scoreTagAlignment(targetProfile, tags);
  const preferenceScore = scorePreferences(selectedPartsBySlot, targetProfile);
  const { synergyScore, antiSynergyPenalty } = scoreSynergy(selectedPartIds);

  return {
    vectorDistance,
    normalizedDistance,
    vectorScore,
    tagScore,
    preferenceScore,
    synergyScore,
    antiSynergyPenalty
  };
};

export const buildCoachFeedbackBySlot = (
  selectedPartsBySlot: Record<SlotKey, string>,
  targetProfile: TargetProfile
): Record<SlotKey, string> => {
  const parts = Object.fromEntries(
    Object.entries(selectedPartsBySlot).map(([slot, partId]) => [slot, getPart(partId)])
  ) as Record<SlotKey, Part | undefined>;

  return {
    style: targetProfile.constraints.preferredStyleTags?.length
      ? `スタイルは ${targetProfile.constraints.preferredStyleTags.join(" / ")} 系が有利。現在は ${parts.style?.name ?? "未設定"}。`
      : `スタイル ${parts.style?.name ?? "未設定"} は曲の文法を決めます。まず依頼の世界観に寄せましょう。`,
    instrument: `楽器 ${parts.instrument?.name ?? "未設定"} は音色の主役です。requiredTags(${targetProfile.requiredTags.join(", ")})に寄るタグを優先しましょう。`,
    mood: `ムード ${parts.mood?.name ?? "未設定"} は最終印象を決定します。forbiddenTags(${targetProfile.forbiddenTags.join(", ") || "なし"})に注意。`,
    gimmick: targetProfile.constraints.preferredGimmickTags?.length
      ? `ギミックは ${targetProfile.constraints.preferredGimmickTags.join(" / ")} 系が狙い。現在は ${parts.gimmick?.name ?? "未設定"}。`
      : `ギミック ${parts.gimmick?.name ?? "未設定"} は短尺でのフックを作る要素です。`
  };
};
