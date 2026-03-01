import type { ProfileVector, TargetHiddenParams, TargetProfile } from "@/lib/types";

const VECTOR_KEYS: Array<keyof ProfileVector> = [
  "energy",
  "warmth",
  "brightness",
  "acousticness",
  "complexity",
  "nostalgia"
];

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const unique = (items: string[]) => [...new Set(items.filter((item) => item.trim().length > 0))];

export const normalizeProfileVector = (vector: Partial<Record<keyof ProfileVector, unknown>>): ProfileVector => {
  const values = VECTOR_KEYS.map((key) => {
    const raw = vector[key];
    const numeric = typeof raw === "number" ? raw : Number(raw);
    if (!Number.isFinite(numeric)) return 50;
    return clamp(Math.round(numeric), 0, 100);
  });

  return {
    energy: values[0],
    warmth: values[1],
    brightness: values[2],
    acousticness: values[3],
    complexity: values[4],
    nostalgia: values[5]
  };
};

const deriveRequiredTags = (vector: ProfileVector): string[] => {
  const tags: string[] = [];
  if (vector.nostalgia >= 65) tags.push("nostalgic");
  if (vector.acousticness >= 62) tags.push("acoustic");
  if (vector.warmth >= 60) tags.push("warm");
  if (vector.brightness >= 66) tags.push("bright");
  if (vector.energy >= 66) tags.push("lively");
  if (!tags.length) tags.push("nostalgic");
  return unique(tags);
};

const deriveOptionalTags = (vector: ProfileVector): string[] => {
  const tags: string[] = [];
  if (vector.warmth >= 58) tags.push("cozy");
  if (vector.energy <= 40) tags.push("quiet");
  if (vector.complexity >= 58) tags.push("emotional");
  if (vector.brightness <= 38) tags.push("night");
  return unique(tags);
};

const derivePreferredStyleTags = (vector: ProfileVector): string[] => {
  if (vector.nostalgia >= 70) return ["citypop_80s"];
  if (vector.brightness >= 70) return ["pop_2000s"];
  if (vector.energy >= 62) return ["hiphop_90s"];
  return ["citypop_80s"];
};

const derivePreferredGimmickTags = (vector: ProfileVector): string[] => {
  if (vector.energy >= 66) return ["filter_rise"];
  if (vector.complexity >= 56) return ["harmony_stack"];
  return ["beat_mute"];
};

const deriveForbiddenTags = (vector: ProfileVector): string[] => {
  const tags: string[] = [];
  if (vector.energy <= 28) tags.push("festival");
  return unique(tags);
};

export const buildTargetProfileFromVector = (
  vectorInput: Partial<Record<keyof ProfileVector, unknown>>,
  baseProfile?: TargetProfile
): TargetProfile => {
  const vector = normalizeProfileVector(vectorInput);
  const requiredTags = unique([...(baseProfile?.requiredTags ?? []), ...deriveRequiredTags(vector)]);
  const optionalTags = unique([...(baseProfile?.optionalTags ?? []), ...deriveOptionalTags(vector)]);
  const forbiddenTags = unique([...(baseProfile?.forbiddenTags ?? []), ...deriveForbiddenTags(vector)]);

  return {
    vector,
    requiredTags,
    optionalTags,
    forbiddenTags,
    constraints: {
      preferredStyleTags: derivePreferredStyleTags(vector),
      preferredGimmickTags: derivePreferredGimmickTags(vector),
      avoidPartIds: vector.brightness <= 25 ? ["style_2000s_pop"] : []
    }
  };
};

export const toTargetHiddenParams = (profile: TargetProfile): TargetHiddenParams => ({
  vector: normalizeProfileVector(profile.vector),
  tags: unique([...profile.requiredTags, ...profile.optionalTags]),
  constraints: {
    ...(profile.constraints ?? {})
  }
});
