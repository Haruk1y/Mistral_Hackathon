import { nanoid } from "nanoid";
import { CATALOG_PARTS, REQUEST_TEMPLATES } from "@/lib/catalog";
import type {
  InterpreterRequest,
  InterpreterResponse,
  Part,
  SlotKey,
  TargetHiddenParams,
  TargetProfile
} from "@/lib/types";
import { SLOT_KEYS } from "@/lib/types";
import { interpreterResponseSchema } from "@/lib/schemas";

const defaultProfile: TargetProfile = {
  vector: {
    energy: 45,
    warmth: 55,
    brightness: 50,
    acousticness: 65,
    complexity: 38,
    nostalgia: 60
  },
  requiredTags: ["nostalgic"],
  optionalTags: ["cozy", "acoustic"],
  forbiddenTags: [],
  constraints: {
    preferredStyleTags: ["citypop_80s"],
    preferredGimmickTags: ["beat_mute"]
  }
};

const keywordProfiles: Array<{ keywords: string[]; profile: TargetProfile }> = [
  {
    keywords: ["rain", "雨", "quiet", "静か", "evening", "夜"],
    profile: REQUEST_TEMPLATES[0].targetProfile
  },
  {
    keywords: ["smile", "bright", "楽しい", "元気", "market"],
    profile: REQUEST_TEMPLATES[1].targetProfile
  },
  {
    keywords: ["focus", "study", "集中", "勉強"],
    profile: REQUEST_TEMPLATES[2].targetProfile
  },
  {
    keywords: ["memory", "nostalgia", "懐か", "old town", "思い出"],
    profile: REQUEST_TEMPLATES[3].targetProfile
  }
];

const toHiddenParams = (profile: TargetProfile): TargetHiddenParams => ({
  vector: profile.vector,
  tags: [...new Set([...profile.requiredTags, ...profile.optionalTags])],
  constraints: {
    ...profile.constraints
  }
});

const pickProfile = (requestText: string): TargetProfile => {
  const lower = requestText.toLowerCase();
  const ranked = keywordProfiles
    .map((entry) => ({
      entry,
      score: entry.keywords.filter((keyword) => lower.includes(keyword.toLowerCase())).length
    }))
    .sort((a, b) => b.score - a.score);

  if (ranked[0]?.score > 0) {
    return ranked[0].entry.profile;
  }

  return defaultProfile;
};

const scorePartForProfile = (part: Part, profile: TargetProfile) => {
  let score = 0;

  for (const tag of part.tags) {
    if (profile.requiredTags.includes(tag)) score += 4;
    if (profile.optionalTags.includes(tag)) score += 2;
    if (profile.forbiddenTags.includes(tag)) score -= 6;
  }

  if (part.slot === "style" && profile.constraints.preferredStyleTags?.some((tag) => part.tags.includes(tag))) {
    score += 4;
  }

  if (
    part.slot === "gimmick" &&
    profile.constraints.preferredGimmickTags?.some((tag) => part.tags.includes(tag))
  ) {
    score += 4;
  }

  if (profile.constraints.avoidPartIds?.includes(part.id)) {
    score -= 8;
  }

  return score;
};

const buildResponseFromProfile = (
  input: InterpreterRequest,
  profile: TargetProfile,
  modelSource: InterpreterResponse["evaluationMeta"]["model_source"],
  latencyMs: number,
  parseError?: string
): InterpreterResponse => {
  const inventory = CATALOG_PARTS.filter((part) => input.inventoryPartIds.includes(part.id));

  const recommended = Object.fromEntries(
    SLOT_KEYS.map((slot) => {
      const candidates = inventory
        .filter((part) => part.slot === slot)
        .sort((a, b) => scorePartForProfile(b, profile) - scorePartForProfile(a, profile))
        .slice(0, 3)
        .map((part) => part.id);

      const fallback = CATALOG_PARTS.filter((part) => part.slot === slot)
        .slice(0, 3)
        .map((part) => part.id);

      return [slot, candidates.length ? candidates : fallback];
    })
  ) as Record<SlotKey, string[]>;

  return {
    recommended,
    targetProfile: profile,
    targetHiddenParams: toHiddenParams(profile),
    hintToPlayer: "STYLEで文法を決め、MOODとGIMMICKで短尺フックを作ると成功率が上がります。",
    rationale: [
      `weather=${input.weather} を加味して targetProfile を選定しました。`,
      "requiredTags に一致するパーツを優先して候補化しました。",
      "forbiddenTags は減点し、上位3候補を返しています。"
    ],
    evaluationMeta: {
      json_valid: true,
      parse_error: parseError,
      model_source: modelSource,
      latency_ms: latencyMs,
      trace_id: nanoid(),
      cost_usd: 0
    }
  };
};

const extractJsonBlock = (text: string): string | null => {
  const fenced = text.match(/```json\s*([\s\S]*?)```/i);
  if (fenced?.[1]) return fenced[1].trim();

  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    return text.slice(firstBrace, lastBrace + 1);
  }

  return null;
};

const resolveHfModelConfig = () => {
  const run1ModelId = process.env.HF_RUN1_MODEL_ID?.trim();
  const fineTunedModelId = process.env.MISTRAL_FINE_TUNED_MODEL_ID?.trim();
  const defaultModelId = process.env.HF_MISTRAL_MODEL_ID?.trim();

  if (run1ModelId) {
    return {
      modelId: run1ModelId,
      modelSource: "fine_tuned" as const
    };
  }

  if (fineTunedModelId) {
    return {
      modelId: fineTunedModelId,
      modelSource: "fine_tuned" as const
    };
  }

  return {
    modelId: defaultModelId || "mistralai/Ministral-3-3B-Instruct-2512",
    modelSource: "prompt_baseline" as const
  };
};

const callHfModel = async (input: InterpreterRequest): Promise<InterpreterResponse | null> => {
  const { modelId: hfModelId, modelSource } = resolveHfModelConfig();
  const hfToken = process.env.HF_API_TOKEN || process.env.HF_TOKEN;
  const hfInferenceBaseUrl =
    (process.env.HF_INFERENCE_BASE_URL || "https://router.huggingface.co/hf-inference/models").replace(/\/$/, "");

  if (!hfToken) {
    return null;
  }

  const startedAt = performance.now();
  const prompt = [
    "You are Request Interpreter for a music composition game.",
    "Return strict JSON only.",
    "Schema:",
    '{"recommended":{"style":["part_id"],"instrument":["part_id"],"mood":["part_id"],"gimmick":["part_id"]},"targetProfile":{"vector":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0},"requiredTags":[],"optionalTags":[],"forbiddenTags":[],"constraints":{"preferredStyleTags":[],"preferredGimmickTags":[],"avoidPartIds":[]}},"targetHiddenParams":{"vector":{"energy":0,"warmth":0,"brightness":0,"acousticness":0,"complexity":0,"nostalgia":0},"tags":[],"constraints":{}},"hintToPlayer":"","rationale":[],"evaluationMeta":{"json_valid":true,"model_source":"prompt_baseline"}}',
    `requestText=${input.requestText}`,
    `weather=${input.weather}`,
    `inventoryPartIds=${input.inventoryPartIds.join(",")}`,
    "Use only provided inventory part IDs in recommended arrays."
  ].join("\n");

  const response = await fetch(`${hfInferenceBaseUrl}/${hfModelId}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${hfToken}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      inputs: prompt,
      parameters: {
        max_new_tokens: 550,
        temperature: 0.2,
        return_full_text: false
      }
    })
  });

  if (!response.ok) {
    return null;
  }

  const payload = await response.json();
  const text = Array.isArray(payload)
    ? payload[0]?.generated_text || ""
    : payload.generated_text || payload?.[0]?.generated_text || "";

  if (!text) {
    return null;
  }

  const jsonBlock = extractJsonBlock(text);
  if (!jsonBlock) {
    return null;
  }

  try {
    const parsed = JSON.parse(jsonBlock) as Partial<InterpreterResponse>;

    if (!parsed.targetHiddenParams && parsed.targetProfile) {
      parsed.targetHiddenParams = toHiddenParams(parsed.targetProfile);
    }

    const latency = Math.max(0, performance.now() - startedAt);
    parsed.evaluationMeta = {
      json_valid: true,
      model_source: modelSource,
      latency_ms: latency,
      trace_id: nanoid(),
      cost_usd: 0
    };

    const validated = interpreterResponseSchema.safeParse(parsed);
    if (!validated.success) {
      return null;
    }

    return validated.data;
  } catch {
    return null;
  }
};

export const runInterpreter = async (input: InterpreterRequest): Promise<InterpreterResponse> => {
  const startedAt = performance.now();
  const hfResponse = await callHfModel(input);
  if (hfResponse) {
    return hfResponse;
  }

  const profile = pickProfile(input.requestText);
  const latencyMs = Math.max(0, performance.now() - startedAt);

  return buildResponseFromProfile(input, profile, "rule_baseline", latencyMs);
};
