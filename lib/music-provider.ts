import { nanoid } from "nanoid";
import type { CreateMusicRequest } from "@/lib/types";
import { CATALOG_PARTS } from "@/lib/catalog";

export type ElevenLabsDetailedResult = {
  audioUrl: string | null;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  outputSanityScore: number;
  traceId: string;
};

const getPartName = (partId: string) => CATALOG_PARTS.find((part) => part.id === partId)?.name ?? partId;

const toInt = (value: string | undefined, fallback: number) => {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const pickObject = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object") return null;
  return value as Record<string, unknown>;
};

const pickBoolean = (value: unknown): boolean | undefined => {
  if (typeof value === "boolean") return value;
  return undefined;
};

const pickNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
};

const pickString = (value: unknown): string | undefined => {
  if (typeof value === "string" && value.length > 0) return value;
  return undefined;
};

const normalizeAudioUrl = async (response: Response, payload: unknown): Promise<string | null> => {
  if (!response.ok) return null;

  const contentType = response.headers.get("content-type") || "";
  if (contentType.startsWith("audio/")) {
    const buffer = Buffer.from(await response.arrayBuffer());
    return `data:${contentType};base64,${buffer.toString("base64")}`;
  }

  const objectPayload = pickObject(payload);
  if (!objectPayload) return null;

  const nestedData = pickObject(objectPayload.data);

  const candidates = [
    pickString(objectPayload.audio_url),
    pickString(objectPayload.audioUrl),
    pickString(objectPayload.url),
    nestedData ? pickString(nestedData.audio_url) : undefined,
    nestedData ? pickString(nestedData.audioUrl) : undefined,
    nestedData ? pickString(nestedData.url) : undefined
  ];

  return candidates.find((candidate) => Boolean(candidate)) ?? null;
};

const extractCompositionPlan = (payload: unknown): unknown => {
  const objectPayload = pickObject(payload);
  if (!objectPayload) return undefined;

  return (
    objectPayload.composition_plan ??
    objectPayload.compositionPlan ??
    pickObject(objectPayload.data)?.composition_plan ??
    pickObject(objectPayload.data)?.compositionPlan
  );
};

const extractSongMetadata = (payload: unknown): unknown => {
  const objectPayload = pickObject(payload);
  if (!objectPayload) return undefined;

  return (
    objectPayload.song_metadata ??
    objectPayload.songMetadata ??
    pickObject(objectPayload.data)?.song_metadata ??
    pickObject(objectPayload.data)?.songMetadata
  );
};

const extractTraceId = (payload: unknown): string | undefined => {
  const objectPayload = pickObject(payload);
  if (!objectPayload) return undefined;

  const nestedData = pickObject(objectPayload.data);
  const metadata = pickObject(extractSongMetadata(payload));

  return (
    pickString(objectPayload.trace_id) ??
    pickString(objectPayload.traceId) ??
    pickString(nestedData?.trace_id) ??
    pickString(nestedData?.traceId) ??
    pickString(metadata?.trace_id) ??
    pickString(metadata?.traceId)
  );
};

const extractDurationMs = (songMetadata: unknown): number | undefined => {
  const metadata = pickObject(songMetadata);
  if (!metadata) return undefined;

  return (
    pickNumber(metadata.music_length_ms) ??
    pickNumber(metadata.duration_ms) ??
    pickNumber(metadata.length_ms)
  );
};

export const computeOutputSanityScore = (input: {
  audioUrl: string | null;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  musicLengthMs: number;
  forceInstrumental: boolean;
}): number => {
  let score = 0;

  if (input.audioUrl) score += 20;
  if (input.compositionPlan && typeof input.compositionPlan === "object") score += 30;
  if (input.songMetadata && typeof input.songMetadata === "object") score += 30;

  const metadata = pickObject(input.songMetadata);
  const plan = pickObject(input.compositionPlan);

  if (input.forceInstrumental) {
    const instrumental =
      pickBoolean(metadata?.instrumental) ??
      pickBoolean(metadata?.force_instrumental) ??
      pickBoolean(plan?.force_instrumental);

    if (instrumental === true) {
      score += 10;
    }
  }

  const durationMs = extractDurationMs(input.songMetadata);
  if (typeof durationMs === "number") {
    const diff = Math.abs(durationMs - input.musicLengthMs);
    if (diff <= 3000) {
      score += 10;
    } else if (diff <= 8000) {
      score += 5;
    }
  }

  return clamp(score, 0, 100);
};

export const createSinePrompt = (payload: CreateMusicRequest): string => {
  const hiddenParamsSummary = payload.targetHiddenParams
    ? `Target hidden params: energy=${payload.targetHiddenParams.vector.energy}, warmth=${payload.targetHiddenParams.vector.warmth}, brightness=${payload.targetHiddenParams.vector.brightness}, acousticness=${payload.targetHiddenParams.vector.acousticness}, complexity=${payload.targetHiddenParams.vector.complexity}, nostalgia=${payload.targetHiddenParams.vector.nostalgia}`
    : "Target hidden params: unknown";

  const lines = [
    "Compose nostalgic retro pixel-town background music.",
    "Return instrumental music suitable for a game scene.",
    `User request: ${payload.requestText}`,
    hiddenParamsSummary,
    "Selected composition parts:",
    ...Object.entries(payload.selectedPartsBySlot).map(
      ([slot, partId]) => `- ${slot}: ${getPartName(partId)} (${partId})`
    ),
    "Style: warm, cozy, handcrafted, street evening, non-vocal, emotional but simple."
  ];

  return lines.join("\n");
};

export const generateMusicWithElevenLabs = async (
  payload: CreateMusicRequest
): Promise<ElevenLabsDetailedResult | null> => {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return null;
  }

  const endpoint =
    process.env.ELEVENLABS_MUSIC_DETAILED_ENDPOINT ||
    process.env.ELEVENLABS_MUSIC_ENDPOINT ||
    "https://api.elevenlabs.io/v1/music/compose-detailed";

  const modelId = process.env.ELEVENLABS_MODEL_ID || "music_v1";
  const musicLengthMs = clamp(toInt(process.env.ELEVENLABS_MUSIC_LENGTH_MS, 30000), 3000, 600000);
  const forceInstrumental = (process.env.ELEVENLABS_FORCE_INSTRUMENTAL || "true") !== "false";
  const prompt = createSinePrompt(payload);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 45_000);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": apiKey
      },
      body: JSON.stringify({
        model_id: modelId,
        prompt,
        force_instrumental: forceInstrumental,
        music_length_ms: musicLengthMs,
        output_format: "mp3_44100_128"
      }),
      signal: controller.signal
    });

    if (!response.ok) {
      return null;
    }

    const contentType = response.headers.get("content-type") || "";
    const rawPayload: unknown = contentType.includes("application/json") ? await response.json() : null;

    const audioUrl = await normalizeAudioUrl(response, rawPayload);
    const compositionPlan = extractCompositionPlan(rawPayload);
    const songMetadata = extractSongMetadata(rawPayload);
    const traceId = extractTraceId(rawPayload) ?? `music_${nanoid()}`;

    return {
      audioUrl,
      compositionPlan,
      songMetadata,
      outputSanityScore: computeOutputSanityScore({
        audioUrl,
        compositionPlan,
        songMetadata,
        musicLengthMs,
        forceInstrumental
      }),
      traceId
    };
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
};
