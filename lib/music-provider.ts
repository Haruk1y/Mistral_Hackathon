import type { CreateMusicRequest } from "@/lib/types";
import { CATALOG_PARTS } from "@/lib/catalog";

const getPartName = (partId: string) => CATALOG_PARTS.find((part) => part.id === partId)?.name ?? partId;

export const createSinePrompt = (payload: CreateMusicRequest): string => {
  const lines = [
    "Compose nostalgic retro pixel-town background music.",
    `User request: ${payload.requestText}`,
    "Selected composition parts:",
    ...Object.entries(payload.selectedPartsBySlot).map(
      ([slot, partId]) => `- ${slot}: ${getPartName(partId)} (${partId})`
    ),
    "Style: warm, cozy, handcrafted, street evening, non-vocal, emotional but simple."
  ];

  return lines.join("\n");
};

const normalizeAudioUrl = async (response: Response, payload: unknown): Promise<string | null> => {
  if (!response.ok) {
    return null;
  }

  const contentType = response.headers.get("content-type") || "";

  if (contentType.startsWith("audio/")) {
    const buffer = Buffer.from(await response.arrayBuffer());
    return `data:${contentType};base64,${buffer.toString("base64")}`;
  }

  if (payload && typeof payload === "object") {
    const objectPayload = payload as Record<string, unknown>;
    const candidate =
      (objectPayload.audio_url as string | undefined) ||
      (objectPayload.url as string | undefined) ||
      (objectPayload?.data && typeof objectPayload.data === "object"
        ? ((objectPayload.data as Record<string, unknown>).audio_url as string | undefined)
        : undefined);

    if (candidate) {
      return candidate;
    }
  }

  return null;
};

export const generateMusicWithElevenLabs = async (prompt: string): Promise<string | null> => {
  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return null;
  }

  const endpoint = process.env.ELEVENLABS_MUSIC_ENDPOINT || "https://api.elevenlabs.io/v1/music/compose";

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20_000);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": apiKey
      },
      body: JSON.stringify({
        prompt,
        duration_seconds: 30,
        format: "mp3"
      }),
      signal: controller.signal
    });

    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      const data = await response.json();
      return normalizeAudioUrl(response, data);
    }

    return normalizeAudioUrl(response, null);
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
};
