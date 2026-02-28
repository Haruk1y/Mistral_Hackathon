import { nanoid } from "nanoid";
import type { RequestGenerationRequest, RequestGenerationResponse } from "@/lib/types";

const resolveModelConfig = () => {
  const run1ModelId = process.env.HF_RUN1_MODEL_ID?.trim();
  const fineTunedModelId = process.env.MISTRAL_FINE_TUNED_MODEL_ID?.trim();

  if (run1ModelId) {
    return { modelId: run1ModelId, modelSource: "fine_tuned" as const };
  }

  if (fineTunedModelId) {
    return { modelId: fineTunedModelId, modelSource: "fine_tuned" as const };
  }

  return null;
};

const extractText = (payload: unknown): string => {
  if (Array.isArray(payload)) {
    const first = payload[0];
    if (first && typeof first === "object" && "generated_text" in first) {
      return String((first as { generated_text?: unknown }).generated_text ?? "");
    }
  }

  if (payload && typeof payload === "object") {
    const objectPayload = payload as Record<string, unknown>;
    if (typeof objectPayload.generated_text === "string") return objectPayload.generated_text;
    if (Array.isArray(objectPayload.choices) && objectPayload.choices[0]) {
      const firstChoice = objectPayload.choices[0] as Record<string, unknown>;
      if (typeof firstChoice.text === "string") return firstChoice.text;
      if (firstChoice.message && typeof firstChoice.message === "object") {
        const message = firstChoice.message as Record<string, unknown>;
        if (typeof message.content === "string") return message.content;
      }
    }
  }

  return "";
};

const DEBUG_MARKER_REGEX =
  /(requiredtags|forbiddentags|optionaltags|hinttoplayer|targetprofile|constraints|rationale|json)/i;

const splitSentenceCandidates = (input: string): string[] => {
  return input
    .split(/[\r\n]+/)
    .flatMap((line) => line.match(/[^.!?。！？]+[.!?。！？]?/g) ?? [])
    .map((item) =>
      item
        .trim()
        .replace(/^["'`]+|["'`]+$/g, "")
        .replace(/^request\s*[:\-]\s*/i, "")
        .replace(/^\s*[-*•]\s*/u, "")
        .replace(/\s+/g, " ")
        .trim()
    )
    .filter(Boolean);
};

const sanitizeOneLineRequest = (rawText: string, fallback: string): string => {
  const compactFallback = fallback.replace(/\s+/g, " ").trim();
  const candidates = splitSentenceCandidates(rawText);
  const safeCandidate =
    candidates.find((candidate) => candidate.length >= 8 && !DEBUG_MARKER_REGEX.test(candidate)) ??
    candidates.find((candidate) => candidate.length >= 8) ??
    "";
  const selected = safeCandidate || compactFallback;
  if (selected.length <= 160) return selected;
  return `${selected.slice(0, 157).trimEnd()}...`;
};

const normalizeComparable = (input: string): string =>
  input
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const forceParaphraseTemplate = (templateText: string, weather: RequestGenerationRequest["weather"]): string => {
  const base = templateText
    .replace(/^please\s+/i, "")
    .replace(/^can you\s+/i, "")
    .replace(/^could you\s+/i, "")
    .trim()
    .replace(/\.$/, "");

  const weatherHint =
    weather === "rainy" ? "with a rainy-evening touch" : weather === "sunny" ? "with a soft daylight feel" : "";

  const sentence = `Could you ${base}${weatherHint ? ` ${weatherHint}` : ""}?`
    .replace(/\s+/g, " ")
    .trim();

  return sentence.length <= 160 ? sentence : `${sentence.slice(0, 157).trimEnd()}...`;
};

export const generateRequestText = async (
  input: RequestGenerationRequest
): Promise<RequestGenerationResponse> => {
  const startedAt = performance.now();
  const hfToken = process.env.HF_API_TOKEN || process.env.HF_TOKEN;
  const modelConfig = resolveModelConfig();
  const fallbackParaphrase = forceParaphraseTemplate(input.templateText, input.weather);
  const hfInferenceBaseUrl =
    (process.env.HF_INFERENCE_BASE_URL || "https://router.huggingface.co/hf-inference/models").replace(/\/$/, "");

  if (!modelConfig) {
    return {
      requestText: fallbackParaphrase,
      modelSource: "rule_baseline",
      latencyMs: Math.max(0, performance.now() - startedAt),
      parseError: "missing_fine_tuned_model_id",
      traceId: `request_${nanoid()}`
    };
  }

  if (!hfToken) {
    return {
      requestText: fallbackParaphrase,
      modelSource: "rule_baseline",
      latencyMs: Math.max(0, performance.now() - startedAt),
      parseError: "missing_hf_token",
      traceId: `request_${nanoid()}`
    };
  }

  const { modelId, modelSource } = modelConfig;
  const callInference = async (prompt: string) => {
    const response = await fetch(`${hfInferenceBaseUrl}/${modelId}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${hfToken}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          max_new_tokens: 56,
          temperature: 0.65,
          return_full_text: false
        }
      })
    });

    if (!response.ok) {
      const errorText = (await response.text()).slice(0, 220);
      return { ok: false as const, parseError: `http_${response.status}:${errorText}` };
    }

    const payload = await response.json();
    const generated = extractText(payload);
    return { ok: true as const, generated };
  };

  const basePrompt = [
    "You write one short customer request for a cozy retro game music commission.",
    "Output exactly one sentence in English.",
    "Do not output explanations or lists.",
    "Do not copy the base template verbatim. Rephrase it in natural spoken style.",
    "Keep it between 12 and 26 words.",
    `Base template: ${input.templateText}`,
    `Weather: ${input.weather}`,
    `Customer: ${input.customerName || input.customerId}`,
    `Customer vibe: ${input.customerPersonality || "warm and human"}`
  ].join("\n");

  try {
    const first = await callInference(basePrompt);
    if (!first.ok) {
      return {
        requestText: fallbackParaphrase,
        modelSource: "rule_baseline",
        latencyMs: Math.max(0, performance.now() - startedAt),
        parseError: first.parseError,
        traceId: `request_${nanoid()}`
      };
    }

    let requestText = sanitizeOneLineRequest(first.generated, input.templateText);
    let parseError: string | undefined;

    if (normalizeComparable(requestText) === normalizeComparable(input.templateText)) {
      const rewritePrompt = [
        "Rewrite this customer request in different wording.",
        "Output exactly one sentence in English.",
        "Do not copy any phrase directly from the original sentence.",
        "Keep intent, constraints, and mood the same.",
        `Original sentence: ${input.templateText}`,
        `Weather: ${input.weather}`,
        `Customer vibe: ${input.customerPersonality || "warm and human"}`
      ].join("\n");

      const second = await callInference(rewritePrompt);
      if (second.ok) {
        requestText = sanitizeOneLineRequest(second.generated, input.templateText);
      } else {
        parseError = `rewrite_failed:${second.parseError}`;
      }
    }

    if (normalizeComparable(requestText) === normalizeComparable(input.templateText)) {
      requestText = forceParaphraseTemplate(input.templateText, input.weather);
      parseError = parseError ? `${parseError};template_like_output_paraphrased` : "template_like_output_paraphrased";
    }

    return {
      requestText,
      modelSource,
      latencyMs: Math.max(0, performance.now() - startedAt),
      parseError,
      traceId: `request_${nanoid()}`
    };
  } catch {
    return {
      requestText: fallbackParaphrase,
      modelSource: "rule_baseline",
      latencyMs: Math.max(0, performance.now() - startedAt),
      parseError: "network_or_parse_error",
      traceId: `request_${nanoid()}`
    };
  }
};
