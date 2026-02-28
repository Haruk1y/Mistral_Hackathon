import { nanoid } from "nanoid";
import type { CreateMusicRequest, MusicJobStatusResponse } from "@/lib/types";
import { runInterpreter } from "@/lib/interpreter";
import { computeOutputSanityScore, createSinePrompt, generateMusicWithElevenLabs } from "@/lib/music-provider";
import { buildPromptEvalFeedback, evaluatePromptHiddenParams } from "@/lib/prompt-hidden-param-eval";
import { generateSineWaveDataUri } from "@/lib/audio";

type InternalMusicJob = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  audioUrl?: string;
  error?: string;
  rulePrompt?: string;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  outputSanityScore?: number;
  promptInferenceHiddenParams?: CreateMusicRequest["targetHiddenParams"];
  promptInferenceMeta?: NonNullable<Awaited<ReturnType<typeof runInterpreter>>["evaluationMeta"]>;
  promptEval?: ReturnType<typeof evaluatePromptHiddenParams>;
  promptFeedback?: string;
  traceId?: string;
  attemptCount: number;
  createdAt: string;
  updatedAt: string;
};

declare global {
  // eslint-disable-next-line no-var
  var __otokotobaMusicJobs: Map<string, InternalMusicJob> | undefined;
}

const getJobs = () => {
  if (!global.__otokotobaMusicJobs) {
    global.__otokotobaMusicJobs = new Map<string, InternalMusicJob>();
  }

  return global.__otokotobaMusicJobs;
};

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const toInt = (value: string | undefined, fallback: number) => {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const toBool = (value: string | undefined, fallback: boolean) => {
  if (!value) return fallback;
  return ["1", "true", "yes", "on"].includes(value.toLowerCase());
};

const createFallbackResult = () => {
  const audioUrl = generateSineWaveDataUri();
  const musicLengthMs = Math.max(3000, toInt(process.env.ELEVENLABS_MUSIC_LENGTH_MS, 20000));

  return {
    audioUrl,
    outputSanityScore: computeOutputSanityScore({
      audioUrl,
      compositionPlan: undefined,
      songMetadata: undefined,
      musicLengthMs,
      forceInstrumental: (process.env.ELEVENLABS_FORCE_INSTRUMENTAL || "true") !== "false"
    }),
    traceId: `music_fallback_${nanoid()}`
  };
};

const processMusicJob = async (jobId: string, payload: CreateMusicRequest, rulePrompt: string) => {
  const jobs = getJobs();
  const target = jobs.get(jobId);
  if (!target) return;

  target.status = "running";
  target.updatedAt = new Date().toISOString();
  jobs.set(jobId, target);

  try {
    const maxRetries = Math.max(0, toInt(process.env.ELEVENLABS_MAX_RETRIES, 2));
    let result: Awaited<ReturnType<typeof generateMusicWithElevenLabs>> | null = null;
    let attempt = 0;

    while (attempt <= maxRetries) {
      attempt += 1;
      target.attemptCount = attempt;
      target.updatedAt = new Date().toISOString();
      jobs.set(jobId, { ...target });

      result = await generateMusicWithElevenLabs(payload, rulePrompt);
      if (result?.audioUrl) {
        break;
      }

      if (attempt <= maxRetries) {
        await wait(350 * attempt);
      }
    }

    await wait(1400);

    let promptInferenceHiddenParams: CreateMusicRequest["targetHiddenParams"] | undefined;
    let promptInferenceMeta: NonNullable<Awaited<ReturnType<typeof runInterpreter>>["evaluationMeta"]> | undefined;

    try {
      const promptInference = await runInterpreter({
        requestText: `RULE_PROMPT:\n${rulePrompt}`,
        weather: payload.weather ?? "cloudy",
        inventoryPartIds: Object.values(payload.selectedPartsBySlot)
      });

      promptInferenceHiddenParams = promptInference.targetHiddenParams;
      promptInferenceMeta = promptInference.evaluationMeta;
    } catch {
      promptInferenceHiddenParams = undefined;
      promptInferenceMeta = undefined;
    }

    const promptEval =
      payload.targetHiddenParams && promptInferenceHiddenParams
        ? evaluatePromptHiddenParams(payload.targetHiddenParams, promptInferenceHiddenParams)
        : undefined;
    const promptFeedback = buildPromptEvalFeedback(promptEval, payload.targetHiddenParams?.vector);
    const allowFallbackAudio = toBool(process.env.ELEVENLABS_ALLOW_FALLBACK_AUDIO, false);

    if (result?.audioUrl == null && !allowFallbackAudio) {
      jobs.set(jobId, {
        ...target,
        status: "failed",
        rulePrompt,
        promptInferenceHiddenParams,
        promptInferenceMeta,
        promptEval,
        promptFeedback,
        error: "ElevenLabs generation failed. Check endpoint/key/plan and disable fallback beep.",
        updatedAt: new Date().toISOString()
      });
      return;
    }

    const finalized =
      result?.audioUrl != null
        ? {
            audioUrl: result.audioUrl,
            rulePrompt: result.rulePrompt,
            compositionPlan: result.compositionPlan,
            songMetadata: result.songMetadata,
            outputSanityScore: result.outputSanityScore,
            promptInferenceHiddenParams,
            promptInferenceMeta,
            promptEval,
            promptFeedback,
            traceId: result.traceId,
            error: undefined
          }
        : {
            ...createFallbackResult(),
            rulePrompt,
            compositionPlan: undefined,
            songMetadata: undefined,
            promptInferenceHiddenParams,
            promptInferenceMeta,
            promptEval,
            promptFeedback,
            error: "ElevenLabs generation failed; fallback audio generated."
          };

    jobs.set(jobId, {
      ...target,
      status: "done",
      ...finalized,
      updatedAt: new Date().toISOString()
    });
  } catch (error) {
    jobs.set(jobId, {
      ...target,
      status: "failed",
      error: error instanceof Error ? error.message : "Unknown music generation error",
      updatedAt: new Date().toISOString()
    });
  }
};

export const createJob = (payload: CreateMusicRequest): { jobId: string; rulePrompt: string } => {
  const jobs = getJobs();
  const jobId = nanoid();
  const rulePrompt = createSinePrompt(payload);

  jobs.set(jobId, {
    id: jobId,
    status: "queued",
    rulePrompt,
    attemptCount: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  });

  void processMusicJob(jobId, payload, rulePrompt);
  return { jobId, rulePrompt };
};

export const getJobStatus = (jobId: string): MusicJobStatusResponse => {
  const job = getJobs().get(jobId);
  if (!job) {
    return {
      status: "failed",
      error: "Job not found"
    };
  }

  return {
    status: job.status,
    audioUrl: job.audioUrl,
    error: job.error,
    rulePrompt: job.rulePrompt,
    compositionPlan: job.compositionPlan,
    songMetadata: job.songMetadata,
    outputSanityScore: job.outputSanityScore,
    promptInferenceHiddenParams: job.promptInferenceHiddenParams,
    promptInferenceMeta: job.promptInferenceMeta,
    promptEval: job.promptEval,
    promptFeedback: job.promptFeedback,
    traceId: job.traceId
  };
};
