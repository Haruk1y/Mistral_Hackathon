import { nanoid } from "nanoid";
import type { CreateMusicRequest, MusicJobStatusResponse } from "@/lib/types";
import { computeOutputSanityScore, generateMusicWithElevenLabs } from "@/lib/music-provider";
import { generateSineWaveDataUri } from "@/lib/audio";

type InternalMusicJob = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  audioUrl?: string;
  error?: string;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  outputSanityScore?: number;
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

const createFallbackResult = () => {
  const audioUrl = generateSineWaveDataUri();
  const musicLengthMs = Math.max(3000, toInt(process.env.ELEVENLABS_MUSIC_LENGTH_MS, 30000));

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

const processMusicJob = async (jobId: string, payload: CreateMusicRequest) => {
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

      result = await generateMusicWithElevenLabs(payload);
      if (result?.audioUrl) {
        break;
      }

      if (attempt <= maxRetries) {
        await wait(350 * attempt);
      }
    }

    await wait(1400);

    const finalized =
      result?.audioUrl != null
        ? {
            audioUrl: result.audioUrl,
            compositionPlan: result.compositionPlan,
            songMetadata: result.songMetadata,
            outputSanityScore: result.outputSanityScore,
            traceId: result.traceId,
            error: undefined
          }
        : {
            ...createFallbackResult(),
            compositionPlan: undefined,
            songMetadata: undefined,
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

export const createJob = (payload: CreateMusicRequest): { jobId: string } => {
  const jobs = getJobs();
  const jobId = nanoid();

  jobs.set(jobId, {
    id: jobId,
    status: "queued",
    attemptCount: 0,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  });

  void processMusicJob(jobId, payload);
  return { jobId };
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
    compositionPlan: job.compositionPlan,
    songMetadata: job.songMetadata,
    outputSanityScore: job.outputSanityScore,
    traceId: job.traceId
  };
};
