import { nanoid } from "nanoid";
import type { CreateMusicRequest, MusicJobStatusResponse } from "@/lib/types";
import { createSinePrompt, generateMusicWithElevenLabs } from "@/lib/music-provider";
import { generateSineWaveDataUri } from "@/lib/audio";

type InternalMusicJob = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  audioUrl?: string;
  error?: string;
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

const processMusicJob = async (jobId: string, payload: CreateMusicRequest) => {
  const jobs = getJobs();
  const target = jobs.get(jobId);
  if (!target) return;

  target.status = "running";
  target.updatedAt = new Date().toISOString();
  jobs.set(jobId, target);

  try {
    const prompt = createSinePrompt(payload);
    const audioUrl = (await generateMusicWithElevenLabs(prompt)) ?? generateSineWaveDataUri();

    await wait(1400);

    jobs.set(jobId, {
      ...target,
      status: "done",
      audioUrl,
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
    error: job.error
  };
};
