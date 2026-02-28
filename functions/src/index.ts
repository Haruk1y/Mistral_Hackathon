import { setGlobalOptions } from "firebase-functions/v2";
import { onCall, HttpsError } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";
import { randomUUID } from "node:crypto";

initializeApp();
setGlobalOptions({ maxInstances: 10, region: "us-central1" });

type CallableRequest<T> = {
  data: T;
  auth?: {
    uid?: string;
  };
};

const ensureUid = (request: CallableRequest<unknown>): string => {
  const uid = request.auth?.uid;
  if (!uid) {
    throw new HttpsError("unauthenticated", "Authentication required.");
  }
  return uid;
};

type LocalMusicJob = {
  id: string;
  status: "queued" | "running" | "done" | "failed";
  audioUrl?: string;
  error?: string;
  compositionPlan?: unknown;
  songMetadata?: unknown;
  outputSanityScore?: number;
  traceId?: string;
  createdAt: string;
  updatedAt: string;
};

const SILENT_AUDIO_DATA_URI =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAABCxAgAEABAAZGF0YQAAAAA=";

const localJobs = new Map<string, LocalMusicJob>();

const now = () => new Date().toISOString();

const buildRecommended = (inventoryPartIds: string[] | undefined, slot: string): string[] => {
  const inventory = inventoryPartIds ?? [];
  const byPrefix = inventory.filter((partId) => partId.startsWith(`${slot}_`)).slice(0, 3);
  if (byPrefix.length > 0) return byPrefix;
  return inventory.slice(0, 3);
};

export const initUser = onCall(async (request) => {
  const uid = ensureUid(request as CallableRequest<unknown>);
  return {
    ok: true,
    uid,
    message: "initUser callable is wired. Implement Firestore bootstrap in next iteration."
  };
});

export const startDay = onCall(async (request) => {
  const uid = ensureUid(request as CallableRequest<unknown>);
  return {
    ok: true,
    uid,
    message: "startDay callable is wired."
  };
});

export const beginCommission = onCall(async (request) => {
  const uid = ensureUid(request as CallableRequest<unknown>);
  return {
    ok: true,
    uid,
    message: "beginCommission callable is wired."
  };
});

export const runInterpreter = onCall(async (request) => {
  ensureUid(request as CallableRequest<unknown>);
  const payload = request.data as {
    commissionId?: string;
    requestText?: string;
    weather?: string;
    inventoryPartIds?: string[];
  };

  if (!payload.requestText || !payload.weather) {
    throw new HttpsError("invalid-argument", "requestText and weather are required.");
  }

  return {
    recommended: {
      style: buildRecommended(payload.inventoryPartIds, "style"),
      instrument: buildRecommended(payload.inventoryPartIds, "inst"),
      mood: buildRecommended(payload.inventoryPartIds, "mood"),
      gimmick: buildRecommended(payload.inventoryPartIds, "gimmick")
    },
    targetProfile: {
      vector: {
        energy: 45,
        warmth: 58,
        brightness: 50,
        acousticness: 62,
        complexity: 35,
        nostalgia: 64
      },
      requiredTags: ["nostalgic"],
      optionalTags: ["cozy", "warm"],
      forbiddenTags: [],
      constraints: {}
    },
    targetHiddenParams: {
      vector: {
        energy: 45,
        warmth: 58,
        brightness: 50,
        acousticness: 62,
        complexity: 35,
        nostalgia: 64
      },
      tags: ["nostalgic", "cozy", "warm"],
      constraints: {}
    },
    hintToPlayer: "Callable mode: styleとmoodを先に合わせると安定します。",
    rationale: ["Firebase callable fallback response."],
    evaluationMeta: {
      json_valid: true,
      model_source: "firebase_callable",
      trace_id: randomUUID(),
      latency_ms: 1
    }
  };
});

export const submitComposition = onCall(async (request) => {
  ensureUid(request as CallableRequest<unknown>);
  const payload = request.data as {
    commissionId?: string;
    selectedPartsBySlot?: Record<string, string>;
  };

  if (!payload.commissionId || !payload.selectedPartsBySlot) {
    throw new HttpsError("invalid-argument", "commissionId and selectedPartsBySlot are required.");
  }

  return {
    score: 72,
    rank: "A",
    rewardMoney: 160,
    coachFeedbackBySlot: {
      style: "Callable fallback: styleの方向性は良いです。",
      instrument: "Callable fallback: instrumentは依頼と整合しています。",
      mood: "Callable fallback: moodを微調整すると更に改善します。",
      gimmick: "Callable fallback: gimmickは主張しすぎない方が安定します。"
    },
    distanceBreakdown: {
      vectorDistance: 42,
      normalizedDistance: 0.31,
      vectorScore: 42,
      tagScore: 18,
      preferenceScore: 4,
      synergyScore: 8,
      antiSynergyPenalty: 0
    },
    evalSnapshot: {
      intent_score_mean: 72,
      timestamp: now()
    }
  };
});

export const createMusicJob = onCall(async (request) => {
  ensureUid(request as CallableRequest<unknown>);
  const payload = request.data as {
    commissionId?: string;
    requestText?: string;
    selectedPartsBySlot?: Record<string, string>;
  };

  if (!payload.commissionId || !payload.requestText || !payload.selectedPartsBySlot) {
    throw new HttpsError(
      "invalid-argument",
      "commissionId, requestText, and selectedPartsBySlot are required."
    );
  }

  const jobId = randomUUID();
  const traceId = randomUUID();
  localJobs.set(jobId, {
    id: jobId,
    status: "running",
    traceId,
    createdAt: now(),
    updatedAt: now()
  });

  setTimeout(() => {
    const target = localJobs.get(jobId);
    if (!target) return;

    localJobs.set(jobId, {
      ...target,
      status: "done",
      audioUrl: SILENT_AUDIO_DATA_URI,
      compositionPlan: {
        source: "firebase_callable_fallback"
      },
      songMetadata: {
        force_instrumental: true,
        music_length_ms: 30000
      },
      outputSanityScore: 70,
      updatedAt: now()
    });
  }, 800);

  return {
    jobId
  };
});

export const purchasePart = onCall(async (request) => {
  ensureUid(request as CallableRequest<unknown>);
  const payload = request.data as { partId?: string };
  if (!payload.partId) {
    throw new HttpsError("invalid-argument", "partId is required.");
  }

  return {
    ok: true,
    partId: payload.partId,
    message: "purchasePart callable is wired."
  };
});

export const createShareLink = onCall(async (request) => {
  ensureUid(request as CallableRequest<unknown>);
  const payload = request.data as { trackId?: string };

  if (!payload.trackId) {
    throw new HttpsError("invalid-argument", "trackId is required.");
  }

  return {
    ok: true,
    trackId: payload.trackId,
    shareUrl: `https://example.com/share/${payload.trackId}`,
    message: "createShareLink callable is wired."
  };
});

export const getMusicJobStatus = onCall(async (request) => {
  ensureUid(request as CallableRequest<unknown>);
  const payload = request.data as { jobId?: string };

  if (!payload.jobId) {
    throw new HttpsError("invalid-argument", "jobId is required.");
  }

  const job = localJobs.get(payload.jobId);
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
});

export const syncGameState = onCall(async (request) => {
  ensureUid(request as CallableRequest<unknown>);
  if (!request.data || typeof request.data !== "object") {
    throw new HttpsError("invalid-argument", "state payload is required.");
  }

  return {
    ok: true
  };
});
