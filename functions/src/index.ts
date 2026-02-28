import { setGlobalOptions } from "firebase-functions/v2";
import { onCall, HttpsError } from "firebase-functions/v2/https";
import { initializeApp } from "firebase-admin/app";

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
    ok: true,
    commissionId: payload.commissionId ?? null,
    requestText: payload.requestText,
    weather: payload.weather,
    inventoryPartIds: payload.inventoryPartIds ?? [],
    message: "runInterpreter callable is wired."
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
    ok: true,
    commissionId: payload.commissionId,
    selectedPartsBySlot: payload.selectedPartsBySlot,
    message: "submitComposition callable is wired."
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

  return {
    ok: true,
    commissionId: payload.commissionId,
    message: "createMusicJob callable is wired."
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
