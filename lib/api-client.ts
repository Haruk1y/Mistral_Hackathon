import type {
  CreateMusicRequest,
  CreateMusicResponse,
  GameState,
  InterpreterRequest,
  InterpreterResponse,
  MusicJobStatusResponse,
  SubmitCompositionRequest,
  SubmitCompositionResponse
} from "@/lib/types";

const request = async <T>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
  if (!response.ok) {
    const maybeText = await response.text();
    throw new Error(maybeText || `Request failed: ${response.status}`);
  }

  return response.json() as Promise<T>;
};

const callableBaseUrl = process.env.NEXT_PUBLIC_FIREBASE_FUNCTIONS_BASE_URL;

const callCallable = async <T>(name: string, payload: unknown): Promise<T> => {
  if (!callableBaseUrl) {
    throw new Error("Callable base URL is not configured.");
  }

  const response = await request<{ result?: T } & T>(`${callableBaseUrl}/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: payload })
  });

  if (response && typeof response === "object" && "result" in response) {
    return (response.result as T) ?? (response as unknown as T);
  }

  return response as unknown as T;
};

export const runInterpreterApi = async (payload: InterpreterRequest) => {
  if (callableBaseUrl) {
    return callCallable<InterpreterResponse>("runInterpreter", payload);
  }

  return request<InterpreterResponse>("/api/interpreter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
};

export const submitCompositionApi = async (payload: SubmitCompositionRequest & { targetProfile: unknown }) => {
  if (callableBaseUrl) {
    return callCallable<SubmitCompositionResponse>("submitComposition", payload);
  }

  return request<SubmitCompositionResponse>("/api/compose/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
};

export const createMusicJobApi = async (payload: CreateMusicRequest) => {
  if (callableBaseUrl) {
    return callCallable<CreateMusicResponse>("createMusicJob", payload);
  }

  return request<CreateMusicResponse>("/api/music/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
};

export const getMusicJobApi = async (jobId: string) => {
  if (callableBaseUrl) {
    return callCallable<MusicJobStatusResponse>("getMusicJobStatus", { jobId });
  }

  return request<MusicJobStatusResponse>(`/api/music/jobs/${jobId}`);
};

export const syncGameStateApi = async (state: GameState) => {
  if (callableBaseUrl) {
    return callCallable<{ ok: true }>("syncGameState", state);
  }

  return request<{ ok: true }>("/api/game/state", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state)
  });
};

export const initUserApi = async () => {
  if (!callableBaseUrl) {
    return { ok: true as const, mode: "local" as const };
  }

  return callCallable<{ ok: true }>("initUser", {});
};

export const startDayApi = async () => {
  if (!callableBaseUrl) {
    return { ok: true as const, mode: "local" as const };
  }

  return callCallable<{ ok: true }>("startDay", {});
};

export const beginCommissionApi = async (commissionId: string) => {
  if (!callableBaseUrl) {
    return { ok: true as const, mode: "local" as const, commissionId };
  }

  return callCallable<{ ok: true }>("beginCommission", { commissionId });
};

export const purchasePartApi = async (partId: string) => {
  if (!callableBaseUrl) {
    return { ok: true as const, mode: "local" as const, partId };
  }

  return callCallable<{ ok: true }>("purchasePart", { partId });
};

export const createShareLinkApi = async (trackId: string) => {
  if (!callableBaseUrl) {
    return { ok: true as const, mode: "local" as const, shareUrl: `/share/${trackId}` };
  }

  return callCallable<{ ok: true; shareUrl: string }>("createShareLink", { trackId });
};
