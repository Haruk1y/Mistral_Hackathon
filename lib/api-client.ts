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

export const runInterpreterApi = (payload: InterpreterRequest) =>
  request<InterpreterResponse>("/api/interpreter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

export const submitCompositionApi = (payload: SubmitCompositionRequest & { targetProfile: unknown }) =>
  request<SubmitCompositionResponse>("/api/compose/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

export const createMusicJobApi = (payload: CreateMusicRequest) =>
  request<CreateMusicResponse>("/api/music/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

export const getMusicJobApi = (jobId: string) =>
  request<MusicJobStatusResponse>(`/api/music/jobs/${jobId}`);

export const syncGameStateApi = (state: GameState) =>
  request<{ ok: true }>("/api/game/state", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state)
  });
