import { createInitialState, migrateState } from "@/lib/game-engine";
import { gameStateSchema } from "@/lib/schemas";
import type { GameState } from "@/lib/types";

const STORAGE_KEY = "otokotoba.gameState.v1";

export const loadGameState = (): GameState => {
  if (typeof window === "undefined") {
    return createInitialState();
  }

  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return createInitialState();
  }

  try {
    const parsed = JSON.parse(raw);
    const validated = gameStateSchema.safeParse(parsed);
    if (!validated.success) {
      return createInitialState();
    }

    return migrateState(validated.data as GameState);
  } catch {
    return createInitialState();
  }
};

export const persistGameState = (state: GameState): void => {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
};

export const resetGameState = (): GameState => {
  const state = createInitialState();
  persistGameState(state);
  return state;
};
