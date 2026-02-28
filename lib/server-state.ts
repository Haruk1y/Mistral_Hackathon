import { createInitialState } from "@/lib/game-engine";
import { gameStateSchema } from "@/lib/schemas";
import type { GameState } from "@/lib/types";

declare global {
  // eslint-disable-next-line no-var
  var __otokotobaServerState: GameState | undefined;
}

export const getServerState = (): GameState => {
  if (!global.__otokotobaServerState) {
    global.__otokotobaServerState = createInitialState();
  }

  return global.__otokotobaServerState;
};

export const setServerState = (state: unknown): GameState => {
  const parsed = gameStateSchema.safeParse(state);
  if (!parsed.success) {
    throw new Error("Invalid state payload");
  }

  global.__otokotobaServerState = parsed.data as GameState;
  return global.__otokotobaServerState;
};
