"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useGameState } from "@/components/use-game-state";

type GameContextValue = ReturnType<typeof useGameState>;

const GameContext = createContext<GameContextValue | null>(null);

export const GameProvider = ({ children }: { children: ReactNode }) => {
  const value = useGameState();

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
};

export const useGame = () => {
  const context = useContext(GameContext);
  if (!context) {
    throw new Error("useGame must be used within GameProvider");
  }

  return context;
};
