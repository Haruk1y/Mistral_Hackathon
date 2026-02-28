import type { ReactNode } from "react";
import { GameProvider } from "@/components/game-context";
import { GameNav } from "@/components/game-nav";
import { LocaleProvider } from "@/components/locale-context";
import { StageLayer, VirtualStage } from "@/components/virtual-stage";

export default function GameLayout({ children }: { children: ReactNode }) {
  return (
    <LocaleProvider>
      <GameProvider>
        <main className="game-shell">
          <GameNav />
          <VirtualStage>
            <StageLayer>{children}</StageLayer>
          </VirtualStage>
        </main>
      </GameProvider>
    </LocaleProvider>
  );
}
