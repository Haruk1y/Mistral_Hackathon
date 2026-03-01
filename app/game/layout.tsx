import type { ReactNode } from "react";
import { DemoControlsProvider } from "@/components/demo-controls-context";
import { GameProvider } from "@/components/game-context";
import { GameNav } from "@/components/game-nav";
import { LocaleProvider } from "@/components/locale-context";
import { StageLayer, VirtualStage } from "@/components/virtual-stage";

export default function GameLayout({ children }: { children: ReactNode }) {
  return (
    <LocaleProvider>
      <DemoControlsProvider>
        <GameProvider>
          <main className="game-shell">
            <GameNav />
            <VirtualStage>
              <StageLayer>{children}</StageLayer>
            </VirtualStage>
          </main>
        </GameProvider>
      </DemoControlsProvider>
    </LocaleProvider>
  );
}
