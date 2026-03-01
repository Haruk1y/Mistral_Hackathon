"use client";

import { StreetCommissionLayer } from "@/components/street-commission-layer";
import { useDemoControls } from "@/components/demo-controls-context";
import { useLocale } from "@/components/locale-context";
import { PixelButton } from "@/components/pixel-ui";
import { useGame } from "@/components/game-context";

export const StreetView = () => {
  const { state, commissions, selectCommission, reset, startNextDay, canStartNextDay } = useGame();
  const { text } = useLocale();
  const { showDemoControls } = useDemoControls();
  const weatherClass = `scene-weather-${state?.weather ?? "sunny"}`;

  return (
    <div className={`street-layout ${weatherClass}`}>
      <StreetCommissionLayer
        day={state?.day ?? 1}
        weather={state?.weather ?? "sunny"}
        commissions={commissions}
        onSelect={selectCommission}
      />
      {showDemoControls ? (
        <div className="street-floating-actions">
          <PixelButton type="button" onClick={reset}>
            {text("streetResetSave")}
          </PixelButton>
          <PixelButton type="button" onClick={startNextDay} disabled={!canStartNextDay}>
            {text("streetStartNextDay")}
          </PixelButton>
        </div>
      ) : null}
    </div>
  );
};
