"use client";

import Link from "next/link";
import { useLocale } from "@/components/locale-context";
import { PixelButton, PixelPanel, PixelTag } from "@/components/pixel-ui";
import { useGame } from "@/components/game-context";
import { getCustomerName } from "@/lib/game-engine";
import { commissionStatusLabel } from "@/lib/i18n";

export const CommissionList = () => {
  const { commissions, selectedCommission, selectCommission, runInterpreterForCommission, busy } = useGame();
  const { locale, text } = useLocale();

  return (
    <PixelPanel title={text("commissionPanelTitle")} className="street-panel">
      <ul className="commission-list">
        {commissions.map((commission) => (
          <li key={commission.id}>
            <div className="commission-row">
              <div>
                <strong className="customer-id-label">{getCustomerName(commission.customerId)}</strong>
                <p>{commission.requestText}</p>
              </div>
              <div className="commission-actions">
                <PixelTag>{commissionStatusLabel(locale, commission.status)}</PixelTag>
                <PixelButton
                  type="button"
                  onClick={() => selectCommission(commission.id)}
                  disabled={selectedCommission?.id === commission.id}
                >
                  {text("commissionSelect")}
                </PixelButton>
                <PixelButton
                  type="button"
                  onClick={() => void runInterpreterForCommission(commission.id)}
                  disabled={busy}
                >
                  {text("commissionInterpret")}
                </PixelButton>
                <Link className="pixel-button-link" href={`/game/compose/${commission.id}`}>
                  {text("commissionCompose")}
                </Link>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </PixelPanel>
  );
};
