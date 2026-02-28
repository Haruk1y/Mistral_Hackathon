"use client";

import { PixelButton, PixelPanel, PixelTag } from "@/components/pixel-ui";
import { useGame } from "@/components/game-context";
import { useLocale } from "@/components/locale-context";
import { CATALOG_PARTS } from "@/lib/catalog";
import { slotLabel } from "@/lib/i18n";

export const ShopView = () => {
  const { state, shopParts, purchasePartById, error } = useGame();
  const { locale, text } = useLocale();

  if (!state) {
    return null;
  }

  return (
    <div className={`shop-layout scene-weather-${state.weather}`}>
      <PixelPanel title={text("shopTitle")} className="shop-left">
        <p>{text("shopGuide")}</p>
        <p>
          {text("shopPlayerMoney")}: <strong>{state.money}G</strong>
        </p>
      </PixelPanel>

      <PixelPanel title={text("shopItemsTitle")} className="shop-middle">
        <div className="shop-grid">
          {shopParts.map((part) => {
            const owned = state.inventoryPartIds.includes(part.id);
            return (
              <article key={part.id} className={`shop-item ${owned ? "owned" : ""}`}>
                <h4>{part.name}</h4>
                <p>{part.description}</p>
                <div className="stack-row">
                  <PixelTag>{slotLabel(locale, part.slot)}</PixelTag>
                  <PixelTag>{part.price}G</PixelTag>
                </div>
                <small>{part.tags.join(" / ")}</small>
                <PixelButton type="button" disabled={owned || state.money < part.price} onClick={() => purchasePartById(part.id)}>
                  {owned ? text("shopOwned") : text("shopPurchase")}
                </PixelButton>
              </article>
            );
          })}
        </div>
      </PixelPanel>

      <PixelPanel title={text("shopSummaryTitle")} className="shop-right">
        <ul>
          <li>
            {text("shopTotalParts")}: {CATALOG_PARTS.length}
          </li>
          <li>
            {text("shopOwnedParts")}: {state.inventoryPartIds.length}
          </li>
          <li>
            {text("shopMissingParts")}: {CATALOG_PARTS.length - state.inventoryPartIds.length}
          </li>
        </ul>
        {error ? <p className="error-text">{error}</p> : null}
      </PixelPanel>
    </div>
  );
};
