"use client";

import { PixelButton, PixelPanel, PixelTag } from "@/components/pixel-ui";
import { useGame } from "@/components/game-context";
import { useLocale } from "@/components/locale-context";
import { slotLabel } from "@/lib/i18n";

const PART_CARD_BASE_ASSET =
  "/assets/parts/sprites_trimmed/generated_image_february_28_2026_11_39am/generated_image_february_28_2026_11_39am_r01_c01.png";

export const ShopView = () => {
  const { state, shopParts, purchasePartById, error } = useGame();
  const { locale, text } = useLocale();

  if (!state) {
    return null;
  }

  return (
    <div className={`shop-layout scene-weather-${state.weather}`}>
      <PixelPanel title={text("shopItemsTitle")} className="shop-middle">
        <p className="shop-money-line">
          {text("shopPlayerMoney")}: <strong>{state.money}G</strong>
        </p>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="shop-grid">
          {shopParts.map((part) => {
            const owned = state.inventoryPartIds.includes(part.id);
            return (
              <article key={part.id} className={`shop-item ${owned ? "owned" : ""}`}>
                <div className="shop-item-main">
                  <span className="slot-part-thumb shop-part-thumb">
                    <img
                      src={PART_CARD_BASE_ASSET}
                      alt=""
                      aria-hidden
                      className="slot-part-card-base"
                      loading="lazy"
                    />
                    {part.iconAsset ? (
                      <img
                        src={part.iconAsset}
                        alt={part.name}
                        className="slot-part-thumb-image"
                        loading="lazy"
                      />
                    ) : (
                      <span className={`slot-part-thumb-fallback slot-${part.slot}`} aria-hidden>
                        {slotLabel(locale, part.slot).slice(0, 2)}
                      </span>
                    )}
                  </span>
                  <div className="shop-item-copy">
                    <h4>{part.name}</h4>
                    <p>{part.description}</p>
                  </div>
                </div>
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
    </div>
  );
};
