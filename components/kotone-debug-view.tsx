"use client";

import { useGame } from "@/components/game-context";
import { useLocale } from "@/components/locale-context";
import { PixelPanel, PixelTag } from "@/components/pixel-ui";
import { CATALOG_PARTS, getPartDescription } from "@/lib/catalog";
import { slotLabel } from "@/lib/i18n";
import { SLOT_KEYS } from "@/lib/types";

const PART_CARD_BASE_ASSET =
  "/assets/parts/sprites_trimmed/generated_image_february_28_2026_11_39am/generated_image_february_28_2026_11_39am_r01_c01.png";

export const KotoneDebugView = () => {
  const { locale, text } = useLocale();
  const { state } = useGame();
  const weatherClass = `scene-weather-${state?.weather ?? "sunny"}`;

  return (
    <div className={`kotone-layout ${weatherClass}`}>
      <PixelPanel title={text("kotoneDebugTitle")} className="kotone-main">
        <p className="kotone-debug-lead">
          <span>{text("kotoneDebugGuide")}</span>
          <strong>
            {text("kotoneDebugTotal")}: {CATALOG_PARTS.length}
          </strong>
        </p>

        <div className="kotone-debug-groups">
          {SLOT_KEYS.map((slot) => {
            const parts = CATALOG_PARTS.filter((part) => part.slot === slot);
            return (
              <section key={slot} className="kotone-debug-group">
                <header className="kotone-debug-group-header">
                  <h3>{slotLabel(locale, slot)}</h3>
                  <span>{parts.length}</span>
                </header>
                <div className="kotone-debug-grid">
                  {parts.map((part) => (
                    <article key={part.id} className="kotone-debug-item">
                      <div className="kotone-debug-item-main">
                        <span className="slot-part-thumb kotone-debug-thumb">
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
                        <div className="kotone-debug-item-copy">
                          <h4>{part.name}</h4>
                          <p>{getPartDescription(part, locale)}</p>
                        </div>
                      </div>
                      <p className="kotone-debug-item-id">{part.id}</p>
                      <div className="stack-row wrap">
                        <PixelTag>{part.price}G</PixelTag>
                        {part.tags.map((tag) => (
                          <PixelTag key={`${part.id}-${tag}`}>{tag}</PixelTag>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      </PixelPanel>
    </div>
  );
};
