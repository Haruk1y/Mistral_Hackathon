"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useLocale } from "@/components/locale-context";
import { PixelButton, PixelPanel } from "@/components/pixel-ui";
import { useGame } from "@/components/game-context";
import { CATALOG_CUSTOMERS, CATALOG_PARTS, getPartDescription } from "@/lib/catalog";
import { jobStatusLabel, slotLabel } from "@/lib/i18n";
import { STREET_CROWD_CHARACTER_BY_ID } from "@/lib/street-crowd-catalog";
import type { MusicJobStatus, Part, SlotKey } from "@/lib/types";
import { SLOT_KEYS } from "@/lib/types";

const initialSlotState = (): Record<SlotKey, string> =>
  Object.fromEntries(SLOT_KEYS.map((slot) => [slot, ""])) as Record<SlotKey, string>;

const PART_CARD_BASE_ASSET =
  "/assets/parts/sprites_trimmed/generated_image_february_28_2026_11_39am/generated_image_february_28_2026_11_39am_r01_c01.png";

const buildCustomerDialogue = (input: {
  displayName: string;
  score: number;
  weather: "sunny" | "cloudy" | "rainy";
  locale: "ja" | "en";
}) => {
  if (input.locale === "ja") {
    if (input.score >= 88) {
      return {
        reaction: `${input.displayName}: 「わあ、今の一曲すごく好き。街の空気がそのまま音になったみたい。」`,
        tip: "Kotone: 「次は余韻を少しだけ長めにして、情景をもっと深く見せましょう。」"
      };
    }

    if (input.score >= 72) {
      return {
        reaction: `${input.displayName}: 「いい感じ。雰囲気は掴めてるから、もう一歩だけ色を揃えたいな。」`,
        tip:
          input.weather === "rainy"
            ? "Kotone: 「雨の夜を歩くような静かな流れを、もう少し前に出してみましょう。」"
            : "Kotone: 「主役のフレーズを少しはっきりさせると、印象がぐっと強くなります。」"
      };
    }

    return {
      reaction: `${input.displayName}: 「悪くないけど、私が欲しかった景色とは少し違うかも。」`,
      tip: "Kotone: 「テンポ感か空気感のどちらかを思い切って寄せると、狙いが伝わりやすくなります。」"
    };
  }

  if (input.score >= 88) {
    return {
      reaction: `${input.displayName}: "I love this one. It really sounds like our town right now."`,
      tip: 'Kotone: "Next, try a slightly longer tail so the scene can breathe."'
    };
  }

  if (input.score >= 72) {
    return {
      reaction: `${input.displayName}: "Nice direction. The mood is there, it just needs one more clear color."`,
      tip:
        input.weather === "rainy"
          ? 'Kotone: "Push the calm rainy-night flow a little more."'
          : 'Kotone: "Make the lead phrase stand out a bit more."'
    };
  }

  return {
    reaction: `${input.displayName}: "Not bad, but it still feels a little far from what I imagined."`,
    tip: 'Kotone: "Commit harder to either tempo feel or atmosphere for the next try."'
  };
};

export const CompositionWorkbench = ({
  commissionId,
  streetCastId
}: {
  commissionId: string;
  streetCastId?: string;
}) => {
  const { locale, text } = useLocale();
  const {
    state,
    busy,
    error,
    runInterpreterForCommission,
    submitComposition,
    pollMusicJob,
    selectCommission
  } = useGame();

  const commission = state?.commissions[commissionId];
  const [selectedPartsBySlot, setSelectedPartsBySlot] = useState<Record<SlotKey, string>>(initialSlotState);
  const [activeSlot, setActiveSlot] = useState<SlotKey>(SLOT_KEYS[0] ?? "style");
  const [jobStatus, setJobStatus] = useState<MusicJobStatus | null>(null);
  const [autoInterpreterTargetId, setAutoInterpreterTargetId] = useState<string | null>(null);
  const [isComposing, setIsComposing] = useState(false);

  useEffect(() => {
    if (!commission) return;
    selectCommission(commission.id);
    setActiveSlot(SLOT_KEYS[0] ?? "style");

    if (commission.selectedPartsBySlot) {
      const restored = initialSlotState();
      for (const slot of SLOT_KEYS) {
        restored[slot] = commission.selectedPartsBySlot[slot] ?? "";
      }
      setSelectedPartsBySlot(restored);
      return;
    }

    setSelectedPartsBySlot(initialSlotState());
  }, [commission, selectCommission]);

  const inventoryParts = useMemo(() => {
    if (!state) return [];
    return CATALOG_PARTS.filter((part) => state.inventoryPartIds.includes(part.id));
  }, [state]);

  useEffect(() => {
    if (!commission) return;
    if (!commission.requestGenerationSource || commission.requestGenerationSource === "template") return;
    if (commission.interpreterOutput) return;
    if (autoInterpreterTargetId === commission.id) return;

    setAutoInterpreterTargetId(commission.id);
    void runInterpreterForCommission(commission.id);
  }, [autoInterpreterTargetId, commission, runInterpreterForCommission]);

  const partsById = useMemo(
    () => Object.fromEntries(CATALOG_PARTS.map((part) => [part.id, part])) as Record<string, Part>,
    []
  );

  const bySlot = useMemo(
    () =>
      Object.fromEntries(
        SLOT_KEYS.map((slot) => [slot, inventoryParts.filter((part) => part.slot === slot)])
      ) as Record<SlotKey, typeof inventoryParts>,
    [inventoryParts]
  );

  const activeOptions = bySlot[activeSlot];
  const activeRecommended = commission?.interpreterOutput?.recommended[activeSlot] ?? [];
  const selectedActivePart = useMemo(() => {
    if (!activeOptions.length) return undefined;
    const selectedPartId = selectedPartsBySlot[activeSlot];
    if (!selectedPartId) return undefined;
    return activeOptions.find((part) => part.id === selectedPartId);
  }, [activeOptions, selectedPartsBySlot, activeSlot]);

  const customer = useMemo(() => {
    if (!commission) return undefined;
    return CATALOG_CUSTOMERS.find((item) => item.id === commission.customerId);
  }, [commission]);

  const streetCast = useMemo(() => {
    if (!commission || !streetCastId) return undefined;
    const candidate = STREET_CROWD_CHARACTER_BY_ID[streetCastId];
    if (!candidate) return undefined;
    if (candidate.customerId !== commission.customerId) return undefined;
    return candidate;
  }, [commission, streetCastId]);

  const displayName = commission ? commission.customerId.toLowerCase() : "unknown";
  const displayPortraitAsset = streetCast?.portraitAsset ?? customer?.portraitAsset;
  const generatedTrack = commission?.trackId ? state?.tracks[commission.trackId] : undefined;

  const allSlotsSelected = SLOT_KEYS.every((slot) => selectedPartsBySlot[slot]);
  const dialogue = useMemo(
    () =>
      buildCustomerDialogue({
        displayName,
        score: commission?.score ?? 0,
        weather: commission?.weather ?? "cloudy",
        locale
      }),
    [commission?.score, commission?.weather, displayName, locale]
  );

  useEffect(() => {
    if (!commission?.interpreterOutput) return;
    if (!inventoryParts.length) return;

    const hasAnySelected = SLOT_KEYS.some((slot) => Boolean(selectedPartsBySlot[slot]));
    if (hasAnySelected) return;

    const suggested = initialSlotState();
    for (const slot of SLOT_KEYS) {
      const recommendedIds = commission.interpreterOutput.recommended[slot] ?? [];
      const inventoryForSlot = inventoryParts.filter((part) => part.slot === slot);
      const matched = recommendedIds.find((partId) => inventoryForSlot.some((part) => part.id === partId));
      suggested[slot] = matched ?? inventoryForSlot[0]?.id ?? "";
    }

    setSelectedPartsBySlot(suggested);
  }, [commission?.interpreterOutput, inventoryParts, selectedPartsBySlot]);

  const handleCompose = async () => {
    if (!commission || !allSlotsSelected) return;

    setIsComposing(true);
    try {
      const jobId = await submitComposition(commission.id, selectedPartsBySlot);
      if (!jobId) return;

      setJobStatus("queued");

      const startedAt = Date.now();
      const timeoutMs = 90_000;

      while (Date.now() - startedAt < timeoutMs) {
        const status = await pollMusicJob(jobId);
        if (!status) {
          setJobStatus("failed");
          return;
        }

        setJobStatus(status.status);
        if (status.status === "done" || status.status === "failed") {
          break;
        }

        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
    } finally {
      setIsComposing(false);
    }
  };

  if (!commission) {
    return (
      <PixelPanel title={text("composeTitle")}>
        <p>{text("composeNotFound")}</p>
      </PixelPanel>
    );
  }

  return (
    <>
      <div className={`compose-grid scene-weather-${commission.weather}`}>
      <PixelPanel title={text("composeCustomerTitle")} className="compose-customer">
        <div className="compose-customer-header">
          {displayPortraitAsset ? (
            <img
              src={displayPortraitAsset}
              alt={displayName}
              className="compose-customer-portrait"
              loading="lazy"
            />
          ) : (
            <div className="compose-customer-portrait placeholder" aria-hidden />
          )}
          <div>
            <p>
              <strong className="customer-id-label">{displayName}</strong>
            </p>
          </div>
        </div>
        <p className="compose-request">{commission.requestText}</p>
      </PixelPanel>

      <PixelPanel title={text("composePanelTitle")} className="compose-panel">
        <div className="slot-grid">
          {SLOT_KEYS.map((slot) => {
            const selectedPartId = selectedPartsBySlot[slot];
            const selectedPart = selectedPartId ? partsById[selectedPartId] : undefined;
            const isActive = activeSlot === slot;

            return (
              <button
                key={slot}
                type="button"
                className={`slot-anchor ${isActive ? "is-active" : ""}`}
                onClick={() => setActiveSlot(slot)}
              >
                <span className="slot-anchor-label">{slotLabel(locale, slot)}</span>
                <span className="slot-anchor-value">{selectedPart?.name ?? text("composeSelectPlaceholder")}</span>
              </button>
            );
          })}
        </div>

        <div className="compose-actions">
          <PixelButton type="button" onClick={() => void handleCompose()} disabled={!allSlotsSelected || busy}>
            {text("composeAndPlay")}
          </PixelButton>
          <PixelButton type="button" onClick={() => setSelectedPartsBySlot(initialSlotState())}>
            {text("composeReset")}
          </PixelButton>
        </div>

        {commission.score !== undefined && commission.rank ? (
          <div className="result-box">
            <h4>{text("composeResultTitle")}</h4>
            <p>
              {text("composeScoreLabel")}: {commission.score} / {text("composeRankLabel")}: {commission.rank}
            </p>
            <p>
              {text("composeRewardLabel")}: +{commission.rewardMoney ?? 0}G
            </p>
            <div className="compose-result-actions">
              <Link href="/game/street" className="pixel-button-link">
                {text("galleryBackToStreet")}
              </Link>
            </div>
          </div>
        ) : null}

        {jobStatus ? (
          <p className="muted">
            {text("composeMusicJobLabel")}: {jobStatusLabel(locale, jobStatus)}
          </p>
        ) : null}
        {error ? <p className="error-text">{error}</p> : null}
      </PixelPanel>

      <PixelPanel title={generatedTrack ? text("composeResultTitle") : text("composePartsPanelTitle")} className="compose-parts-panel">
        {generatedTrack ? (
          <div className="compose-generated-result">
            <audio controls src={generatedTrack.audioUrl} className="audio-player" />
            <div className="stack-row wrap">
              {Object.entries(generatedTrack.usedPartsBySlot).map(([slot, partId]) => (
                <span key={`${generatedTrack.id}-${slot}`} className="pixel-tag">
                  {slot.toUpperCase()}: {partsById[partId]?.name ?? partId}
                </span>
              ))}
            </div>
            <p className="compose-dialogue">{dialogue.reaction}</p>
            <p className="compose-dialogue muted">{dialogue.tip}</p>
          </div>
        ) : activeOptions.length ? (
          <div className="slot-part-picker-layout">
            <aside className="slot-part-inspector" aria-live="polite">
              {selectedActivePart ? (
                <>
                  <h4 className="slot-part-inspector-name">{selectedActivePart.name}</h4>
                  <p className="slot-part-inspector-description">{getPartDescription(selectedActivePart, locale)}</p>
                  <p className="slot-part-inspector-meta">
                    {slotLabel(locale, selectedActivePart.slot)} / {selectedActivePart.price}G
                  </p>
                  <p className="slot-part-inspector-tags">{selectedActivePart.tags.join(" / ")}</p>
                </>
              ) : (
                <p className="muted">{text("composeSelectPlaceholder")}</p>
              )}
            </aside>
            <div className="slot-part-carousel" role="listbox" aria-label={slotLabel(locale, activeSlot)}>
              {activeOptions.map((part) => {
                const selected = selectedPartsBySlot[activeSlot] === part.id;
                const isRecommended = activeRecommended.includes(part.id);

                return (
                  <button
                    key={part.id}
                    type="button"
                    className={`slot-part-card ${selected ? "is-selected" : ""} ${isRecommended ? "is-recommended" : ""}`}
                    onClick={() =>
                      setSelectedPartsBySlot((prev) => ({
                        ...prev,
                        [activeSlot]: part.id
                      }))
                    }
                    aria-pressed={selected}
                    title={part.name}
                  >
                    <span className="slot-part-thumb">
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
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <p className="muted">{text("composeNoPartsInSlot")}</p>
        )}
      </PixelPanel>
      </div>
      {isComposing ? (
        <div className="compose-loading-overlay" role="status" aria-live="polite" aria-label="Generating music">
          <div className="compose-loading-panel">
            <span className="compose-spinner" aria-hidden />
            <p>{locale === "ja" ? "音楽生成中..." : "Generating music..."}</p>
            <p className="muted">{locale === "ja" ? "少し時間がかかります" : "This may take a little while."}</p>
          </div>
        </div>
      ) : null}
    </>
  );
};
