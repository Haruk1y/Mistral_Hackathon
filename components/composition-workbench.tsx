"use client";

import { useEffect, useMemo, useState } from "react";
import { useLocale } from "@/components/locale-context";
import { PixelButton, PixelPanel } from "@/components/pixel-ui";
import { useGame } from "@/components/game-context";
import { CATALOG_CUSTOMERS, CATALOG_PARTS } from "@/lib/catalog";
import { getCustomerName } from "@/lib/game-engine";
import { jobStatusLabel, slotLabel } from "@/lib/i18n";
import { STREET_CROWD_CHARACTER_BY_ID } from "@/lib/street-crowd-catalog";
import type { MusicJobStatus, Part, SlotKey } from "@/lib/types";
import { SLOT_KEYS } from "@/lib/types";

const initialSlotState = (): Record<SlotKey, string> =>
  Object.fromEntries(SLOT_KEYS.map((slot) => [slot, ""])) as Record<SlotKey, string>;

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
    submitComposition,
    pollMusicJob,
    selectCommission
  } = useGame();

  const commission = state?.commissions[commissionId];
  const [selectedPartsBySlot, setSelectedPartsBySlot] = useState<Record<SlotKey, string>>(initialSlotState);
  const [activeSlot, setActiveSlot] = useState<SlotKey>(SLOT_KEYS[0] ?? "style");
  const [jobStatus, setJobStatus] = useState<MusicJobStatus | null>(null);

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

  const displayName = streetCast?.name ?? (commission ? getCustomerName(commission.customerId) : "Unknown");
  const displayPortraitAsset = streetCast?.portraitAsset ?? customer?.portraitAsset;

  const allSlotsSelected = SLOT_KEYS.every((slot) => selectedPartsBySlot[slot]);

  const handleCompose = async () => {
    if (!commission || !allSlotsSelected) return;

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
  };

  if (!commission) {
    return (
      <PixelPanel title={text("composeTitle")}>
        <p>{text("composeNotFound")}</p>
      </PixelPanel>
    );
  }

  return (
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
              <strong>{displayName}</strong>
            </p>
            {customer?.personality ? <p className="muted">{customer.personality}</p> : null}
          </div>
        </div>
        <p className="compose-request">{commission.requestText}</p>
        {commission.interpreterOutput ? (
          <>
            <p className="muted">
              {text("composeHintPrefix")}: {commission.interpreterOutput.hintToPlayer}
            </p>
            <ul>
              {commission.interpreterOutput.rationale.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </>
        ) : null}
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
            {commission.coachFeedbackBySlot ? (
              <ul>
                {SLOT_KEYS.map((slot) => (
                  <li key={slot}>
                    <strong>{slotLabel(locale, slot)}:</strong> {commission.coachFeedbackBySlot?.[slot]}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}

        {jobStatus ? (
          <p className="muted">
            {text("composeMusicJobLabel")}: {jobStatusLabel(locale, jobStatus)}
          </p>
        ) : null}
        {commission.trackId && state?.tracks[commission.trackId] ? (
          <audio controls src={state.tracks[commission.trackId].audioUrl} className="audio-player" />
        ) : null}
        {error ? <p className="error-text">{error}</p> : null}
      </PixelPanel>

      <PixelPanel title={text("composePartsPanelTitle")} className="compose-parts-panel">
        <div className="compose-parts-header">
          <p className="slot-picker-title">
            {text("composePartsSlotPrefix")}: {slotLabel(locale, activeSlot)}
          </p>
          <p className="muted">{text("composePartsHint")}</p>
        </div>

        {activeOptions.length ? (
          <div className="slot-part-tabs">
            {activeOptions.map((part) => {
              const selected = selectedPartsBySlot[activeSlot] === part.id;
              const isRecommended = activeRecommended.includes(part.id);

              return (
                <button
                  key={part.id}
                  type="button"
                  className={`slot-part-tab ${selected ? "is-selected" : ""} ${isRecommended ? "is-recommended" : ""}`}
                  onClick={() =>
                    setSelectedPartsBySlot((prev) => ({
                      ...prev,
                      [activeSlot]: part.id
                    }))
                  }
                  aria-pressed={selected}
                >
                  <span className="slot-part-thumb">
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
                  <span className="slot-part-content">
                    <span className="slot-part-name">
                      {part.name}
                      {isRecommended ? " ★" : ""}
                    </span>
                    <span className="slot-part-description">{part.description}</span>
                  </span>
                </button>
              );
            })}
          </div>
        ) : (
          <p className="muted">{text("composeNoPartsInSlot")}</p>
        )}
      </PixelPanel>
    </div>
  );
};
