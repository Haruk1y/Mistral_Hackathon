"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { nanoid } from "nanoid";
import {
  createMusicJobApi,
  getMusicJobApi,
  runInterpreterApi,
  submitCompositionApi,
  syncGameStateApi
} from "@/lib/api-client";
import { CATALOG_PARTS } from "@/lib/catalog";
import {
  canPurchasePart,
  createInitialState,
  getCustomerName,
  markCommissionMixing,
  purchasePart,
  startNextDay as startNextDayState,
  updateCommission
} from "@/lib/game-engine";
import { loadGameState, persistGameState, resetGameState } from "@/lib/local-state";
import type { Commission, GameState, SlotKey, Track } from "@/lib/types";

const persistAndSync = (nextState: GameState) => {
  persistGameState(nextState);
  void syncGameStateApi(nextState).catch(() => {
    // local-first: ignore network sync errors
  });
};

export const useGameState = () => {
  const [state, setState] = useState<GameState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setState(loadGameState());
  }, []);

  const updateState = useCallback((updater: (prev: GameState) => GameState) => {
    setState((prev) => {
      const base = prev ?? createInitialState();
      const next = updater(base);
      persistAndSync(next);
      return next;
    });
  }, []);

  const selectCommission = useCallback(
    (commissionId: string) => {
      updateState((prev) => ({ ...prev, selectedCommissionId: commissionId }));
    },
    [updateState]
  );

  const runInterpreterForCommission = useCallback(
    async (commissionId: string) => {
      if (!state) return;
      const commission = state.commissions[commissionId];
      if (!commission) return;

      setBusy(true);
      setError(null);

      try {
        updateState((prev) => markCommissionMixing(prev, commissionId));

        const response = await runInterpreterApi({
          requestText: commission.requestText,
          weather: commission.weather,
          inventoryPartIds: state.inventoryPartIds
        });

        updateState((prev) =>
          updateCommission(prev, commissionId, (current) => ({
            ...current,
            interpreterOutput: response,
            targetProfile: response.targetProfile,
            targetHiddenParams: response.targetHiddenParams,
            interpreterHiddenParams: response.targetHiddenParams,
            generationSource:
              response.evaluationMeta.model_source === "fine_tuned"
                ? "ft_model"
                : response.evaluationMeta.model_source === "prompt_baseline"
                  ? "prompt_baseline"
                  : "baseline",
            traceId: response.evaluationMeta.trace_id,
            status: "mixing",
            updatedAt: new Date().toISOString()
          }))
        );
      } catch (unknownError) {
        setError(unknownError instanceof Error ? unknownError.message : "Interpreter failed");
      } finally {
        setBusy(false);
      }
    },
    [state, updateState]
  );

  const submitComposition = useCallback(
    async (commissionId: string, selectedPartsBySlot: Record<SlotKey, string>) => {
      if (!state) return;
      const commission = state.commissions[commissionId];
      if (!commission?.targetProfile) {
        setError("targetProfile is not ready");
        return;
      }

      setBusy(true);
      setError(null);

      try {
        const result = await submitCompositionApi({
          commissionId,
          selectedPartsBySlot,
          targetProfile: commission.targetProfile
        });

        updateState((prev) => {
          const updatedCommission: Commission = {
            ...prev.commissions[commissionId],
            selectedPartsBySlot,
            score: result.score,
            rank: result.rank,
            rewardMoney: result.rewardMoney,
            coachFeedbackBySlot: result.coachFeedbackBySlot,
            distanceBreakdown: result.distanceBreakdown,
            evalSnapshot: result.evalSnapshot,
            status: "generating",
            updatedAt: new Date().toISOString()
          };

          return {
            ...prev,
            money: prev.money + result.rewardMoney,
            commissions: {
              ...prev.commissions,
              [commissionId]: updatedCommission
            },
            updatedAt: new Date().toISOString()
          };
        });

        const createResponse = await createMusicJobApi({
          commissionId,
          requestText: commission.requestText,
          selectedPartsBySlot,
          targetHiddenParams: commission.targetHiddenParams
        });

        updateState((prev) => ({
          ...prev,
          jobs: {
            ...prev.jobs,
              [createResponse.jobId]: {
                id: createResponse.jobId,
                commissionId,
                requestText: commission.requestText,
                selectedPartsBySlot,
                status: "queued",
                traceId: commission.traceId,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString()
              }
          },
          commissions: {
            ...prev.commissions,
            [commissionId]: {
              ...prev.commissions[commissionId],
              jobId: createResponse.jobId,
              status: "generating",
              updatedAt: new Date().toISOString()
            }
          },
          updatedAt: new Date().toISOString()
        }));

        return createResponse.jobId;
      } catch (unknownError) {
        setError(unknownError instanceof Error ? unknownError.message : "Submit failed");
      } finally {
        setBusy(false);
      }
    },
    [state, updateState]
  );

  const pollMusicJob = useCallback(
    async (jobId: string) => {
      if (!state) return;

      try {
        const status = await getMusicJobApi(jobId);

        updateState((prev) => {
          const existing = prev.jobs[jobId];
          if (!existing) return prev;

          const next = {
            ...prev,
            jobs: {
              ...prev.jobs,
              [jobId]: {
                ...existing,
                status: status.status,
                audioUrl: status.audioUrl,
                error: status.error,
                compositionPlan: status.compositionPlan,
                songMetadata: status.songMetadata,
                outputSanityScore: status.outputSanityScore,
                traceId: status.traceId ?? existing.traceId,
                updatedAt: new Date().toISOString()
              }
            }
          };

          if (status.status === "done" && status.audioUrl) {
            const commission = next.commissions[existing.commissionId];
            if (
              commission?.selectedPartsBySlot &&
              commission.rank &&
              typeof commission.score === "number" &&
              !commission.trackId
            ) {
              const track: Track = {
                id: nanoid(),
                commissionId: commission.id,
                audioUrl: status.audioUrl,
                usedPartsBySlot: commission.selectedPartsBySlot,
                score: commission.score,
                rank: commission.rank,
                compositionPlan: status.compositionPlan,
                songMetadata: status.songMetadata,
                outputSanityScore: status.outputSanityScore,
                traceId: status.traceId ?? commission.traceId,
                createdAt: new Date().toISOString()
              };

              next.tracks = {
                ...next.tracks,
                [track.id]: track
              };
              next.trackOrder = [track.id, ...next.trackOrder];
              next.commissions = {
                ...next.commissions,
                [commission.id]: {
                  ...commission,
                  status: "delivered",
                  trackId: track.id,
                  traceId: status.traceId ?? commission.traceId,
                  updatedAt: new Date().toISOString()
                }
              };
            }
          }

          next.updatedAt = new Date().toISOString();
          return next;
        });

        return status;
      } catch (unknownError) {
        setError(unknownError instanceof Error ? unknownError.message : "Polling failed");
        return undefined;
      }
    },
    [state, updateState]
  );

  const purchasePartById = useCallback(
    (partId: string) => {
      if (!state) return;
      const purchasable = canPurchasePart(state, partId);
      if (!purchasable.ok) {
        setError(purchasable.reason ?? "Purchase failed");
        return;
      }

      updateState((prev) => purchasePart(prev, partId));
    },
    [state, updateState]
  );

  const reset = useCallback(() => {
    const next = resetGameState();
    setState(next);
    setError(null);
  }, []);

  const startNextDay = useCallback(() => {
    updateState((prev) => startNextDayState(prev));
  }, [updateState]);

  const derived = useMemo(() => {
    if (!state) {
      return {
        selectedCommission: undefined,
        commissions: [],
        tracks: [],
        shopParts: [],
        canStartNextDay: false
      };
    }

    const commissions = state.commissionOrder.map((id) => state.commissions[id]);
    const canStartNextDay = commissions.every((commission) => commission.status === "delivered");
    const selectedCommission = state.selectedCommissionId
      ? state.commissions[state.selectedCommissionId]
      : commissions[0];
    const tracks = state.trackOrder.map((id) => state.tracks[id]);
    const shopParts = state.shopStock
      .map((stockItem) => CATALOG_PARTS.find((part) => part.id === stockItem.partId))
      .filter((part): part is (typeof CATALOG_PARTS)[number] => Boolean(part));

    return {
      selectedCommission,
      commissions,
      tracks,
      shopParts,
      selectedCustomerName: selectedCommission ? getCustomerName(selectedCommission.customerId) : "",
      canStartNextDay
    };
  }, [state]);

  return {
    state,
    busy,
    error,
    ...derived,
    selectCommission,
    runInterpreterForCommission,
    submitComposition,
    pollMusicJob,
    purchasePartById,
    reset,
    startNextDay
  };
};
