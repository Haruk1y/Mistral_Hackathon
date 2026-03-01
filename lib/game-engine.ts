import { nanoid } from "nanoid";
import {
  CATALOG_CUSTOMERS,
  CATALOG_PARTS,
  REQUEST_TEMPLATES,
  SHOP_DEFAULT_STOCK,
  STARTER_PART_IDS,
  WEATHER_OPTIONS
} from "@/lib/catalog";
import { SLOT_KEYS } from "@/lib/types";
import type {
  Commission,
  CreateMusicRequest,
  GameState,
  MusicJob,
  TargetProfile,
  Rank,
  ShopStockItem,
  Track,
  Weather
} from "@/lib/types";

export const GAME_SCHEMA_VERSION = 2;

const now = () => new Date().toISOString();

const normalizeWeather = (weather: Weather): Weather => (weather === "rainy" ? "cloudy" : weather);

const normalizeStateWeather = (state: GameState): GameState => {
  const normalizedWeather = normalizeWeather(state.weather);
  let commissionWeatherChanged = false;

  const normalizedCommissions = Object.fromEntries(
    Object.entries(state.commissions).map(([commissionId, commission]) => {
      const normalizedCommissionWeather = normalizeWeather(commission.weather);
      if (normalizedCommissionWeather !== commission.weather) {
        commissionWeatherChanged = true;
      }
      return [
        commissionId,
        {
          ...commission,
          weather: normalizedCommissionWeather
        }
      ] as const;
    })
  );

  if (!commissionWeatherChanged && normalizedWeather === state.weather) {
    return state;
  }

  return {
    ...state,
    weather: normalizedWeather,
    commissions: normalizedCommissions,
    updatedAt: now()
  };
};

const randomWeather = (): Weather => WEATHER_OPTIONS[Math.floor(Math.random() * WEATHER_OPTIONS.length)];

const weightedRequests = (weather: Weather) => {
  return REQUEST_TEMPLATES.flatMap((template) =>
    template.weatherBias.includes(weather) ? [template, template] : [template]
  );
};

const pickRandomRequest = (weather: Weather) => {
  const weighted = weightedRequests(weather);
  return weighted[Math.floor(Math.random() * weighted.length)];
};

const toTargetHiddenParams = (profile: TargetProfile) => ({
  vector: profile.vector,
  tags: [...new Set([...profile.requiredTags, ...profile.optionalTags])],
  constraints: {
    ...profile.constraints
  }
});

const createDailyCommissions = (weather: Weather, count = 3): Commission[] => {
  const weighted = weightedRequests(weather);
  const shuffled = [...weighted];

  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const selected = [];
  const usedCustomerIds = new Set<string>();

  for (const request of shuffled) {
    if (selected.length >= count) break;
    if (usedCustomerIds.has(request.customerId)) continue;
    selected.push(request);
    usedCustomerIds.add(request.customerId);
  }

  while (selected.length < count) {
    selected.push(pickRandomRequest(weather));
  }

  return selected.map((request) => ({
    id: nanoid(),
    customerId: request.customerId,
    requestText: request.text,
    weather,
    status: "queued",
    requestGenerationSource: "template",
    targetProfile: request.targetProfile,
    targetHiddenParams: toTargetHiddenParams(request.targetProfile),
    createdAt: now(),
    updatedAt: now()
  }));
};

const ensureCommissionVariety = (state: GameState): GameState => {
  const currentCommissions = state.commissionOrder
    .map((id) => state.commissions[id])
    .filter((commission): commission is Commission => Boolean(commission));

  if (currentCommissions.length < 3) return state;
  if (currentCommissions.some((commission) => commission.status !== "queued")) return state;

  const uniqueCustomers = new Set(currentCommissions.map((commission) => commission.customerId));
  if (uniqueCustomers.size >= 2) return state;

  const regenerated = createDailyCommissions(state.weather, currentCommissions.length);
  const commissionMap = Object.fromEntries(regenerated.map((commission) => [commission.id, commission]));

  return {
    ...state,
    commissions: commissionMap,
    commissionOrder: regenerated.map((commission) => commission.id),
    selectedCommissionId: regenerated[0]?.id,
    updatedAt: now()
  };
};

export const createInitialState = (): GameState => {
  const weather = randomWeather();
  const commissions = createDailyCommissions(weather, 3);

  const commissionMap = Object.fromEntries(commissions.map((commission) => [commission.id, commission]));

  return {
    schemaVersion: GAME_SCHEMA_VERSION,
    money: 150,
    day: 1,
    weather,
    inventoryPartIds: STARTER_PART_IDS,
    commissions: commissionMap,
    commissionOrder: commissions.map((commission) => commission.id),
    tracks: {},
    trackOrder: [],
    jobs: {},
    shopStock: SHOP_DEFAULT_STOCK,
    selectedCommissionId: commissions[0].id,
    updatedAt: now()
  };
};

export const startNextDay = (state: GameState): GameState => {
  const weather = randomWeather();
  const commissions = createDailyCommissions(weather, 3);
  const commissionMap = Object.fromEntries(commissions.map((commission) => [commission.id, commission]));

  return {
    ...state,
    day: state.day + 1,
    weather,
    commissions: commissionMap,
    commissionOrder: commissions.map((commission) => commission.id),
    selectedCommissionId: commissions[0].id,
    updatedAt: now()
  };
};

export const migrateState = (state: GameState): GameState => {
  if (state.schemaVersion === GAME_SCHEMA_VERSION) {
    return ensureCommissionVariety(normalizeStateWeather(state));
  }

  const fresh = ensureCommissionVariety(createInitialState());
  const validInventory = state.inventoryPartIds.filter((partId) =>
    CATALOG_PARTS.some((part) => part.id === partId)
  );
  const slotCoverage = new Set(
    validInventory
      .map((partId) => CATALOG_PARTS.find((part) => part.id === partId)?.slot)
      .filter((slot): slot is (typeof SLOT_KEYS)[number] => Boolean(slot))
  );
  const hasAllSlots = SLOT_KEYS.every((slot) => slotCoverage.has(slot));

  return ensureCommissionVariety(
    normalizeStateWeather({
    ...fresh,
    money: typeof state.money === "number" ? state.money : fresh.money,
    day: typeof state.day === "number" && state.day > 0 ? state.day : fresh.day,
    inventoryPartIds: hasAllSlots && validInventory.length ? [...new Set(validInventory)] : fresh.inventoryPartIds,
    schemaVersion: GAME_SCHEMA_VERSION,
    updatedAt: now()
    })
  );
};

export const getPartById = (partId: string) => CATALOG_PARTS.find((part) => part.id === partId);

export const getCustomerName = (customerId: string) =>
  CATALOG_CUSTOMERS.find((customer) => customer.id === customerId)?.name ?? customerId.toUpperCase();

export const updateCommission = (
  state: GameState,
  commissionId: string,
  updater: (commission: Commission) => Commission
): GameState => {
  const current = state.commissions[commissionId];
  if (!current) return state;

  return {
    ...state,
    commissions: {
      ...state.commissions,
      [commissionId]: updater(current)
    },
    updatedAt: now()
  };
};

export const markCommissionMixing = (state: GameState, commissionId: string): GameState =>
  updateCommission(state, commissionId, (commission) => ({
    ...commission,
    status: "mixing",
    updatedAt: now()
  }));

export const applyCompositionResult = (
  state: GameState,
  input: {
    commissionId: string;
    selectedPartsBySlot: Track["usedPartsBySlot"];
    score: number;
    rank: Rank;
    rewardMoney: number;
    coachFeedbackBySlot: Record<keyof Track["usedPartsBySlot"], string>;
  }
): GameState => {
  return {
    ...state,
    money: state.money + input.rewardMoney,
    commissions: {
      ...state.commissions,
      [input.commissionId]: {
        ...state.commissions[input.commissionId],
        status: "generating",
        selectedPartsBySlot: input.selectedPartsBySlot,
        score: input.score,
        rank: input.rank,
        rewardMoney: input.rewardMoney,
        coachFeedbackBySlot: input.coachFeedbackBySlot,
        updatedAt: now()
      }
    },
    updatedAt: now()
  };
};

export const canPurchasePart = (state: GameState, partId: string): { ok: boolean; reason?: string } => {
  const part = getPartById(partId);
  if (!part) return { ok: false, reason: "Part not found" };
  if (state.inventoryPartIds.includes(partId)) return { ok: false, reason: "Part already owned" };
  if (state.money < part.price) return { ok: false, reason: "Not enough money" };
  return { ok: true };
};

export const purchasePart = (state: GameState, partId: string): GameState => {
  const part = getPartById(partId);
  if (!part) return state;

  return {
    ...state,
    money: state.money - part.price,
    inventoryPartIds: [...state.inventoryPartIds, partId],
    shopStock: state.shopStock.map((item: ShopStockItem) =>
      item.partId === partId ? { ...item, unlocked: false } : item
    ),
    updatedAt: now()
  };
};

export const createMusicJob = (state: GameState, input: CreateMusicRequest): { state: GameState; job: MusicJob } => {
  const job: MusicJob = {
    id: nanoid(),
    commissionId: input.commissionId,
    requestText: input.requestText,
    selectedPartsBySlot: input.selectedPartsBySlot,
    status: "queued",
    createdAt: now(),
    updatedAt: now()
  };

  const nextState: GameState = {
    ...state,
    jobs: {
      ...state.jobs,
      [job.id]: job
    },
    commissions: {
      ...state.commissions,
      [input.commissionId]: {
        ...state.commissions[input.commissionId],
        jobId: job.id,
        status: "generating",
        updatedAt: now()
      }
    },
    updatedAt: now()
  };

  return { state: nextState, job };
};

export const attachTrackToCommission = (
  state: GameState,
  input: {
    commissionId: string;
    jobId: string;
    audioUrl: string;
  }
): GameState => {
  const commission = state.commissions[input.commissionId];
  if (!commission?.selectedPartsBySlot || !commission.rank || typeof commission.score !== "number") {
    return state;
  }

  const track: Track = {
    id: nanoid(),
    commissionId: input.commissionId,
    audioUrl: input.audioUrl,
    usedPartsBySlot: commission.selectedPartsBySlot,
    score: commission.score,
    rank: commission.rank,
    createdAt: now()
  };

  return {
    ...state,
    tracks: {
      ...state.tracks,
      [track.id]: track
    },
    trackOrder: [track.id, ...state.trackOrder],
    commissions: {
      ...state.commissions,
      [input.commissionId]: {
        ...commission,
        status: "delivered",
        trackId: track.id,
        updatedAt: now()
      }
    },
    jobs: {
      ...state.jobs,
      [input.jobId]: {
        ...state.jobs[input.jobId],
        status: "done",
        audioUrl: input.audioUrl,
        updatedAt: now()
      }
    },
    updatedAt: now()
  };
};
