import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { SLOT_KEYS } from "@/lib/types";
import type { ProfileVector, SlotKey } from "@/lib/types";

const VECTOR_KEYS: Array<keyof ProfileVector> = [
  "energy",
  "warmth",
  "brightness",
  "acousticness",
  "complexity",
  "nostalgia"
];

const DEFAULT_DATASET_PATHS = [
  "data/ft/ft_request_param_train.jsonl",
  "data/ft/ft_request_param_valid.jsonl",
  "data/ft/ft_request_param_test.jsonl"
];
const VECTOR_FALLBACK: ProfileVector = {
  energy: 50,
  warmth: 50,
  brightness: 50,
  acousticness: 50,
  complexity: 50,
  nostalgia: 50
};

export type SlotSelection = Record<SlotKey, string>;

type RequestEntry = {
  requestText: string;
  vector: ProfileVector;
  tokens: Set<string>;
};

type RuleEntry = {
  promptText: string;
  slots: SlotSelection;
  vector: ProfileVector;
};

type LoadedDataset = {
  sourceKey: string;
  sourcePaths: string[];
  requestEntries: RequestEntry[];
  requestExactMap: Map<string, ProfileVector>;
  ruleEntries: RuleEntry[];
  ruleExactMap: Map<string, ProfileVector>;
  partMeanMap: Map<string, ProfileVector>;
  globalMean: ProfileVector;
};

export type DatasetRequestSample = {
  requestText: string;
  vector: ProfileVector;
};

export type DatasetVectorMatch = {
  vector: ProfileVector;
  strategy: "request_exact" | "request_fuzzy" | "rule_combo_exact" | "rule_combo_nearest" | "rule_part_average" | "global_mean";
  matchedText?: string;
  matchedSlots?: SlotSelection;
};

declare global {
  // eslint-disable-next-line no-var
  var __kotoneFtTestDatasetCache: LoadedDataset | undefined;
}

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const normalizeVector = (raw: unknown): ProfileVector | null => {
  if (!raw || typeof raw !== "object") return null;
  const payload = raw as Record<string, unknown>;
  const values: number[] = [];

  for (const key of VECTOR_KEYS) {
    const value = toFiniteNumber(payload[key]);
    if (value == null) return null;
    values.push(value);
  }

  const inferredScale = Math.max(...values) <= 10.5 ? 10 : 1;
  const normalized = values.map((value) => clamp(Math.round(value * inferredScale), 0, 100));

  return {
    energy: normalized[0],
    warmth: normalized[1],
    brightness: normalized[2],
    acousticness: normalized[3],
    complexity: normalized[4],
    nostalgia: normalized[5]
  };
};

const averageVectors = (vectors: ProfileVector[]): ProfileVector => {
  if (!vectors.length) return { ...VECTOR_FALLBACK };
  const totals = {
    energy: 0,
    warmth: 0,
    brightness: 0,
    acousticness: 0,
    complexity: 0,
    nostalgia: 0
  };
  for (const vector of vectors) {
    totals.energy += vector.energy;
    totals.warmth += vector.warmth;
    totals.brightness += vector.brightness;
    totals.acousticness += vector.acousticness;
    totals.complexity += vector.complexity;
    totals.nostalgia += vector.nostalgia;
  }
  const count = vectors.length;
  return {
    energy: Math.round(totals.energy / count),
    warmth: Math.round(totals.warmth / count),
    brightness: Math.round(totals.brightness / count),
    acousticness: Math.round(totals.acousticness / count),
    complexity: Math.round(totals.complexity / count),
    nostalgia: Math.round(totals.nostalgia / count)
  };
};

const normalizeText = (text: string): string => text.toLowerCase().replace(/\s+/g, " ").trim();

const tokenizeText = (text: string): Set<string> => {
  const normalized = normalizeText(text).replace(/[^a-z0-9\s]/g, " ");
  const tokens = normalized
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token.length >= 2);
  return new Set(tokens);
};

const jaccard = (left: Set<string>, right: Set<string>): number => {
  if (!left.size || !right.size) return 0;
  let intersection = 0;
  for (const token of left) {
    if (right.has(token)) intersection += 1;
  }
  const union = left.size + right.size - intersection;
  if (!union) return 0;
  return intersection / union;
};

const hashString = (text: string): number => {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

const comboKey = (slots: SlotSelection): string => SLOT_KEYS.map((slot) => slots[slot]).join("|");

const comboKeyToSlots = (key: string): SlotSelection | null => {
  const values = key.split("|").map((value) => value.trim());
  if (values.length !== SLOT_KEYS.length) return null;
  if (values.some((value) => value.length === 0)) return null;

  return {
    style: values[0]!,
    instrument: values[1]!,
    mood: values[2]!,
    gimmick: values[3]!
  };
};

const extractSlotPartId = (text: string, slot: SlotKey): string | null => {
  const pattern = new RegExp(`-\\s*${slot}\\s*:\\s*[^\\n]*\\(([^)]+)\\)`, "i");
  const match = text.match(pattern);
  const partId = match?.[1]?.trim() ?? "";
  return partId.length > 0 ? partId : null;
};

const stripRulePrefix = (text: string): string => text.replace(/^RULE_PROMPT:\s*/i, "").trim();

export const extractSelectedPartsFromRulePrompt = (text: string): SlotSelection | null => {
  const prompt = stripRulePrefix(text);
  const style = extractSlotPartId(prompt, "style");
  const instrument = extractSlotPartId(prompt, "instrument");
  const mood = extractSlotPartId(prompt, "mood");
  const gimmick = extractSlotPartId(prompt, "gimmick");
  if (!style || !instrument || !mood || !gimmick) return null;
  return { style, instrument, mood, gimmick };
};

const resolveDatasetPaths = (): string[] => {
  const envMulti = process.env.KOTONE_REFERENCE_DATASET_PATHS?.trim();
  if (envMulti) {
    const candidates = envMulti
      .split(/[,\n]/)
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
    if (candidates.length > 0) {
      return [...new Set(candidates.map((item) => resolve(process.cwd(), item)))];
    }
  }

  const envSingle = process.env.KOTONE_REFERENCE_DATASET_PATH?.trim();
  if (envSingle) {
    return [resolve(process.cwd(), envSingle)];
  }

  return DEFAULT_DATASET_PATHS.map((item) => resolve(process.cwd(), item));
};

const toAveragedMap = (aggregator: Map<string, ProfileVector[]>): Map<string, ProfileVector> => {
  return new Map(Array.from(aggregator.entries()).map(([key, vectors]) => [key, averageVectors(vectors)]));
};

const loadDataset = async (): Promise<LoadedDataset | null> => {
  const paths = resolveDatasetPaths();
  const sourceKey = paths.join("|");

  if (global.__kotoneFtTestDatasetCache?.sourceKey === sourceKey) {
    return global.__kotoneFtTestDatasetCache;
  }

  const requestEntries: RequestEntry[] = [];
  const requestExactAgg = new Map<string, ProfileVector[]>();
  const ruleEntries: RuleEntry[] = [];
  const ruleExactAgg = new Map<string, ProfileVector[]>();
  const partAgg = new Map<string, ProfileVector[]>();
  const allVectors: ProfileVector[] = [];
  let loadedFileCount = 0;

  for (const path of paths) {
    let content = "";
    try {
      content = await readFile(path, "utf8");
      loadedFileCount += 1;
    } catch {
      continue;
    }

    for (const line of content.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(trimmed) as Record<string, unknown>;
      } catch {
        continue;
      }

      const requestText = String(parsed.request_text ?? "").trim();
      if (!requestText) continue;

      const sourceType = String(parsed.source_type ?? "unknown").toLowerCase();
      const vector = normalizeVector((parsed.target_hidden_params as Record<string, unknown> | undefined)?.vector);
      if (!vector) continue;
      allVectors.push(vector);

      if (sourceType === "request_text") {
        const entry: RequestEntry = {
          requestText,
          vector,
          tokens: tokenizeText(requestText)
        };
        requestEntries.push(entry);
        const key = normalizeText(requestText);
        const bucket = requestExactAgg.get(key) ?? [];
        bucket.push(vector);
        requestExactAgg.set(key, bucket);
        continue;
      }

      if (sourceType === "rule_prompt") {
        const slots = extractSelectedPartsFromRulePrompt(requestText);
        if (!slots) continue;

        const entry: RuleEntry = {
          promptText: requestText,
          slots,
          vector
        };
        ruleEntries.push(entry);
        const key = comboKey(slots);
        const comboBucket = ruleExactAgg.get(key) ?? [];
        comboBucket.push(vector);
        ruleExactAgg.set(key, comboBucket);

        for (const slot of SLOT_KEYS) {
          const partId = slots[slot];
          const partBucket = partAgg.get(partId) ?? [];
          partBucket.push(vector);
          partAgg.set(partId, partBucket);
        }
      }
    }
  }

  if (!loadedFileCount || !allVectors.length) {
    return null;
  }

  const loaded: LoadedDataset = {
    sourceKey,
    sourcePaths: paths,
    requestEntries,
    requestExactMap: toAveragedMap(requestExactAgg),
    ruleEntries,
    ruleExactMap: toAveragedMap(ruleExactAgg),
    partMeanMap: toAveragedMap(partAgg),
    globalMean: averageVectors(allVectors)
  };
  global.__kotoneFtTestDatasetCache = loaded;
  return loaded;
};

export const pickRequestSampleFromTestDataset = async (seedHint?: string): Promise<DatasetRequestSample | null> => {
  const loaded = await loadDataset();
  if (!loaded || loaded.requestEntries.length === 0) return null;

  const index =
    typeof seedHint === "string" && seedHint.trim().length > 0
      ? hashString(seedHint.trim()) % loaded.requestEntries.length
      : Math.floor(Math.random() * loaded.requestEntries.length);
  const picked = loaded.requestEntries[index]!;

  return {
    requestText: picked.requestText,
    vector: picked.vector
  };
};

export const findVectorByRequestText = async (requestText: string): Promise<DatasetVectorMatch | null> => {
  const loaded = await loadDataset();
  if (!loaded) return null;

  const normalized = normalizeText(requestText);
  const exact = loaded.requestExactMap.get(normalized);
  if (exact) {
    return {
      vector: exact,
      strategy: "request_exact",
      matchedText: requestText
    };
  }

  const queryTokens = tokenizeText(requestText);
  if (!queryTokens.size || loaded.requestEntries.length === 0) return null;

  let best: RequestEntry | null = null;
  let bestScore = 0;
  for (const row of loaded.requestEntries) {
    const score = jaccard(queryTokens, row.tokens);
    if (score > bestScore) {
      best = row;
      bestScore = score;
    }
  }

  if (!best || bestScore < 0.42) return null;
  return {
    vector: best.vector,
    strategy: "request_fuzzy",
    matchedText: best.requestText
  };
};

export const findVectorByRulePromptOrSlots = async (input: {
  rulePrompt?: string;
  selectedPartsBySlot?: SlotSelection;
}): Promise<DatasetVectorMatch | null> => {
  const loaded = await loadDataset();
  if (!loaded) return null;

  const slots = input.selectedPartsBySlot ?? (input.rulePrompt ? extractSelectedPartsFromRulePrompt(input.rulePrompt) : null);
  if (!slots) return null;

  const key = comboKey(slots);
  const exact = loaded.ruleExactMap.get(key);
  if (exact) {
    return {
      vector: exact,
      strategy: "rule_combo_exact",
      matchedSlots: slots
    };
  }

  let best: RuleEntry | null = null;
  let bestOverlap = -1;
  for (const row of loaded.ruleEntries) {
    let overlap = 0;
    for (const slot of SLOT_KEYS) {
      if (row.slots[slot] === slots[slot]) overlap += 1;
    }
    if (overlap > bestOverlap) {
      bestOverlap = overlap;
      best = row;
    }
  }

  if (best && bestOverlap >= 2) {
    return {
      vector: best.vector,
      strategy: "rule_combo_nearest",
      matchedText: best.promptText,
      matchedSlots: best.slots
    };
  }

  const partVectors = SLOT_KEYS.map((slot) => loaded.partMeanMap.get(slots[slot])).filter(
    (vector): vector is ProfileVector => Boolean(vector)
  );
  if (partVectors.length) {
    return {
      vector: averageVectors(partVectors),
      strategy: "rule_part_average",
      matchedSlots: slots
    };
  }

  return {
    vector: loaded.globalMean,
    strategy: "global_mean",
    matchedSlots: slots
  };
};

export const isCombinationCovered = async (slots: SlotSelection): Promise<boolean> => {
  const loaded = await loadDataset();
  if (!loaded) return false;
  return loaded.ruleExactMap.has(comboKey(slots));
};

export const getCoveredCombinations = async (inventoryPartIds?: string[]): Promise<SlotSelection[]> => {
  const loaded = await loadDataset();
  if (!loaded) return [];

  const owned = inventoryPartIds ? new Set(inventoryPartIds.map((partId) => partId.trim()).filter(Boolean)) : null;

  const combos = Array.from(loaded.ruleExactMap.keys())
    .map((key) => comboKeyToSlots(key))
    .filter((combo): combo is SlotSelection => Boolean(combo))
    .filter((combo) => {
      if (!owned) return true;
      return SLOT_KEYS.every((slot) => owned.has(combo[slot]));
    })
    .sort((a, b) => comboKey(a).localeCompare(comboKey(b)));

  return combos;
};
