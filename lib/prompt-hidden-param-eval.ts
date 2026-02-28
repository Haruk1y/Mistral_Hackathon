import type {
  HiddenParamVectorKey,
  ProfileVector,
  PromptHiddenParamEval,
  TargetHiddenParams
} from "@/lib/types";

const VECTOR_KEYS: HiddenParamVectorKey[] = [
  "energy",
  "warmth",
  "brightness",
  "acousticness",
  "complexity",
  "nostalgia"
];

const DIMENSION_ACTION_HINT: Record<HiddenParamVectorKey, string> = {
  energy: "Promptでテンポ感・躍動語（driving, upbeat など）を増減し、energyを明示する。",
  warmth: "音色語（warm, mellow, woody）を具体化し、warmthの方向を固定する。",
  brightness: "高域や質感（bright highs, airy top end）を明示し、brightness誤差を抑える。",
  acousticness: "acoustic / organic / live-instrument を明示し、電子要素の比率を指示する。",
  complexity: "構成語（minimal / layered / dense）を追加してcomplexityを制御する。",
  nostalgia: "era語（80s/90s retro, memory, vintage）を増やしnostalgiaを誘導する。"
};

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

const jaccard = (a: string[], b: string[]) => {
  const sa = new Set(a);
  const sb = new Set(b);

  if (sa.size === 0 && sb.size === 0) {
    return 1;
  }

  let intersection = 0;
  for (const tag of sa) {
    if (sb.has(tag)) intersection += 1;
  }

  const union = sa.size + sb.size - intersection;
  if (union <= 0) return 0;
  return intersection / union;
};

const mean = (values: number[]) => {
  if (values.length === 0) return 0;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
};

const buildSummary = (input: {
  maeRawMean: number;
  maxErrorDim: HiddenParamVectorKey;
  maxErrorValue: number;
  tagsJaccard: number;
}): string => {
  return `Prompt再推定の平均MAE=${input.maeRawMean.toFixed(2)}、最大誤差は ${input.maxErrorDim} (${input.maxErrorValue.toFixed(
    2
  )})、タグ一致(Jaccard)=${input.tagsJaccard.toFixed(2)}。`;
};

export const evaluatePromptHiddenParams = (
  target: TargetHiddenParams,
  predicted: TargetHiddenParams
): PromptHiddenParamEval => {
  const maeRawByDim = Object.fromEntries(
    VECTOR_KEYS.map((key) => [key, Math.abs(target.vector[key] - predicted.vector[key])])
  ) as Record<HiddenParamVectorKey, number>;

  const squaredErrors = VECTOR_KEYS.map((key) => Math.pow(target.vector[key] - predicted.vector[key], 2));
  const maeRawValues = VECTOR_KEYS.map((key) => maeRawByDim[key]);
  const mseRaw = mean(squaredErrors);
  const maeRawMean = mean(maeRawValues);

  const maxErrorDim = VECTOR_KEYS.reduce(
    (worst, key) => (maeRawByDim[key] > maeRawByDim[worst] ? key : worst),
    VECTOR_KEYS[0]
  );
  const maxErrorValue = maeRawByDim[maxErrorDim];

  const tagsJaccard = clamp(jaccard(target.tags, predicted.tags), 0, 1);

  return {
    mae_raw_mean: maeRawMean,
    mse_raw: mseRaw,
    mae_raw_by_dim: maeRawByDim,
    max_error_dim: maxErrorDim,
    max_error_value: maxErrorValue,
    tags_jaccard: tagsJaccard,
    summary: buildSummary({
      maeRawMean,
      maxErrorDim,
      maxErrorValue,
      tagsJaccard
    }),
    next_actions: [
      DIMENSION_ACTION_HINT[maxErrorDim],
      "rule_promptに数値意図を補足する（例: low energy / high nostalgia）。",
      "誤差が大きい次元を hard case として次サイクルの学習データへ追加する。"
    ]
  };
};

export const buildPromptEvalFeedback = (
  evalResult: PromptHiddenParamEval | undefined,
  vector: ProfileVector | undefined
): string => {
  if (!evalResult) {
    return "Prompt再推定が未実行です。";
  }

  const targetHint = vector
    ? `目標: energy=${vector.energy}, warmth=${vector.warmth}, brightness=${vector.brightness}, acousticness=${vector.acousticness}, complexity=${vector.complexity}, nostalgia=${vector.nostalgia}. `
    : "";

  return `${targetHint}${evalResult.summary} 次アクション: ${evalResult.next_actions[0]}`;
};
