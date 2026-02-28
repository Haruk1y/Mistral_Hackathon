import type { Customer, Part, RequestTemplate, TargetProfile, Weather } from "@/lib/types";

const profile = (
  vector: TargetProfile["vector"],
  requiredTags: string[],
  optionalTags: string[] = [],
  forbiddenTags: string[] = [],
  constraints: TargetProfile["constraints"] = {}
): TargetProfile => ({ vector, requiredTags, optionalTags, forbiddenTags, constraints });

export const CATALOG_PARTS: Part[] = [
  {
    id: "style_80s_citypop",
    slot: "style",
    name: "80s City Pop",
    description: "都会的で温かい80年代シティポップ文法。",
    iconAsset: "/assets/ui/parts/part_style_placeholder.svg",
    price: 110,
    tags: ["citypop_80s", "groovy", "warm", "nostalgic"],
    vector: { energy: 52, warmth: 74, brightness: 62, complexity: 40, nostalgia: 78 }
  },
  {
    id: "style_90s_hiphop",
    slot: "style",
    name: "90s Hip-Hop",
    description: "90年代ヒップホップの太いビート感。",
    iconAsset: "/assets/ui/parts/part_style_placeholder.svg",
    price: 120,
    tags: ["hiphop_90s", "street", "steady", "night"],
    vector: { energy: 58, warmth: 52, brightness: 44, complexity: 46, nostalgia: 66 }
  },
  {
    id: "style_2000s_pop",
    slot: "style",
    name: "2000s Pop",
    description: "キャッチーで明るい2000年代ポップ。",
    iconAsset: "/assets/ui/parts/part_style_placeholder.svg",
    price: 130,
    tags: ["pop_2000s", "bright", "catchy", "festival"],
    vector: { energy: 76, warmth: 55, brightness: 82, complexity: 38, nostalgia: 48 }
  },
  {
    id: "inst_piano_upright",
    slot: "instrument",
    name: "Upright Piano",
    description: "木の響きが残るクラシックなピアノ。",
    iconAsset: "/assets/ui/parts/part_instrument_placeholder.svg",
    price: 150,
    tags: ["acoustic", "nostalgic", "cozy"],
    vector: { acousticness: 88, warmth: 70, nostalgia: 78, complexity: 44 }
  },
  {
    id: "inst_soft_strings",
    slot: "instrument",
    name: "Soft Strings",
    description: "優しい弦のレイヤー。",
    iconAsset: "/assets/ui/parts/part_instrument_placeholder.svg",
    price: 140,
    tags: ["emotional", "warm", "cinematic"],
    vector: { warmth: 75, acousticness: 72, complexity: 50, nostalgia: 64 }
  },
  {
    id: "inst_analog_synth",
    slot: "instrument",
    name: "Analog Synth",
    description: "角の丸いアナログシンセ主旋律。",
    iconAsset: "/assets/ui/parts/part_instrument_placeholder.svg",
    price: 145,
    tags: ["electronic", "retro", "night"],
    vector: { energy: 68, brightness: 64, acousticness: 24, complexity: 52, nostalgia: 58 }
  },
  {
    id: "mood_rain_ambience",
    slot: "mood",
    name: "Rain Ambience",
    description: "静かな雨音のアンビエンス。",
    iconAsset: "/assets/ui/parts/part_mood_placeholder.svg",
    price: 70,
    tags: ["rain", "calm", "nostalgic"],
    vector: { warmth: 50, brightness: 28, nostalgia: 84, complexity: 20 }
  },
  {
    id: "mood_sun_glow",
    slot: "mood",
    name: "Sun Glow",
    description: "夕暮れの温度感を加える。",
    iconAsset: "/assets/ui/parts/part_mood_placeholder.svg",
    price: 85,
    tags: ["warm", "hopeful", "sunset"],
    vector: { warmth: 84, brightness: 74, nostalgia: 58 }
  },
  {
    id: "mood_night_drive",
    slot: "mood",
    name: "Night Drive",
    description: "夜道を流すようなクールな空気。",
    iconAsset: "/assets/ui/parts/part_mood_placeholder.svg",
    price: 95,
    tags: ["night", "urban", "dark"],
    vector: { energy: 55, warmth: 42, brightness: 36, complexity: 46, nostalgia: 50 }
  },
  {
    id: "gimmick_beat_mute",
    slot: "gimmick",
    name: "Beat Mute",
    description: "一瞬だけビートを抜いて戻すフック。",
    iconAsset: "/assets/ui/parts/part_gimmick_placeholder.svg",
    price: 90,
    tags: ["beat_mute", "hook", "dynamic"],
    vector: { complexity: 52, energy: 60, nostalgia: 44 }
  },
  {
    id: "gimmick_filter_rise",
    slot: "gimmick",
    name: "Filter Rise",
    description: "フィルター上昇でサビ前を煽る。",
    iconAsset: "/assets/ui/parts/part_gimmick_placeholder.svg",
    price: 100,
    tags: ["filter_rise", "build", "bright"],
    vector: { complexity: 56, energy: 72, brightness: 68 }
  },
  {
    id: "gimmick_harmony_stack",
    slot: "gimmick",
    name: "Harmony Stack",
    description: "終盤で和声レイヤーを重ねる演出。",
    iconAsset: "/assets/ui/parts/part_gimmick_placeholder.svg",
    price: 105,
    tags: ["harmony_stack", "lush", "emotional"],
    vector: { warmth: 70, complexity: 58, nostalgia: 62 }
  }
];

export const STARTER_PART_IDS = [
  "style_80s_citypop",
  "style_90s_hiphop",
  "inst_piano_upright",
  "inst_soft_strings",
  "mood_rain_ambience",
  "mood_sun_glow",
  "gimmick_beat_mute"
];

export const SHOP_DEFAULT_STOCK = CATALOG_PARTS.map((part) => ({
  partId: part.id,
  unlocked: !STARTER_PART_IDS.includes(part.id)
}));

export const CATALOG_CUSTOMERS: Customer[] = [
  {
    id: "anna",
    name: "Momo",
    portraitAsset: "/assets/placeholders/portrait.svg",
    personality: "赤いおだんご髪の配達見習い。朝市に合う軽やかな曲が好き。"
  },
  {
    id: "ben",
    name: "Theo",
    portraitAsset: "/assets/placeholders/portrait.svg",
    personality: "丸メガネの書店手伝い。ページをめくる手が進む穏やかな曲を好む。"
  },
  {
    id: "cara",
    name: "Irene",
    portraitAsset: "/assets/placeholders/portrait.svg",
    personality: "眼鏡の仕立て屋。夕暮れに似合う上品で落ち着いた旋律を求める。"
  },
  {
    id: "dan",
    name: "Gideon",
    portraitAsset: "/assets/placeholders/portrait.svg",
    personality: "白髭の旅人。古い街の思い出を呼び起こす温かな音を探している。"
  }
];

export const REQUEST_TEMPLATES: RequestTemplate[] = [
  {
    id: "req_quiet_rain_evening",
    customerId: "anna",
    text: "Can you make something that feels like a quiet evening after rain?",
    weatherBias: ["rainy", "cloudy"],
    targetProfile: profile(
      { energy: 22, warmth: 62, brightness: 30, acousticness: 76, complexity: 34, nostalgia: 82 },
      ["rain", "nostalgic"],
      ["calm", "acoustic", "citypop_80s", "beat_mute"],
      ["festival"],
      { preferredStyleTags: ["citypop_80s", "hiphop_90s"], preferredGimmickTags: ["beat_mute", "filter_rise"] }
    )
  },
  {
    id: "req_market_smile",
    customerId: "ben",
    text: "I want a tune that makes passersby smile at the market.",
    weatherBias: ["sunny", "cloudy"],
    targetProfile: profile(
      { energy: 66, warmth: 72, brightness: 74, acousticness: 58, complexity: 36, nostalgia: 48 },
      ["bright", "catchy"],
      ["playful", "hopeful", "pop_2000s", "filter_rise"],
      ["dark"],
      { preferredStyleTags: ["pop_2000s"], preferredGimmickTags: ["filter_rise", "harmony_stack"] }
    )
  },
  {
    id: "req_focus_gentle",
    customerId: "cara",
    text: "Please compose calm music that helps me focus while studying.",
    weatherBias: ["sunny", "cloudy", "rainy"],
    targetProfile: profile(
      { energy: 32, warmth: 58, brightness: 48, acousticness: 68, complexity: 22, nostalgia: 56 },
      ["calm", "steady"],
      ["cozy", "acoustic"],
      ["festival", "chaotic"],
      { preferredStyleTags: ["citypop_80s", "hiphop_90s"], preferredGimmickTags: ["beat_mute"] }
    )
  },
  {
    id: "req_oldtown_story",
    customerId: "dan",
    text: "Could you play something that sounds like an old town memory?",
    weatherBias: ["cloudy", "rainy", "sunny"],
    targetProfile: profile(
      { energy: 40, warmth: 68, brightness: 44, acousticness: 80, complexity: 42, nostalgia: 88 },
      ["nostalgic", "warm"],
      ["acoustic", "night", "harmony_stack"],
      ["aggressive"],
      { preferredStyleTags: ["citypop_80s"], preferredGimmickTags: ["harmony_stack", "beat_mute"] }
    )
  }
];

export const WEATHER_OPTIONS: Weather[] = ["sunny", "cloudy", "rainy"];

export const WEATHER_LABELS: Record<Weather, string> = {
  sunny: "SUNNY",
  cloudy: "CLOUDY",
  rainy: "RAINY"
};

export const WEATHER_ICON: Record<Weather, string> = {
  sunny: "☀",
  cloudy: "☁",
  rainy: "☔"
};
