import type { Customer, Part, RequestTemplate, TargetProfile, Weather } from "@/lib/types";

const profile = (
  vector: TargetProfile["vector"],
  requiredTags: string[],
  optionalTags: string[] = [],
  forbiddenTags: string[] = [],
  constraints: TargetProfile["constraints"] = {}
): TargetProfile => ({ vector, requiredTags, optionalTags, forbiddenTags, constraints });

export const CATALOG_PARTS: Part[] = [
  // style
  {
    id: "style_80s_citypop",
    slot: "style",
    name: "80s City Pop",
    description: "都会的で温かい80年代シティポップ文法。",
    iconAsset: "/assets/parts/named/style/slot_style_tide_sway.png",
    price: 110,
    tags: ["citypop_80s", "groovy", "warm", "nostalgic"],
    vector: { energy: 52, warmth: 74, brightness: 62, complexity: 40, nostalgia: 78 }
  },
  {
    id: "style_90s_hiphop",
    slot: "style",
    name: "90s Hip-Hop",
    description: "90年代ヒップホップの太いビート感。",
    iconAsset: "/assets/parts/named/style/slot_style_kick_step.png",
    price: 120,
    tags: ["hiphop_90s", "street", "steady", "night"],
    vector: { energy: 58, warmth: 52, brightness: 44, complexity: 46, nostalgia: 66 }
  },
  {
    id: "style_2000s_pop",
    slot: "style",
    name: "2000s Pop",
    description: "キャッチーで明るい2000年代ポップ。",
    iconAsset: "/assets/parts/named/style/slot_style_brush_smile.png",
    price: 130,
    tags: ["pop_2000s", "bright", "catchy", "festival"],
    vector: { energy: 76, warmth: 55, brightness: 82, complexity: 38, nostalgia: 48 }
  },
  {
    id: "style_70s_folk",
    slot: "style",
    name: "70s Folk",
    description: "木の香りが残る70年代フォークの素朴な歩幅。",
    iconAsset: "/assets/parts/named/style/slot_style_log_roll.png",
    price: 115,
    tags: ["folk_70s", "acoustic", "warm", "nostalgic"],
    vector: { energy: 44, warmth: 78, brightness: 46, complexity: 34, nostalgia: 86 }
  },
  {
    id: "style_2010s_edm",
    slot: "style",
    name: "2010s EDM",
    description: "カウントダウンで高揚を作る2010年代EDM。",
    iconAsset: "/assets/parts/named/style/slot_style_countdown_whisp.png",
    price: 145,
    tags: ["edm_2010s", "build", "bright", "digital"],
    vector: { energy: 86, warmth: 38, brightness: 90, complexity: 58, nostalgia: 34 }
  },
  {
    id: "style_60s_parade_jazz",
    slot: "style",
    name: "60s Parade Jazz",
    description: "行進のスネア感が心地よい60年代パレードジャズ。",
    iconAsset: "/assets/parts/named/style/slot_style_march_snare.png",
    price: 125,
    tags: ["jazz_60s", "parade", "steady", "brassy"],
    vector: { energy: 66, warmth: 63, brightness: 60, complexity: 56, nostalgia: 74 }
  },
  {
    id: "style_50s_clockwork_swing",
    slot: "style",
    name: "50s Clockwork Swing",
    description: "チクタクした推進感を持つ50年代スウィング。",
    iconAsset: "/assets/parts/named/style/slot_style_tick_tock_metoro.png",
    price: 135,
    tags: ["swing_50s", "clockwork", "playful", "nostalgic"],
    vector: { energy: 58, warmth: 54, brightness: 56, complexity: 64, nostalgia: 82 }
  },
  {
    id: "style_2010s_dance_pop",
    slot: "style",
    name: "2010s Dance Pop",
    description: "ブーツの跳ねを感じる2010年代ダンスポップ。",
    iconAsset: "/assets/parts/named/style/slot_style_boots_bounce.png",
    price: 140,
    tags: ["dancepop_2010s", "upbeat", "catchy", "festival"],
    vector: { energy: 82, warmth: 57, brightness: 78, complexity: 46, nostalgia: 40 }
  },

  // instrument
  {
    id: "inst_piano_upright",
    slot: "instrument",
    name: "Upright Piano",
    description: "木の響きが残るクラシックなピアノ。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_wood_upright_piano.png",
    price: 150,
    tags: ["acoustic", "nostalgic", "cozy"],
    vector: { acousticness: 88, warmth: 70, nostalgia: 78, complexity: 44 }
  },
  {
    id: "inst_soft_strings",
    slot: "instrument",
    name: "Fairy Harp",
    description: "淡くきらめくハープで空気を柔らかく包む。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_fairy_harp.png",
    price: 140,
    tags: ["emotional", "warm", "cinematic"],
    vector: { warmth: 75, acousticness: 72, complexity: 50, nostalgia: 64 }
  },
  {
    id: "inst_analog_synth",
    slot: "instrument",
    name: "Snake Music Box",
    description: "蛇模様の機械式オルゴールで不思議な余韻を残す。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_musicbox_snake.png",
    price: 145,
    tags: ["mechanical", "retro", "quirky", "night"],
    vector: { energy: 52, brightness: 58, acousticness: 36, complexity: 54, nostalgia: 70 }
  },
  {
    id: "inst_violin_bright",
    slot: "instrument",
    name: "Bright Violin",
    description: "前に抜ける高音で主旋律を照らすヴァイオリン。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_bright_violin.png",
    price: 158,
    tags: ["acoustic", "lead", "bright"],
    vector: { energy: 64, warmth: 56, brightness: 78, acousticness: 82, complexity: 58 }
  },
  {
    id: "inst_cello_warm",
    slot: "instrument",
    name: "Warm Cello",
    description: "低音の深みで曲の芯を作るチェロ。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_cello_earth.png",
    price: 160,
    tags: ["acoustic", "warm", "deep", "emotional"],
    vector: { energy: 42, warmth: 86, brightness: 34, acousticness: 88, complexity: 52, nostalgia: 72 }
  },
  {
    id: "inst_musicbox_garden",
    slot: "instrument",
    name: "Garden Music Box",
    description: "庭園の情景に合う澄んだオルゴール音。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_musicbox_garden.png",
    price: 134,
    tags: ["musicbox", "delicate", "nostalgic"],
    vector: { energy: 38, warmth: 62, brightness: 66, acousticness: 42, complexity: 40, nostalgia: 84 }
  },
  {
    id: "inst_clarinet_oak",
    slot: "instrument",
    name: "Oak Clarinet",
    description: "丸みのある木管の息遣いを加えるクラリネット。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_oak_clarinet.png",
    price: 152,
    tags: ["acoustic", "woodwind", "warm"],
    vector: { energy: 46, warmth: 73, brightness: 52, acousticness: 86, complexity: 54, nostalgia: 68 }
  },
  {
    id: "inst_tambourine_river",
    slot: "instrument",
    name: "River Tambourine",
    description: "水面の反射のように細かく刻むタンバリン。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_river_tambourine.png",
    price: 124,
    tags: ["percussion", "rhythm", "shimmer"],
    vector: { energy: 70, warmth: 46, brightness: 71, acousticness: 62, complexity: 48 }
  },
  {
    id: "inst_mandolin_round",
    slot: "instrument",
    name: "Round Mandolin",
    description: "まろやかな輪郭で和音を支えるマンドリン。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_round_mandolin.png",
    price: 146,
    tags: ["acoustic", "pluck", "rounded"],
    vector: { energy: 53, warmth: 67, brightness: 63, acousticness: 81, complexity: 51, nostalgia: 64 }
  },
  {
    id: "inst_guitar_street",
    slot: "instrument",
    name: "Street Guitar",
    description: "路地裏ライブの空気を持つ木製ギター。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_street_guitar.png",
    price: 154,
    tags: ["acoustic", "street", "groovy"],
    vector: { energy: 65, warmth: 69, brightness: 58, acousticness: 84, complexity: 50, nostalgia: 66 }
  },
  {
    id: "inst_lute_sun",
    slot: "instrument",
    name: "Sun Lute",
    description: "明るいタッチで前向きな印象を作るリュート。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_sun_lute.png",
    price: 141,
    tags: ["acoustic", "bright", "folk"],
    vector: { energy: 58, warmth: 70, brightness: 72, acousticness: 83, complexity: 47, nostalgia: 58 }
  },
  {
    id: "inst_recorder_tall",
    slot: "instrument",
    name: "Tall Recorder",
    description: "素直な高音で旋律をまっすぐ通すリコーダー。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_tall_recorder.png",
    price: 132,
    tags: ["acoustic", "woodwind", "clear"],
    vector: { energy: 57, warmth: 53, brightness: 76, acousticness: 90, complexity: 45, nostalgia: 54 }
  },
  {
    id: "inst_accordion_travel",
    slot: "instrument",
    name: "Travel Accordion",
    description: "旅芸人のような躍動感を加えるアコーディオン。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_travel_accordion.png",
    price: 168,
    tags: ["acoustic", "festival", "folk", "street"],
    vector: { energy: 74, warmth: 71, brightness: 63, acousticness: 78, complexity: 55, nostalgia: 62 }
  },
  {
    id: "inst_violin_vintage",
    slot: "instrument",
    name: "Vintage Violin",
    description: "かすれの味を残したヴィンテージヴァイオリン。",
    iconAsset: "/assets/parts/named/instrument/slot_instrument_vintage_violin.png",
    price: 162,
    tags: ["acoustic", "vintage", "emotional", "nostalgic"],
    vector: { energy: 50, warmth: 71, brightness: 60, acousticness: 84, complexity: 59, nostalgia: 80 }
  },

  // mood
  {
    id: "mood_rain_ambience",
    slot: "mood",
    name: "Rain Ambience",
    description: "静かな雨音のアンビエンス。",
    iconAsset: "/assets/parts/named/mood/slot_mood_rain_heartbreak.png",
    price: 70,
    tags: ["rain", "calm", "nostalgic"],
    vector: { warmth: 50, brightness: 28, nostalgia: 84, complexity: 20 }
  },
  {
    id: "mood_sun_glow",
    slot: "mood",
    name: "Sun Glow",
    description: "夕暮れの温度感を加える。",
    iconAsset: "/assets/parts/named/mood/slot_mood_sunny_nap.png",
    price: 85,
    tags: ["warm", "hopeful", "sunset"],
    vector: { warmth: 84, brightness: 74, nostalgia: 58 }
  },
  {
    id: "mood_night_drive",
    slot: "mood",
    name: "Night Drive",
    description: "夜道を流すようなクールな空気。",
    iconAsset: "/assets/parts/named/mood/slot_mood_moon_calm.png",
    price: 95,
    tags: ["night", "urban", "dark"],
    vector: { energy: 55, warmth: 42, brightness: 36, complexity: 46, nostalgia: 50 }
  },
  {
    id: "mood_cozy_hearth",
    slot: "mood",
    name: "Cozy Hearth",
    description: "暖炉のそばにいるような安心感を足す。",
    iconAsset: "/assets/parts/named/mood/slot_mood_cozy_hearth.png",
    price: 88,
    tags: ["cozy", "warm", "home"],
    vector: { energy: 34, warmth: 90, brightness: 48, complexity: 20, nostalgia: 72 }
  },
  {
    id: "mood_hope_star",
    slot: "mood",
    name: "Hope Star",
    description: "希望の星が差し込むような前向きな空気。",
    iconAsset: "/assets/parts/named/mood/slot_mood_hope_star.png",
    price: 92,
    tags: ["hopeful", "bright", "uplift"],
    vector: { energy: 62, warmth: 68, brightness: 80, complexity: 24, nostalgia: 46 }
  },
  {
    id: "mood_lonely_star",
    slot: "mood",
    name: "Lonely Star",
    description: "一人きりの星空みたいな静かな寂しさ。",
    iconAsset: "/assets/parts/named/mood/slot_mood_lonely_star.png",
    price: 90,
    tags: ["lonely", "night", "calm"],
    vector: { energy: 28, warmth: 36, brightness: 34, complexity: 28, nostalgia: 74 }
  },
  {
    id: "mood_mystic_hood",
    slot: "mood",
    name: "Mystic Hood",
    description: "神秘的な気配で曲に奥行きを作る。",
    iconAsset: "/assets/parts/named/mood/slot_mood_mystic_hood.png",
    price: 98,
    tags: ["mystic", "dark", "cinematic"],
    vector: { energy: 40, warmth: 38, brightness: 30, complexity: 55, nostalgia: 62 }
  },
  {
    id: "mood_party_face",
    slot: "mood",
    name: "Party Face",
    description: "笑顔が連鎖するパーティーの温度感。",
    iconAsset: "/assets/parts/named/mood/slot_mood_party_face.png",
    price: 102,
    tags: ["party", "playful", "festival"],
    vector: { energy: 84, warmth: 64, brightness: 86, complexity: 30, nostalgia: 34 }
  },
  {
    id: "mood_pocket_memory",
    slot: "mood",
    name: "Pocket Memory",
    description: "ポケットの古い写真のようなやさしい追憶。",
    iconAsset: "/assets/parts/named/mood/slot_mood_pocket_memory.png",
    price: 94,
    tags: ["nostalgic", "soft", "memory"],
    vector: { energy: 36, warmth: 72, brightness: 50, complexity: 26, nostalgia: 90 }
  },
  {
    id: "mood_sun_laugh",
    slot: "mood",
    name: "Sun Laugh",
    description: "太陽のように明るく弾む笑い声の空気。",
    iconAsset: "/assets/parts/named/mood/slot_mood_sun_laugh.png",
    price: 99,
    tags: ["bright", "playful", "daytime"],
    vector: { energy: 78, warmth: 75, brightness: 92, complexity: 22, nostalgia: 38 }
  },

  // gimmick
  {
    id: "gimmick_beat_mute",
    slot: "gimmick",
    name: "Beat Mute",
    description: "一瞬だけビートを抜いて戻すフック。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_boxy_lofi.png",
    price: 90,
    tags: ["beat_mute", "hook", "dynamic"],
    vector: { complexity: 52, energy: 60, nostalgia: 44 }
  },
  {
    id: "gimmick_filter_rise",
    slot: "gimmick",
    name: "Filter Rise",
    description: "フィルター上昇でサビ前を煽る。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_leaf_gust.png",
    price: 100,
    tags: ["filter_rise", "build", "bright"],
    vector: { complexity: 56, energy: 72, brightness: 68 }
  },
  {
    id: "gimmick_harmony_stack",
    slot: "gimmick",
    name: "Harmony Stack",
    description: "終盤で和声レイヤーを重ねる演出。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_lantern_glow.png",
    price: 105,
    tags: ["harmony_stack", "lush", "emotional"],
    vector: { warmth: 70, complexity: 58, nostalgia: 62 }
  },
  {
    id: "gimmick_campfire_crackle",
    slot: "gimmick",
    name: "Campfire Crackle",
    description: "焚き火のパチパチ音で温度感を足す。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_campfire_crackle.png",
    price: 96,
    tags: ["texture", "warm", "acoustic"],
    vector: { energy: 36, warmth: 82, brightness: 34, complexity: 42, nostalgia: 78 }
  },
  {
    id: "gimmick_chatty_crowd",
    slot: "gimmick",
    name: "Chatty Crowd",
    description: "賑やかな雑談を薄く重ねて臨場感を演出。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_chatty_crowd.png",
    price: 92,
    tags: ["crowd", "street", "texture"],
    vector: { energy: 62, warmth: 54, brightness: 56, complexity: 46 }
  },
  {
    id: "gimmick_cricket_pulse",
    slot: "gimmick",
    name: "Cricket Pulse",
    description: "虫の鳴きリズムをパルスのように配置する。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_cricket_pulse.png",
    price: 93,
    tags: ["nature", "pulse", "night"],
    vector: { energy: 48, warmth: 46, brightness: 42, complexity: 50, nostalgia: 60 }
  },
  {
    id: "gimmick_moonwind_spark",
    slot: "gimmick",
    name: "Moonwind Spark",
    description: "月風に小さな火花を混ぜたようなアクセント。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_moonwind_spark.png",
    price: 101,
    tags: ["night", "spark", "accent"],
    vector: { energy: 50, warmth: 45, brightness: 64, complexity: 52 }
  },
  {
    id: "gimmick_rainfall",
    slot: "gimmick",
    name: "Rainfall Layer",
    description: "雨粒のレイヤーでしっとりした情景を描く。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_rainfall.png",
    price: 95,
    tags: ["rain", "ambient", "calm"],
    vector: { energy: 28, warmth: 46, brightness: 32, complexity: 40, nostalgia: 72 }
  },
  {
    id: "gimmick_river_flow",
    slot: "gimmick",
    name: "River Flow",
    description: "川の流れのような連続音で曲をつなぐ。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_river_flow.png",
    price: 96,
    tags: ["water", "flow", "ambient"],
    vector: { energy: 34, warmth: 57, brightness: 44, complexity: 44, nostalgia: 63 }
  },
  {
    id: "gimmick_stone_rattle",
    slot: "gimmick",
    name: "Stone Rattle",
    description: "石を鳴らすような乾いた粒立ちでリズム補強。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_stone_rattle.png",
    price: 99,
    tags: ["percussive", "dry", "texture"],
    vector: { energy: 66, warmth: 36, brightness: 51, complexity: 55 }
  },
  {
    id: "gimmick_temple_bell",
    slot: "gimmick",
    name: "Temple Bell",
    description: "鐘の一打でセクションの切り替えを印象付ける。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_temple_bell.png",
    price: 110,
    tags: ["bell", "transition", "ceremonial"],
    vector: { energy: 46, warmth: 68, brightness: 49, complexity: 53, nostalgia: 76 }
  },
  {
    id: "gimmick_thunder_pop",
    slot: "gimmick",
    name: "Thunder Pop",
    description: "雷のような一撃でドロップ前を強調する。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_thunder_pop.png",
    price: 116,
    tags: ["impact", "drop", "storm"],
    vector: { energy: 88, warmth: 30, brightness: 62, complexity: 57 }
  },
  {
    id: "gimmick_whisper_left",
    slot: "gimmick",
    name: "Whisper Left",
    description: "左側にささやきを置いて空間演出を作る。",
    iconAsset: "/assets/parts/named/gimmick/slot_gimmick_whisper_left.png",
    price: 87,
    tags: ["whisper", "stereo", "subtle"],
    vector: { energy: 24, warmth: 43, brightness: 37, complexity: 46, nostalgia: 52 }
  }
];

export const STARTER_PART_IDS = [
  "style_80s_citypop",
  "style_90s_hiphop",
  "style_2000s_pop",
  "inst_piano_upright",
  "inst_soft_strings",
  "inst_guitar_street",
  "mood_rain_ambience",
  "mood_sun_glow",
  "mood_night_drive",
  "gimmick_beat_mute",
  "gimmick_filter_rise",
  "gimmick_harmony_stack"
];

export const SHOP_DEFAULT_STOCK = CATALOG_PARTS.map((part) => ({
  partId: part.id,
  unlocked: !STARTER_PART_IDS.includes(part.id)
}));

export const CATALOG_CUSTOMERS: Customer[] = [
  {
    id: "anna",
    name: "こはる",
    portraitAsset: "/assets/characters/portraits/street-crowd-v2/por_crowd_06_face_160.png",
    personality: "赤いおだんご髪の配達見習い。朝市に合う軽やかな曲が好き。"
  },
  {
    id: "ben",
    name: "湊",
    portraitAsset: "/assets/characters/portraits/street-crowd-v2/por_crowd_09_face_160.png",
    personality: "丸メガネの書店手伝い。ページをめくる手が進む穏やかな曲を好む。"
  },
  {
    id: "cara",
    name: "澄江",
    portraitAsset: "/assets/characters/portraits/street-crowd-v2/por_crowd_14_face_160.png",
    personality: "眼鏡の仕立て屋。夕暮れに似合う上品で落ち着いた旋律を求める。"
  },
  {
    id: "dan",
    name: "宗玄",
    portraitAsset: "/assets/characters/portraits/street-crowd-v2/por_crowd_20_face_160.png",
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

export const WEATHER_OPTIONS: Weather[] = ["sunny", "cloudy"];

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
