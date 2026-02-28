export const STYLE_SUFFIX =
  "nostalgic soft farming-town pixel-art, warm muted earthy palette, gentle contrast, clean 1px outlines, subtle dithering, cozy handcrafted atmosphere, consistent wood-paper-cobblestone material language, avoid harsh saturation and glossy highlights, no modern UI style, create original visuals without imitating any specific game's exact assets";

export const NEGATIVE_SUFFIX =
  "no text, no logo, no watermark, no signature, no realistic photo, no 3d render, no blur";

export const SHEET_KINDS = [
  "character_master",
  "portrait_sheet",
  "ui_sheet",
  "fx_sheet",
  "bg_sheet"
];

const MASTER_SHEET_IMAGE_SIZE_BY_KIND = {
  character_master: "2K",
  portrait_sheet: "2K",
  ui_sheet: "2K",
  fx_sheet: "2K",
  bg_sheet: "4K"
};

function inferKind(spec) {
  if (spec.id.startsWith("character_master_")) return "character_master";
  if (spec.id.startsWith("portrait_sheet_")) return "portrait_sheet";
  if (spec.id === "fx_sheet") return "fx_sheet";
  if (spec.id.startsWith("bg_") || spec.tags.includes("background")) return "bg_sheet";
  if (
    spec.tags.includes("ui") ||
    spec.id.includes("_ui_") ||
    spec.id.endsWith("_ui_sheet") ||
    spec.id.includes("interaction_sheet") ||
    spec.id.includes("icon_sheet") ||
    spec.id.startsWith("otoword_")
  ) {
    return "ui_sheet";
  }

  // Non-master-sheet sprite specs are treated as character assets for V2 defaults.
  return "character_master";
}

function inferQaProfile(kind) {
  if (kind === "bg_sheet") return "light";
  if (kind === "character_master" || kind === "portrait_sheet") return "strict";
  return "standard";
}

function isMasterSheetSpec(spec) {
  const tags = Array.isArray(spec.tags) ? spec.tags : [];
  return tags.includes("master-sheet") || tags.includes("mvp-master-sheet");
}

function inferImageSize(spec, kind) {
  if (!isMasterSheetSpec(spec)) return undefined;
  return MASTER_SHEET_IMAGE_SIZE_BY_KIND[kind] || "2K";
}

function inferImageAspectRatio(spec) {
  if (!isMasterSheetSpec(spec)) return undefined;
  const width = Number(spec.width);
  const height = Number(spec.height);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return undefined;
  }

  const ratio = width / height;
  if (Math.abs(ratio - 1) < 0.01) return "1:1";
  if (Math.abs(ratio - 2) < 0.01) return "2:1";
  if (Math.abs(ratio - 16 / 9) < 0.01) return "16:9";
  if (Math.abs(ratio - 4 / 3) < 0.01) return "4:3";
  if (Math.abs(ratio - 3 / 2) < 0.01) return "3:2";
  return undefined;
}

function withV2Defaults(spec) {
  const kind = spec.kind || inferKind(spec);
  return {
    ...spec,
    kind,
    styleGroup: spec.styleGroup || "soft_nostalgic_town_v1",
    alphaRequired: spec.alphaRequired ?? true,
    qaProfile: spec.qaProfile || inferQaProfile(kind),
    rewriteEnabled: spec.rewriteEnabled ?? true,
    imageSize: spec.imageSize || inferImageSize(spec, kind),
    imageAspectRatio: spec.imageAspectRatio || inferImageAspectRatio(spec)
  };
}

const RAW_ASSET_SPECS = [
  {
    id: "bg_street_day",
    outputPath: "public/assets/bg/bg_street_day_960x540.png",
    width: 960,
    height: 540,
    tags: ["mvp", "background"],
    prompt:
      "Street view of a cozy old-town market road in daylight, cobblestone street, warm sunlight, composition with open center area for UI overlay, side-scrolling game background feeling."
  },
  {
    id: "bg_street_rain",
    outputPath: "public/assets/bg/bg_street_rain_960x540.png",
    width: 960,
    height: 540,
    tags: ["mvp", "background"],
    prompt:
      "Street view of the same old-town market road in rainy weather, wet cobblestone reflections, warm shop lights, moody but cozy atmosphere, composition with open center area for UI overlay."
  },
  {
    id: "bg_shop",
    outputPath: "public/assets/bg/bg_shop_960x540.png",
    width: 960,
    height: 540,
    tags: ["extended", "background"],
    prompt:
      "Inside a compact music supplies shop in retro pixel-art style, wooden shelves, instruments and records, warm lamp lighting, open center area for UI."
  },
  {
    id: "bg_compose_rain",
    outputPath: "public/assets/bg/bg_compose_rain_960x540.png",
    width: 960,
    height: 540,
    tags: ["extended", "background"],
    prompt:
      "Rainy evening alley with nostalgic lights and subtle mist, cinematic but simple pixel-art composition for a composition screen background."
  },
  {
    id: "spr_player_idle",
    outputPath: "public/assets/characters/sprites/spr_player_idle_32x48.png",
    width: 32,
    height: 48,
    tags: ["mvp", "sprite"],
    prompt:
      "Single 32x48 sprite sheet frame of a retro pixel-art player character standing idle, front-right orientation, clear silhouette, transparent-like plain background."
  },
  {
    id: "spr_player_walk",
    outputPath: "public/assets/characters/sprites/spr_player_walk_64x48.png",
    width: 64,
    height: 48,
    tags: ["mvp", "sprite"],
    prompt:
      "Two-frame horizontal walking sprite strip, total canvas 64x48, retro pixel-art player character, consistent outfit with idle sprite, right-facing walk cycle."
  },
  {
    id: "spr_mob_01_idle",
    outputPath: "public/assets/characters/sprites/spr_mob_01_idle_32x48.png",
    width: 32,
    height: 48,
    tags: ["mvp", "sprite"],
    prompt:
      "Single 32x48 sprite sheet frame of a passerby NPC character standing idle, retro pixel-art, distinct silhouette, neutral clothes, transparent-like plain background."
  },
  {
    id: "spr_mob_01_walk",
    outputPath: "public/assets/characters/sprites/spr_mob_01_walk_64x48.png",
    width: 64,
    height: 48,
    tags: ["mvp", "sprite"],
    prompt:
      "Two-frame horizontal walking sprite strip, total canvas 64x48, retro pixel-art passerby NPC matching mob idle style, right-facing walk cycle."
  },
  {
    id: "ui_balloon_speech",
    outputPath: "public/assets/ui/ui_balloon_speech_280x80.png",
    width: 280,
    height: 80,
    tags: ["mvp", "ui"],
    prompt:
      "Pixel-art speech balloon panel for dialogue, warm paper tone with wooden border, empty inner area, no text."
  },
  {
    id: "ui_balloon_tail",
    outputPath: "public/assets/ui/ui_balloon_tail_24x24.png",
    width: 24,
    height: 24,
    tags: ["mvp", "ui"],
    prompt:
      "Pixel-art speech balloon tail piece matching the dialogue panel style, plain background."
  },
  {
    id: "character_master_all_cast",
    outputPath: "public/assets/master-sheets/character_master_all_cast_4096x4096.png",
    width: 4096,
    height: 4096,
    tags: ["minimum-master-sheet", "master-sheet"],
    imageSize: "4K",
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single ultra-packed character master sheet for the whole playable cast in nostalgic retro pixel art. Include player, one town passerby mob, and four customers (Anna, Ben, Cara, Dan) in one canvas. For each character include: waist-up portrait, orthographic turnaround set (front, side, back full body), idle frame, 2-frame right-facing walk strip, five expression portraits (neutral, smile, surprised, troubled, success), and two lip-sync mouth tiles (A, B). Keep strict identity consistency and shared palette language across all characters. One packed sheet only, no text."
  },
  {
    id: "ui_world_mega_sheet",
    outputPath: "public/assets/master-sheets/ui_world_mega_sheet_4096x4096.png",
    width: 4096,
    height: 4096,
    kind: "ui_sheet",
    tags: ["minimum-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/character_master_all_cast_4096x4096.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single ultra-packed world UI and slot-parts sprite sheet in nostalgic pixel art. Include town interaction markers, shop UI parts, composition panel frames, otoword base icons for tempo/key/instrument/rhythm/mood, many slot part icons near capacity for all five slots, and overlay badges (locked, recommended, selected, conflict, new, equipped, rare tiers). Include short FX strips (selection ring, click ripple, sparkle, warning pulse). Keep strict grid-friendly packing, clean separations for slicing, one canvas only, no text."
  },
  {
    id: "bg_world_mega_sheet",
    outputPath: "public/assets/master-sheets/bg_world_mega_sheet_4096x4096.png",
    width: 4096,
    height: 4096,
    tags: ["minimum-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/character_master_all_cast_4096x4096.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single mega background variation sheet in nostalgic pixel art that packs all major scenes. Include town street, shop interior, and compose area with consistent camera framing per scene, and provide time/weather variants (day, golden hour, rainy overcast, night) for each scene in one packed atlas-like layout. Keep every panel readable and slice-ready, no characters, no text, no logos."
  },
  {
    id: "character_master_player",
    outputPath: "public/assets/master-sheets/character_master_player_2048x2048.png",
    width: 2048,
    height: 2048,
    tags: ["mvp-master-sheet", "master-sheet"],
    prompt:
      "Single complete character master sheet for the player in nostalgic Japanese pixel art. Include one waist-up portrait, one orthographic turnaround set (front, side, back full body), one idle full-body frame, one 4-frame right-facing walking strip, five small expression portraits (neutral, smile, surprised, troubled, success), and two lip-sync mouth tiles (A, B). Keep one consistent character design, transparent or plain neutral background, no extra characters."
  },
  {
    id: "character_master_mob01",
    outputPath: "public/assets/master-sheets/character_master_mob01_2048x2048.png",
    width: 2048,
    height: 2048,
    tags: ["mvp-master-sheet", "master-sheet"],
    prompt:
      "Single complete character master sheet for a town passerby NPC in nostalgic pixel art. Include one waist-up portrait, one orthographic turnaround set (front, side, back full body), one idle frame, one 4-frame right-facing walking strip, five small expression portraits (neutral, smile, surprised, troubled, success), and two lip-sync mouth tiles. Keep style and palette aligned with the player sheet, plain neutral background, no extra characters."
  },
  {
    id: "character_master_customers",
    outputPath: "public/assets/master-sheets/character_master_customers_3072x3072.png",
    width: 3072,
    height: 3072,
    tags: ["mvp-master-sheet", "master-sheet"],
    imageSize: "4K",
    prompt:
      "Single packed character master sheet containing four customer characters (Anna, Ben, Cara, Dan) in nostalgic retro pixel art. For each customer include: one waist-up portrait, one orthographic turnaround set (front, side, back full body), one idle frame, one 2-frame right-facing walk strip, five expression portraits (neutral, smile, surprised, troubled, success), and two lip-sync mouth tiles (A, B). Keep strict identity consistency per customer and style consistency across all four customers. One canvas only, no extra characters, no text."
  },
  {
    id: "portrait_sheet_player",
    outputPath: "public/assets/master-sheets/portrait_sheet_player_1024x1024.png",
    width: 1024,
    height: 1024,
    tags: ["extended-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_player_2048x2048.png"],
    prompt:
      "Single portrait variation sheet for the main player character in retro dot-pixel style. Arrange five consistent face portraits in one sheet: neutral, smile, surprised, troubled, success. Clean spacing, same head scale across tiles, no text."
  },
  {
    id: "portrait_sheet_cust_anna",
    outputPath: "public/assets/master-sheets/portrait_sheet_cust_anna_1024x1024.png",
    width: 1024,
    height: 1024,
    tags: ["extended-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_customers_3072x3072.png"],
    prompt:
      "Single portrait variation sheet for customer Anna in nostalgic retro pixel art. Arrange five consistent face portraits in one sheet: neutral, smile, surprised, troubled, success. Cozy town atmosphere style, no text."
  },
  {
    id: "portrait_sheet_cust_ben",
    outputPath: "public/assets/master-sheets/portrait_sheet_cust_ben_1024x1024.png",
    width: 1024,
    height: 1024,
    tags: ["extended-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_customers_3072x3072.png"],
    prompt:
      "Single portrait variation sheet for customer Ben in nostalgic retro pixel art. Arrange five consistent face portraits in one sheet: neutral, smile, surprised, troubled, success. Keep clear silhouette and warm palette, no text."
  },
  {
    id: "portrait_sheet_cust_cara",
    outputPath: "public/assets/master-sheets/portrait_sheet_cust_cara_1024x1024.png",
    width: 1024,
    height: 1024,
    tags: ["extended-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_customers_3072x3072.png"],
    prompt:
      "Single portrait variation sheet for customer Cara in nostalgic retro pixel art. Arrange five consistent face portraits in one sheet: neutral, smile, surprised, troubled, success. Keep consistent lighting and palette, no text."
  },
  {
    id: "portrait_sheet_cust_dan",
    outputPath: "public/assets/master-sheets/portrait_sheet_cust_dan_1024x1024.png",
    width: 1024,
    height: 1024,
    tags: ["extended-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_customers_3072x3072.png"],
    prompt:
      "Single portrait variation sheet for customer Dan in nostalgic retro pixel art. Arrange five consistent face portraits in one sheet: neutral, smile, surprised, troubled, success. Keep same pixel style as other customer sheets, no text."
  },
  {
    id: "portrait_sheet_customers",
    outputPath: "public/assets/master-sheets/portrait_sheet_customers_2048x2048.png",
    width: 2048,
    height: 2048,
    tags: ["extended-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_customers_3072x3072.png"],
    prompt:
      "Single packed portrait sprite sheet containing four customer characters (Anna, Ben, Cara, Dan) in nostalgic retro pixel art. For each customer, include five consistent face portraits: neutral, smile, surprised, troubled, success. Arrange as a clean grid with clear group spacing and uniform tile size, no text."
  },
  {
    id: "town_interaction_sheet",
    outputPath: "public/assets/master-sheets/town_interaction_sheet_2048x1024.png",
    width: 2048,
    height: 1024,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_customers_3072x3072.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single town interaction UI sheet for a retro pixel game. Include interaction markers and clickable objects for: shop entrance, gallery sign, compose spot, and commission marker. Provide visual state variations for each object: default, hover, active, visited, disabled. Keep all elements separated in one packed sheet, no text."
  },
  {
    id: "shop_ui_sheet",
    outputPath: "public/assets/master-sheets/shop_ui_sheet_2048x2048.png",
    width: 2048,
    height: 2048,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/town_interaction_sheet_2048x1024.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single shop UI master sheet in nostalgic wood-and-paper pixel style. Include category tabs, item cards, purchase buttons, and price badges. Provide state variants: default, hover, pressed, disabled, owned, insufficient. Keep edges clean for slicing and reuse, no text."
  },
  {
    id: "otoword_icon_sheet",
    outputPath: "public/assets/master-sheets/otoword_icon_sheet_2048x2048.png",
    width: 2048,
    height: 2048,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/shop_ui_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single otoword icon master sheet in crisp retro pixel art. Include base icons for music slots: tempo, key, instrument, rhythm, mood. Also include overlay badges for locked, recommended, selected, conflict, new, equipped. Arrange in a clean grid for sprite extraction, no text."
  },
  {
    id: "otoword_tempo_parts_sheet",
    outputPath: "public/assets/master-sheets/otoword_tempo_parts_sheet_4096x4096.png",
    width: 4096,
    height: 4096,
    tags: ["max-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/otoword_icon_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single ultra-dense sprite sheet for tempo slot parts in nostalgic pixel art. Fill the sheet close to capacity with many unique tempo part icons and card mini-illustrations. Include broad ranges from very slow to very fast, pulse styles, groove intensity variants, and metronome motifs. Keep strict 1-sheet packing, even tile rhythm, clean separations for slicing, no text."
  },
  {
    id: "otoword_key_parts_sheet",
    outputPath: "public/assets/master-sheets/otoword_key_parts_sheet_4096x4096.png",
    width: 4096,
    height: 4096,
    tags: ["max-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/otoword_icon_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single ultra-dense sprite sheet for key slot parts in nostalgic pixel art. Pack as many unique key and scale themed icons as possible: major/minor color families, modal motifs, bright-dark tonal badges, root-note symbolic variants, harmonic tension markers. Keep one coherent style, one packed sheet only, no text."
  },
  {
    id: "otoword_instrument_parts_sheet",
    outputPath: "public/assets/master-sheets/otoword_instrument_parts_sheet_4096x4096.png",
    width: 4096,
    height: 4096,
    tags: ["max-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/otoword_icon_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single ultra-dense sprite sheet for instrument slot parts in nostalgic pixel art. Fill near capacity with diverse instrument icons and card mini-art: piano families, guitars, basses, strings, woodwinds, brass, synth leads, pads, percussion, unusual timbres. Ensure each icon is visually distinct and readable at small size. One packed sheet only, no text."
  },
  {
    id: "otoword_rhythm_parts_sheet",
    outputPath: "public/assets/master-sheets/otoword_rhythm_parts_sheet_4096x4096.png",
    width: 4096,
    height: 4096,
    tags: ["max-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/otoword_icon_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single ultra-dense sprite sheet for rhythm slot parts in nostalgic pixel art. Pack many rhythm pattern icons and card mini-art: steady, swing, shuffle, waltz, breakbeat, lo-fi, syncopated, marching, triplet-heavy, sparse grooves. Keep grid-like packing, clean outlines, and consistent pixel language. One sheet only, no text."
  },
  {
    id: "otoword_mood_parts_sheet",
    outputPath: "public/assets/master-sheets/otoword_mood_parts_sheet_4096x4096.png",
    width: 4096,
    height: 4096,
    tags: ["max-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/otoword_icon_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single ultra-dense sprite sheet for mood slot parts in nostalgic pixel art. Fill the sheet with many mood/atmosphere icons and mini-art variants: warm, melancholic, hopeful, quiet, playful, dreamy, rainy, nostalgic, mysterious, festive, calm, tense. Keep visual clarity at icon scale and one consistent art direction. One packed sheet only, no text."
  },
  {
    id: "otoword_slot_overlays_sheet",
    outputPath: "public/assets/master-sheets/otoword_slot_overlays_sheet_4096x2048.png",
    width: 4096,
    height: 2048,
    tags: ["max-master-sheet", "master-sheet"],
    imageSize: "4K",
    referenceImagePaths: ["public/assets/master-sheets/otoword_icon_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single dense overlay and badge sprite sheet for otoword slot UI in nostalgic pixel style. Include many state overlays and rarity marks: locked, recommended, selected, conflict, new, equipped, rare, epic, legendary, quest, weather-bonus, combo-ready, cooldown, discounted, favorite, pinned. Provide multiple decorative frame and glow variants. One packed sheet only, no text."
  },
  {
    id: "compose_ui_sheet",
    outputPath: "public/assets/master-sheets/compose_ui_sheet_2048x2048.png",
    width: 2048,
    height: 2048,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/otoword_icon_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single composition-screen UI master sheet in nostalgic pixel style. Include slot frames, evaluation panels, hint panels, and result badges. Provide state variants: empty, filled, recommended, mismatch, perfect. Keep corners and borders suitable for 9-slice usage, no text."
  },
  {
    id: "fx_sheet",
    outputPath: "public/assets/master-sheets/fx_sheet_1024x1024.png",
    width: 1024,
    height: 1024,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/compose_ui_sheet_2048x2048.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single effects sprite sheet for retro pixel UI feedback. Include short frame sequences for selection ring, click ripple, sparkle flash, and warning mark pulse. Keep each sequence in separate rows with transparent or plain background, no text."
  },
  {
    id: "bg_town_sheet",
    outputPath: "public/assets/master-sheets/bg_town_sheet_1920x1080.png",
    width: 1920,
    height: 1080,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/character_master_customers_3072x3072.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Pixel-art style game background, three-quarter top-down (isometric-ish) view, 16:9. A cobblestone main street runs perfectly horizontal from left to right across the middle of the scene (NOT diagonal). Upper side: cozy old-town building facades (timber-frame style), doors, windows, awnings, warm street lamps, small plants. Lower side: a hint of rooftops and hedges at the bottom edge. Keep the center area clear and open for the player character. Background only: no people, no characters, no animals, no tables, no stalls, no vehicles. No UI, no speech bubbles, no numbers, no readable text, no logos, no watermark. Create a single 2x2 variation sheet in one image: four panels with identical camera and layout, tiled with no gaps and no borders. Panels: (top-left) clear daytime soft warm light, (top-right) golden hour sunset with long shadows, (bottom-left) rainy overcast with wet cobblestone reflections and puddles, (bottom-right) night with warm lamp light and readable details in shadows. Pixel art, crisp edges, clean outlines, subtle dithering, cozy warm palette, highly detailed environment background."
  },
  {
    id: "bg_shop_sheet",
    outputPath: "public/assets/master-sheets/bg_shop_sheet_1920x1080.png",
    width: 1920,
    height: 1080,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: ["public/assets/master-sheets/bg_town_sheet_1920x1080.png"],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single background variation sheet for the music shop interior in nostalgic dot-pixel style. Include at least day, cloudy, rainy, and evening ambience variants with the same camera layout. Keep one packed sheet, no characters, no text."
  },
  {
    id: "bg_compose_sheet",
    outputPath: "public/assets/master-sheets/bg_compose_sheet_1920x1080.png",
    width: 1920,
    height: 1080,
    tags: ["mvp-master-sheet", "master-sheet"],
    referenceImagePaths: [
      "public/assets/master-sheets/bg_town_sheet_1920x1080.png",
      "public/assets/master-sheets/bg_shop_sheet_1920x1080.png"
    ],
    styleGroup: "soft_nostalgic_town_v1",
    prompt:
      "Single background variation sheet for the composition area in nostalgic pixel-art style. Include multiple weather/time variants (day, cloudy, rainy, evening) with the same framing, atmospheric lights, and room for UI overlay. No characters, no text."
  }
];

export const ASSET_SPECS = RAW_ASSET_SPECS.map(withV2Defaults);

export const PRESETS = {
  mvp: ["mvp"],
  mvp_master_sheets: ["minimum-master-sheet"],
  minimum_master_sheets: ["minimum-master-sheet"],
  legacy_mvp_master_sheets: ["mvp-master-sheet"],
  rich_master_sheets: ["mvp-master-sheet", "max-master-sheet"],
  slot_parts_max: ["max-master-sheet"],
  all_master_sheets: ["master-sheet"],
  extended: ["mvp", "extended"],
  all: []
};

export function validateAssetSpecUniqueness(specs = ASSET_SPECS) {
  const idSet = new Set();
  const outputPathSet = new Set();

  for (const spec of specs) {
    if (idSet.has(spec.id)) {
      throw new Error(`Duplicate asset id detected in specs: ${spec.id}`);
    }
    idSet.add(spec.id);

    if (outputPathSet.has(spec.outputPath)) {
      throw new Error(`Duplicate outputPath detected in specs: ${spec.outputPath}`);
    }
    outputPathSet.add(spec.outputPath);
  }
}
