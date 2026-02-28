import { describe, expect, it } from "vitest";
import {
  ASSET_SPECS,
  PRESETS,
  validateAssetSpecUniqueness
} from "@/scripts/assets/asset-specs.mjs";
import {
  buildGenerationPrompt,
  buildRewriteInputPrompt,
  evaluateMasterSheetQuality,
  validateMasterSheetManifestItem
} from "@/scripts/assets/gemini-master-sheet-qa.mjs";

describe("validateAssetSpecUniqueness", () => {
  it("throws on duplicate id/outputPath", () => {
    expect(() =>
      validateAssetSpecUniqueness([
        {
          id: "dup",
          outputPath: "public/assets/a.png",
          width: 1,
          height: 1,
          tags: ["x"],
          prompt: "a",
          kind: "ui_sheet",
          alphaRequired: true,
          qaProfile: "standard",
          rewriteEnabled: true
        },
        {
          id: "dup",
          outputPath: "public/assets/a.png",
          width: 2,
          height: 2,
          tags: ["x"],
          prompt: "b",
          kind: "ui_sheet",
          alphaRequired: true,
          qaProfile: "standard",
          rewriteEnabled: true
        }
      ] as any)
    ).toThrow();
  });
});

describe("master-sheet image defaults", () => {
  function selectByPreset(preset: keyof typeof PRESETS) {
    const tags = PRESETS[preset];
    if (!tags || tags.length === 0) return ASSET_SPECS;
    return ASSET_SPECS.filter((spec) => tags.some((tag) => spec.tags.includes(tag)));
  }

  it("assigns recommended imageSize by sheet kind", () => {
    const byId = new Map(ASSET_SPECS.map((spec) => [spec.id, spec]));
    expect(byId.get("shop_ui_sheet")?.imageSize).toBe("2K");
    expect(byId.get("character_master_player")?.imageSize).toBe("2K");
    expect(byId.get("bg_shop_sheet")?.imageSize).toBe("4K");
  });

  it("applies unified style group defaults", () => {
    const byId = new Map(ASSET_SPECS.map((spec) => [spec.id, spec]));
    expect(byId.get("character_master_all_cast")?.styleGroup).toBe("soft_nostalgic_town_v1");
    expect(byId.get("ui_world_mega_sheet")?.styleGroup).toBe("soft_nostalgic_town_v1");
    expect(byId.get("bg_world_mega_sheet")?.styleGroup).toBe("soft_nostalgic_town_v1");
  });

  it("infers master-sheet aspect ratio from target dimensions", () => {
    const byId = new Map(ASSET_SPECS.map((spec) => [spec.id, spec]));
    expect(byId.get("shop_ui_sheet")?.imageAspectRatio).toBe("1:1");
    expect(byId.get("town_interaction_sheet")?.imageAspectRatio).toBe("2:1");
    expect(byId.get("bg_compose_sheet")?.imageAspectRatio).toBe("16:9");
  });

  it("minimizes default master-sheet count via compact preset", () => {
    const compact = selectByPreset("mvp_master_sheets");
    const ids = new Set(compact.map((spec) => spec.id));

    expect(compact).toHaveLength(3);
    expect(ids.has("character_master_all_cast")).toBe(true);
    expect(ids.has("ui_world_mega_sheet")).toBe(true);
    expect(ids.has("bg_world_mega_sheet")).toBe(true);
  });

  it("keeps legacy preset available for previous 11-sheet workflow", () => {
    const legacy = selectByPreset("legacy_mvp_master_sheets");
    const ids = new Set(legacy.map((spec) => spec.id));

    expect(legacy).toHaveLength(11);
    expect(ids.has("character_master_customers")).toBe(true);
    expect(ids.has("portrait_sheet_customers")).toBe(false);
  });

  it("defines dense slot-part sheets for max asset generation", () => {
    const maxSheets = ASSET_SPECS.filter((spec) => spec.tags.includes("max-master-sheet"));
    const ids = new Set(maxSheets.map((spec) => spec.id));

    expect(maxSheets.length).toBeGreaterThanOrEqual(6);
    expect(ids.has("otoword_tempo_parts_sheet")).toBe(true);
    expect(ids.has("otoword_key_parts_sheet")).toBe(true);
    expect(ids.has("otoword_instrument_parts_sheet")).toBe(true);
    expect(ids.has("otoword_rhythm_parts_sheet")).toBe(true);
    expect(ids.has("otoword_mood_parts_sheet")).toBe(true);
    expect(ids.has("otoword_slot_overlays_sheet")).toBe(true);

    for (const spec of maxSheets) {
      expect(spec.imageSize).toBe("4K");
    }
  });
});

describe("prompt builders", () => {
  it("includes kind/size/forbidden tokens", () => {
    const spec = {
      id: "shop_ui_sheet",
      kind: "ui_sheet",
      width: 2048,
      height: 2048,
      referenceImagePaths: ["public/assets/master-sheets/character_master_player_2048x2048.png"],
      imageAspectRatio: "16:9",
      imageSize: "2K",
      styleGroup: "soft_nostalgic_town_v1",
      qaProfile: "standard",
      prompt: "shop assets"
    } as const;

    const rewritePrompt = buildRewriteInputPrompt(spec as any);
    const generationPrompt = buildGenerationPrompt({
      spec: spec as any,
      rewrittenPrompt: "rewritten",
      backgroundKey: "#00FF00"
    });

    expect(rewritePrompt).toContain("sheet_kind=ui_sheet");
    expect(rewritePrompt).toContain("target_size=2048x2048");
    expect(rewritePrompt).toContain("target_aspect_ratio=16:9");
    expect(rewritePrompt).toContain("target_image_size=2K");
    expect(rewritePrompt).toContain("style_group=soft_nostalgic_town_v1");
    expect(rewritePrompt).toContain("reference_image_count=1");
    expect(rewritePrompt).toContain("single packed sprite sheet canvas only");
    expect(rewritePrompt).toContain("no text, no logo, no watermark");

    expect(generationPrompt).toContain("Target canvas: 2048x2048");
    expect(generationPrompt).toContain("Preferred aspect ratio: 16:9");
    expect(generationPrompt).toContain("Preferred image size preset: 2K");
    expect(generationPrompt).toContain("Style group: soft_nostalgic_town_v1");
    expect(generationPrompt).toContain("Use the provided 1 reference image");
    expect(generationPrompt).toContain("one packed sprite-sheet canvas");
    expect(generationPrompt).toContain("single material");
    expect(generationPrompt).toContain("#00FF00");
  });
});

describe("evaluateMasterSheetQuality alpha gate", () => {
  const baseSpec = {
    id: "character_master_player",
    kind: "character_master",
    alphaRequired: true,
    qaProfile: "strict"
  } as const;

  it("fails when alpha ratio is below threshold", () => {
    const result = evaluateMasterSheetQuality({
      spec: baseSpec as any,
      vision: {
        textOrWatermarkDetected: false,
        singleSheetCompliance: 0.95,
        characterConsistency: 0.95,
        elementSeparation: null
      },
      local: {
        decodeOk: true,
        sizeMatch: true,
        transparentPixelRatio: 0.02
      }
    });

    expect(result.gates.alpha).toBe(false);
    expect(result.qualityPassed).toBe(false);
  });

  it("passes alpha gate when ratio meets threshold", () => {
    const result = evaluateMasterSheetQuality({
      spec: baseSpec as any,
      vision: {
        textOrWatermarkDetected: false,
        singleSheetCompliance: 0.95,
        characterConsistency: 0.95,
        elementSeparation: null
      },
      local: {
        decodeOk: true,
        sizeMatch: true,
        transparentPixelRatio: 0.12
      }
    });

    expect(result.gates.alpha).toBe(true);
    expect(result.qualityPassed).toBe(true);
  });
});

describe("validateMasterSheetManifestItem", () => {
  it("validates manifest item shape", () => {
    const valid = validateMasterSheetManifestItem({
      id: "character_master_player",
      file: "character_master_player_2048x2048.png",
      targetSize: "2048x2048",
      actualSize: "1024x1024",
      model: { text: "gemini-2.5-flash", image: "nano-banana-pro-preview" },
      retries: 1,
      qualityScore: 88,
      qualityPassed: true,
      alpha: { required: true, transparentPixelRatio: 0.23 },
      createdAt: new Date().toISOString()
    });

    expect(valid.ok).toBe(true);
    expect(valid.errors).toHaveLength(0);
  });
});
