import sharp from "sharp";

function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

export function parseHexColor(hex) {
  const normalized = String(hex || "").trim().replace(/^#/, "");
  if (normalized.length === 3) {
    const r = Number.parseInt(normalized[0] + normalized[0], 16);
    const g = Number.parseInt(normalized[1] + normalized[1], 16);
    const b = Number.parseInt(normalized[2] + normalized[2], 16);
    if ([r, g, b].some((v) => Number.isNaN(v))) {
      throw new Error(`Invalid hex color: ${hex}`);
    }
    return { r, g, b };
  }

  if (normalized.length !== 6) {
    throw new Error(`Invalid hex color: ${hex}`);
  }

  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);

  if ([r, g, b].some((v) => Number.isNaN(v))) {
    throw new Error(`Invalid hex color: ${hex}`);
  }

  return { r, g, b };
}

export function buildRewriteInputPrompt(spec) {
  const referenceImageCount = Array.isArray(spec.referenceImagePaths)
    ? spec.referenceImagePaths.length
    : 0;
  const tokens = [
    "Output JSON only.",
    `asset_id=${spec.id}`,
    `sheet_kind=${spec.kind}`,
    `target_size=${spec.width}x${spec.height}`,
    `target_aspect_ratio=${spec.imageAspectRatio || "auto"}`,
    `target_image_size=${spec.imageSize || "auto"}`,
    `style_group=${spec.styleGroup || "default"}`,
    `reference_image_count=${referenceImageCount}`,
    `qa_profile=${spec.qaProfile}`,
    "no text, no logo, no watermark, no signature",
    "single material per sheet",
    "single packed sprite sheet canvas only"
  ];

  return [
    "You are a senior pixel-art direction writer for game production.",
    "Rewrite the base intent into a production-ready image prompt.",
    "Return a strict JSON object with keys: rewrittenPrompt, shotNotes.",
    `Base intent: ${spec.prompt}`,
    `Production tokens: ${tokens.join(" | ")}`
  ].join("\n");
}

export function buildGenerationPrompt({ spec, rewrittenPrompt, backgroundKey }) {
  const key = backgroundKey.toUpperCase();
  const sizeLine = `Target canvas: ${spec.width}x${spec.height}`;
  const styleGroupLine = `Style group: ${spec.styleGroup || "default"}`;
  const referenceImageCount = Array.isArray(spec.referenceImagePaths)
    ? spec.referenceImagePaths.length
    : 0;
  const aspectLine = spec.imageAspectRatio
    ? `Preferred aspect ratio: ${spec.imageAspectRatio}`
    : null;
  const imageSizeLine = spec.imageSize ? `Preferred image size preset: ${spec.imageSize}` : null;
  const referenceLine =
    referenceImageCount > 0
      ? `Use the provided ${referenceImageCount} reference image(s) for identity and palette consistency.`
      : null;

  return [
    "Create exactly one game production master sheet image.",
    "Output must be one packed sprite-sheet canvas for this asset id.",
    "Include all requested variants in this single sheet only.",
    "Do not produce multiple images or split outputs.",
    "Do not include any text, letters, numbers, logos, signatures, or watermarks.",
    "One material only (single material). No mixed scenes.",
    sizeLine,
    aspectLine,
    imageSizeLine,
    styleGroupLine,
    referenceLine,
    `Sheet kind: ${spec.kind}`,
    "Use crisp retro pixel-art rendering and clean silhouette separation.",
    `Background must be a flat chroma key color ${key} wherever transparency is expected.`,
    "No gradients in key background.",
    rewrittenPrompt
  ]
    .filter(Boolean)
    .join("\n");
}

export function buildVisionQaPrompt(spec) {
  return [
    "You are a strict game-asset QA assistant.",
    "Inspect the provided image and return JSON only.",
    "Required keys:",
    "text_or_watermark_detected (boolean)",
    "single_sheet_compliance (number 0..1)",
    "character_consistency (number 0..1 or null)",
    "element_separation (number 0..1 or null)",
    "notes (short string)",
    `Asset kind: ${spec.kind}`,
    "Use conservative judgments. If unsure, lower the score."
  ].join("\n");
}

function extractJsonCandidate(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return null;

  const cleaned = trimmed
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/```$/i, "")
    .trim();

  try {
    return JSON.parse(cleaned);
  } catch {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(cleaned.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}

export function parseVisionQaPayload(payload) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  const parts = candidates.flatMap((candidate) => candidate?.content?.parts || []);

  for (const part of parts) {
    if (typeof part?.text === "string") {
      const json = extractJsonCandidate(part.text);
      if (json) {
        return {
          textOrWatermarkDetected: Boolean(json.text_or_watermark_detected),
          singleSheetCompliance: clamp01(json.single_sheet_compliance),
          characterConsistency:
            typeof json.character_consistency === "number"
              ? clamp01(json.character_consistency)
              : null,
          elementSeparation:
            typeof json.element_separation === "number"
              ? clamp01(json.element_separation)
              : null,
          notes: typeof json.notes === "string" ? json.notes : ""
        };
      }
    }
  }

  return {
    textOrWatermarkDetected: false,
    singleSheetCompliance: 0.5,
    characterConsistency: null,
    elementSeparation: null,
    notes: "Vision QA parse fallback"
  };
}

export async function calculateTransparentPixelRatioFromBuffer(buffer) {
  const { data } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const totalPixels = data.length / 4;
  if (totalPixels === 0) return 0;

  let transparentPixels = 0;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] === 0) transparentPixels += 1;
  }

  return transparentPixels / totalPixels;
}

export async function applyChromaKeyAndResize({
  inputBuffer,
  targetWidth,
  targetHeight,
  bgKey,
  tolerance
}) {
  const key = parseHexColor(bgKey);

  const { data, info } = await sharp(inputBuffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const rgba = Buffer.from(data);
  const threshold = Math.max(0, Math.min(255, Number.parseInt(String(tolerance), 10) || 36));

  for (let i = 0; i < rgba.length; i += 4) {
    const dr = Math.abs(rgba[i] - key.r);
    const dg = Math.abs(rgba[i + 1] - key.g);
    const db = Math.abs(rgba[i + 2] - key.b);
    if (dr <= threshold && dg <= threshold && db <= threshold) {
      rgba[i + 3] = 0;
    }
  }

  const keyedBuffer = await sharp(rgba, {
    raw: {
      width: info.width,
      height: info.height,
      channels: 4
    }
  })
    .png({ compressionLevel: 9 })
    .toBuffer();

  const finalBuffer = await sharp(keyedBuffer)
    .resize(targetWidth, targetHeight, {
      fit: "fill",
      kernel: sharp.kernel.nearest
    })
    .png({ compressionLevel: 9 })
    .toBuffer();

  const transparentPixelRatio = await calculateTransparentPixelRatioFromBuffer(finalBuffer);
  const finalMeta = await sharp(finalBuffer).metadata();

  return {
    finalBuffer,
    rawSize: {
      width: info.width,
      height: info.height
    },
    finalSize: {
      width: finalMeta.width || targetWidth,
      height: finalMeta.height || targetHeight
    },
    transparentPixelRatio
  };
}

function isCharacterKind(kind) {
  return kind === "character_master" || kind === "portrait_sheet";
}

function isUiKind(kind) {
  return kind === "ui_sheet" || kind === "fx_sheet";
}

export function evaluateMasterSheetQuality({ spec, vision, local }) {
  const textOrWatermarkDetected = Boolean(vision?.textOrWatermarkDetected);
  const singleSheetCompliance = clamp01(vision?.singleSheetCompliance);
  const characterConsistency =
    typeof vision?.characterConsistency === "number" ? clamp01(vision.characterConsistency) : 0;
  const elementSeparation =
    typeof vision?.elementSeparation === "number" ? clamp01(vision.elementSeparation) : 0;

  const decodeOk = Boolean(local?.decodeOk);
  const sizeMatch = Boolean(local?.sizeMatch);
  const transparentPixelRatio = clamp01(local?.transparentPixelRatio);

  const gates = {
    decodeOk,
    sizeMatch,
    noTextOrWatermark: !textOrWatermarkDetected,
    singleSheetCompliance: singleSheetCompliance >= 0.7,
    characterConsistency:
      !isCharacterKind(spec.kind) || characterConsistency >= 0.7,
    elementSeparation: !isUiKind(spec.kind) || elementSeparation >= 0.65,
    alpha:
      !spec.alphaRequired || spec.qaProfile === "light" || transparentPixelRatio >= 0.08
  };

  let score = 0;
  score += decodeOk ? 10 : 0;
  score += sizeMatch ? 15 : 0;
  score += Math.round(singleSheetCompliance * 25);

  if (isCharacterKind(spec.kind)) {
    score += Math.round(characterConsistency * 25);
  } else if (isUiKind(spec.kind)) {
    score += Math.round(elementSeparation * 20);
  } else {
    score += 10;
  }

  if (!spec.alphaRequired || spec.qaProfile === "light") {
    score += 10;
  } else {
    score += Math.round(Math.min(1, transparentPixelRatio / 0.08) * 15);
  }

  if (textOrWatermarkDetected) {
    score -= 35;
  }

  const qualityScore = Math.max(0, Math.min(100, score));
  const qualityPassed =
    qualityScore >= 75 &&
    gates.decodeOk &&
    gates.sizeMatch &&
    gates.noTextOrWatermark &&
    gates.singleSheetCompliance &&
    gates.characterConsistency &&
    gates.elementSeparation &&
    gates.alpha;

  return {
    qualityScore,
    qualityPassed,
    gates,
    metrics: {
      textOrWatermarkDetected,
      singleSheetCompliance,
      characterConsistency: isCharacterKind(spec.kind) ? characterConsistency : null,
      elementSeparation: isUiKind(spec.kind) ? elementSeparation : null,
      transparentPixelRatio
    }
  };
}

export function validateMasterSheetManifestItem(item) {
  const errors = [];

  if (!item || typeof item !== "object") {
    return { ok: false, errors: ["item must be an object"] };
  }

  const requiredStringFields = ["id", "file", "targetSize", "actualSize", "createdAt"];
  for (const key of requiredStringFields) {
    if (typeof item[key] !== "string" || item[key].trim() === "") {
      errors.push(`${key} must be a non-empty string`);
    }
  }

  if (!item.model || typeof item.model !== "object") {
    errors.push("model must be an object");
  } else {
    if (typeof item.model.text !== "string" || !item.model.text) {
      errors.push("model.text must be a non-empty string");
    }
    if (typeof item.model.image !== "string" || !item.model.image) {
      errors.push("model.image must be a non-empty string");
    }
  }

  if (!Number.isInteger(item.retries) || item.retries < 0) {
    errors.push("retries must be a non-negative integer");
  }

  if (typeof item.qualityScore !== "number" || item.qualityScore < 0 || item.qualityScore > 100) {
    errors.push("qualityScore must be a number between 0 and 100");
  }

  if (typeof item.qualityPassed !== "boolean") {
    errors.push("qualityPassed must be a boolean");
  }

  if (!item.alpha || typeof item.alpha !== "object") {
    errors.push("alpha must be an object");
  } else {
    if (typeof item.alpha.required !== "boolean") {
      errors.push("alpha.required must be a boolean");
    }
    if (
      typeof item.alpha.transparentPixelRatio !== "number" ||
      item.alpha.transparentPixelRatio < 0 ||
      item.alpha.transparentPixelRatio > 1
    ) {
      errors.push("alpha.transparentPixelRatio must be a number between 0 and 1");
    }
  }

  return { ok: errors.length === 0, errors };
}
