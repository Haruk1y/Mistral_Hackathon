#!/usr/bin/env node

import { access, copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { basename, dirname, extname, join } from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import sharp from "sharp";
import {
  ASSET_SPECS,
  PRESETS,
  STYLE_SUFFIX,
  NEGATIVE_SUFFIX,
  validateAssetSpecUniqueness
} from "./asset-specs.mjs";
import {
  applyChromaKeyAndResize,
  buildGenerationPrompt,
  buildRewriteInputPrompt,
  buildVisionQaPrompt,
  evaluateMasterSheetQuality,
  parseVisionQaPayload,
  validateMasterSheetManifestItem,
  calculateTransparentPixelRatioFromBuffer
} from "./gemini-master-sheet-qa.mjs";

const RAW_DIR = "public/assets/master-sheets/_raw";
const REPORT_DIR = "public/assets/master-sheets/_reports";
const ARCHIVE_DIR = "public/assets/master-sheets/_archive";
const MANIFEST_PATH = "data/master-sheets.manifest.json";

function createRunId() {
  return new Date().toISOString().replace(/[-:.]/g, "").replace("T", "_");
}

function parseArgs(argv, env) {
  const options = {
    preset: "mvp_master_sheets",
    yes: false,
    dryRun: false,
    force: false,
    list: false,
    reportOnly: false,
    only: new Set(),
    maxRetries: Number.parseInt(env.ASSET_MAX_RETRIES || "2", 10),
    delayMs: 450,
    textModel: env.GEMINI_TEXT_MODEL || "gemini-2.5-flash",
    imageModel: env.GEMINI_IMAGE_MODEL || "nano-banana-pro-preview",
    imageSize: env.GEMINI_IMAGE_SIZE || "",
    imageAspectRatio: env.GEMINI_IMAGE_ASPECT_RATIO || "",
    requestTimeoutMs: Number.parseInt(env.GEMINI_REQUEST_TIMEOUT_MS || "120000", 10),
    baseUrl: env.GEMINI_API_BASE_URL || "https://generativelanguage.googleapis.com/v1beta",
    alphaBgKey: env.ASSET_ALPHA_BG_KEY || "#00FF00",
    alphaTolerance: Number.parseInt(env.ASSET_ALPHA_TOLERANCE || "36", 10),
    runId: createRunId()
  };

  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg === "--yes") {
      options.yes = true;
      continue;
    }
    if (arg === "--dry-run") {
      options.dryRun = true;
      continue;
    }
    if (arg === "--force") {
      options.force = true;
      continue;
    }
    if (arg === "--list") {
      options.list = true;
      continue;
    }
    if (arg === "--report-only") {
      options.reportOnly = true;
      continue;
    }
    if (arg.startsWith("--preset=")) {
      options.preset = arg.slice("--preset=".length);
      continue;
    }
    if (arg.startsWith("--only=")) {
      const ids = arg
        .slice("--only=".length)
        .split(",")
        .map((id) => id.trim())
        .filter(Boolean);
      for (const id of ids) options.only.add(id);
      continue;
    }
    if (arg.startsWith("--max-retries=")) {
      const parsed = Number.parseInt(arg.slice("--max-retries=".length), 10);
      if (!Number.isNaN(parsed) && parsed >= 0) options.maxRetries = parsed;
      continue;
    }
    if (arg.startsWith("--delay-ms=")) {
      const parsed = Number.parseInt(arg.slice("--delay-ms=".length), 10);
      if (!Number.isNaN(parsed) && parsed >= 0) options.delayMs = parsed;
      continue;
    }
    if (arg.startsWith("--image-model=")) {
      options.imageModel = arg.slice("--image-model=".length);
      continue;
    }
    if (arg.startsWith("--image-size=")) {
      options.imageSize = arg.slice("--image-size=".length);
      continue;
    }
    if (arg.startsWith("--image-aspect-ratio=")) {
      options.imageAspectRatio = arg.slice("--image-aspect-ratio=".length);
      continue;
    }
    if (arg.startsWith("--text-model=")) {
      options.textModel = arg.slice("--text-model=".length);
      continue;
    }
    if (arg.startsWith("--request-timeout-ms=")) {
      const parsed = Number.parseInt(arg.slice("--request-timeout-ms=".length), 10);
      if (!Number.isNaN(parsed) && parsed >= 1000) options.requestTimeoutMs = parsed;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  options.maxRetries = Math.max(0, options.maxRetries);
  options.alphaTolerance = Math.max(0, Math.min(255, options.alphaTolerance));
  options.requestTimeoutMs = Math.max(1000, options.requestTimeoutMs);
  return options;
}

function printHelp() {
  const presetList = Object.keys(PRESETS).join("|");
  console.log(`
Gemini Master Sheet Generator V2 (Direct Gemini API)

Usage:
  node scripts/assets/generate-gemini-master-sheets-v2.mjs [options]

Options:
  --preset=<${presetList}>     Select preset (default: mvp_master_sheets)
  --only=<id1,id2,...>         Generate specific IDs only
  --force                      Overwrite existing final files
  --yes                        Skip confirmation prompt
  --dry-run                    Print planned actions only
  --report-only                Run QA/report on existing finals only
  --max-retries=<n>            Max retries per asset (default: 2)
  --delay-ms=<n>               Delay between API calls (default: 450)
  --image-model=<id>           Override image model
  --image-size=<size>          Override Gemini image size (e.g. 1K, 2K, 4K)
  --image-aspect-ratio=<ratio> Override Gemini image aspect ratio (e.g. 16:9)
  --text-model=<id>            Override text model
  --request-timeout-ms=<n>     Timeout for each Gemini request (default: 120000)
  --list                       List selectable assets and exit
  -h, --help                   Show this help

Environment variables:
  GEMINI_API_KEY
  GEMINI_API_BASE_URL
  GEMINI_IMAGE_MODEL
  GEMINI_IMAGE_SIZE
  GEMINI_IMAGE_ASPECT_RATIO
  GEMINI_REQUEST_TIMEOUT_MS
  GEMINI_TEXT_MODEL
  ASSET_MAX_RETRIES
  ASSET_ALPHA_BG_KEY
  ASSET_ALPHA_TOLERANCE
`);
}

function resolveSpecs(options) {
  let selected = ASSET_SPECS;

  if (options.preset !== "all") {
    const requiredTags = PRESETS[options.preset];
    if (!requiredTags) throw new Error(`Unknown preset: ${options.preset}`);
    selected = selected.filter((spec) => requiredTags.some((tag) => spec.tags.includes(tag)));
  }

  if (options.only.size > 0) {
    selected = selected.filter((spec) => options.only.has(spec.id));
  }

  return selected;
}

async function fileExists(path) {
  try {
    await access(path, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function inferMimeType(path) {
  const lower = String(path || "").toLowerCase();
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".webp")) return "image/webp";
  return "application/octet-stream";
}

function sleep(ms) {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadEnvFile(path) {
  try {
    const content = await readFile(path, "utf8");
    const lines = content.split(/\r?\n/);
    for (const line of lines) {
      if (!line || line.trim().startsWith("#")) continue;
      const index = line.indexOf("=");
      if (index <= 0) continue;

      const key = line.slice(0, index).trim();
      let value = line.slice(index + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }

      if (process.env[key] === undefined) {
        process.env[key] = value;
      }
    }
  } catch {
    // ignore missing file
  }
}

async function confirmExecution(options, specs) {
  if (options.dryRun || options.yes) return true;

  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    throw new Error("Confirmation required. Re-run with --yes in non-interactive mode.");
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  try {
    const answer = await rl.question(
      `Run V2 for ${specs.length} asset(s) preset=${options.preset} force=${options.force} reportOnly=${options.reportOnly} maxRetries=${options.maxRetries}? [y/N] `
    );
    return ["y", "yes"].includes(answer.trim().toLowerCase());
  } finally {
    rl.close();
  }
}

async function callGemini({ apiKey, baseUrl, model, payload, requestTimeoutMs }) {
  const url = `${baseUrl}/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const timeoutMs = Math.max(1000, Number(requestTimeoutMs) || 120000);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(`Gemini request timeout after ${timeoutMs}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }

  const json = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = json?.error?.message || response.statusText || "Gemini request failed";
    throw new Error(`Gemini API error (${response.status}): ${message}`);
  }

  return json;
}

function extractFirstText(payload) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  for (const candidate of candidates) {
    const parts = candidate?.content?.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      if (typeof part?.text === "string" && part.text.trim()) {
        return part.text.trim();
      }
    }
  }
  return "";
}

function extractFirstImage(payload) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  for (const candidate of candidates) {
    const parts = candidate?.content?.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      const inlineData = part?.inlineData || part?.inline_data;
      if (!inlineData || typeof inlineData.data !== "string") continue;
      const mimeType = inlineData.mimeType || inlineData.mime_type || "application/octet-stream";
      if (!String(mimeType).startsWith("image/")) continue;
      return { mimeType, buffer: Buffer.from(inlineData.data, "base64") };
    }
  }

  return null;
}

function parseRewriteText(text, fallback) {
  if (!text) return fallback;

  const cleaned = text
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/```$/i, "")
    .trim();

  try {
    const json = JSON.parse(cleaned);
    if (typeof json.rewrittenPrompt === "string" && json.rewrittenPrompt.trim()) {
      return json.rewrittenPrompt.trim();
    }
  } catch {
    // non-json fallback
  }

  if (cleaned.length > 24) return cleaned;
  return fallback;
}

async function buildRewrittenPrompt({ apiKey, options, spec }) {
  if (!spec.rewriteEnabled) {
    return { rewrittenPrompt: spec.prompt, rewriteWarning: "rewrite disabled by spec" };
  }

  const payload = {
    contents: [
      {
        role: "user",
        parts: [{ text: buildRewriteInputPrompt(spec) }]
      }
    ],
    generationConfig: {
      responseMimeType: "application/json",
      temperature: 0.35
    }
  };

  try {
    const response = await callGemini({
      apiKey,
      baseUrl: options.baseUrl,
      model: options.textModel,
      payload,
      requestTimeoutMs: options.requestTimeoutMs
    });
    const rewriteText = extractFirstText(response);
    return {
      rewrittenPrompt: parseRewriteText(rewriteText, spec.prompt),
      rewriteWarning: ""
    };
  } catch (error) {
    return {
      rewrittenPrompt: spec.prompt,
      rewriteWarning: error instanceof Error ? error.message : String(error)
    };
  }
}

async function runVisionQa({ apiKey, options, spec, imageBuffer }) {
  const payload = {
    contents: [
      {
        role: "user",
        parts: [
          { text: buildVisionQaPrompt(spec) },
          {
            inlineData: {
              mimeType: "image/png",
              data: imageBuffer.toString("base64")
            }
          }
        ]
      }
    ],
    generationConfig: {
      responseMimeType: "application/json",
      temperature: 0.1
    }
  };

  try {
    const response = await callGemini({
      apiKey,
      baseUrl: options.baseUrl,
      model: options.textModel,
      payload,
      requestTimeoutMs: options.requestTimeoutMs
    });
    return parseVisionQaPayload(response);
  } catch (error) {
    return {
      textOrWatermarkDetected: false,
      singleSheetCompliance: 0.5,
      characterConsistency: null,
      elementSeparation: null,
      notes: `Vision QA fallback: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

function toSizeString({ width, height }) {
  return `${width}x${height}`;
}

async function readJson(path, fallback) {
  try {
    const raw = await readFile(path, "utf8");
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}

async function writeJson(path, value) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(value, null, 2) + "\n");
}

async function writeFinalWithBackup({ targetPath, nextBuffer, specId, runId }) {
  const exists = await fileExists(targetPath);
  let archivePath = null;

  if (exists) {
    const ext = extname(targetPath) || ".png";
    const base = basename(targetPath, ext);
    archivePath = join(ARCHIVE_DIR, specId, `${base}__${runId}${ext}`);
    await mkdir(dirname(archivePath), { recursive: true });
    await copyFile(targetPath, archivePath);
  }

  await mkdir(dirname(targetPath), { recursive: true });
  await writeFile(targetPath, nextBuffer);

  return archivePath;
}

function makeManifestItem({ spec, actualSize, retries, quality, options }) {
  const item = {
    id: spec.id,
    file: spec.outputPath.split("/").pop(),
    targetSize: `${spec.width}x${spec.height}`,
    actualSize,
    model: {
      text: options.textModel,
      image: options.imageModel
    },
    retries,
    qualityScore: quality.qualityScore,
    qualityPassed: quality.qualityPassed,
    alpha: {
      required: Boolean(spec.alphaRequired),
      transparentPixelRatio: quality.metrics.transparentPixelRatio
    },
    createdAt: new Date().toISOString()
  };

  const validation = validateMasterSheetManifestItem(item);
  if (!validation.ok) {
    throw new Error(`Manifest item invalid for ${spec.id}: ${validation.errors.join("; ")}`);
  }

  return item;
}

function summarizeFailedGates(gates) {
  return Object.entries(gates)
    .filter(([, ok]) => !ok)
    .map(([name]) => name)
    .join(", ");
}

function resolveImageConfig({ spec, options }) {
  const imageSize = String(options.imageSize || spec.imageSize || "").trim();
  const aspectRatio = String(options.imageAspectRatio || spec.imageAspectRatio || "").trim();
  const imageConfig = {};

  if (imageSize) imageConfig.imageSize = imageSize;
  if (aspectRatio) imageConfig.aspectRatio = aspectRatio;

  return Object.keys(imageConfig).length > 0 ? imageConfig : null;
}

async function runReportOnly({ spec, options, apiKey }) {
  const exists = await fileExists(spec.outputPath);
  if (!exists) {
    throw new Error("final file not found");
  }

  const finalBuffer = await readFile(spec.outputPath);
  const metadata = await sharp(finalBuffer).metadata();
  const transparentPixelRatio = await calculateTransparentPixelRatioFromBuffer(finalBuffer);

  const vision = await runVisionQa({ apiKey, options, spec, imageBuffer: finalBuffer });
  const quality = evaluateMasterSheetQuality({
    spec,
    vision,
    local: {
      decodeOk: true,
      sizeMatch: metadata.width === spec.width && metadata.height === spec.height,
      transparentPixelRatio
    }
  });

  const report = {
    id: spec.id,
    mode: "report-only",
    spec: {
      targetSize: `${spec.width}x${spec.height}`,
      kind: spec.kind,
      qaProfile: spec.qaProfile,
      alphaRequired: spec.alphaRequired
    },
    quality,
    vision,
    finalPath: spec.outputPath,
    checkedAt: new Date().toISOString()
  };

  await writeJson(join(REPORT_DIR, `${spec.id}.json`), report);

  return makeManifestItem({
    spec,
    actualSize: `${metadata.width || 0}x${metadata.height || 0}`,
    retries: 0,
    quality,
    options
  });
}

async function runGeneration({ spec, options, apiKey }) {
  const imageConfig = resolveImageConfig({ spec, options });
  const configuredReferencePaths = Array.isArray(spec.referenceImagePaths)
    ? spec.referenceImagePaths
    : [];
  const usedReferencePaths = [];
  const missingReferencePaths = [];
  const referenceParts = [];

  for (const path of configuredReferencePaths) {
    if (!path || typeof path !== "string") continue;
    if (!(await fileExists(path))) {
      missingReferencePaths.push(path);
      continue;
    }

    const buffer = await readFile(path);
    referenceParts.push({
      inlineData: {
        mimeType: inferMimeType(path),
        data: buffer.toString("base64")
      }
    });
    usedReferencePaths.push(path);
  }

  const report = {
    id: spec.id,
    startedAt: new Date().toISOString(),
    spec: {
      kind: spec.kind,
      qaProfile: spec.qaProfile,
      alphaRequired: spec.alphaRequired,
      targetSize: `${spec.width}x${spec.height}`
    },
    model: {
      text: options.textModel,
      image: options.imageModel
    },
    imageConfig,
    referenceImages: {
      configured: configuredReferencePaths,
      used: usedReferencePaths,
      missing: missingReferencePaths
    },
    attempts: []
  };

  const rewrite = await buildRewrittenPrompt({ apiKey, options, spec });
  report.promptOriginal = spec.prompt;
  report.promptRewritten = rewrite.rewrittenPrompt;
  if (rewrite.rewriteWarning) {
    report.rewriteWarning = rewrite.rewriteWarning;
  }

  let bestAttempt = null;

  for (let attempt = 0; attempt <= options.maxRetries; attempt += 1) {
    const attemptNo = attempt + 1;
    const attemptRecord = {
      attempt: attemptNo,
      status: "running",
      generatedAt: new Date().toISOString()
    };

    try {
      const generationPrompt = buildGenerationPrompt({
        spec,
        rewrittenPrompt: rewrite.rewrittenPrompt,
        backgroundKey: options.alphaBgKey
      });

      const payload = {
        contents: [
          {
            role: "user",
            parts: [
              {
                text: [
                  generationPrompt,
                  STYLE_SUFFIX,
                  NEGATIVE_SUFFIX
                ].join("\n")
              },
              ...referenceParts
            ]
          }
        ],
        generationConfig: {
          responseModalities: ["TEXT", "IMAGE"]
        }
      };
      if (imageConfig) {
        payload.generationConfig.imageConfig = imageConfig;
      }

      const imageResponse = await callGemini({
        apiKey,
        baseUrl: options.baseUrl,
        model: options.imageModel,
        payload,
        requestTimeoutMs: options.requestTimeoutMs
      });
      const imageResult = extractFirstImage(imageResponse);
      if (!imageResult) {
        throw new Error("No image returned by Gemini image model");
      }

      const normalizedRawBuffer = await sharp(imageResult.buffer)
        .png({ compressionLevel: 9 })
        .toBuffer();

      const rawFilePath = join(RAW_DIR, spec.id, `${options.runId}_attempt${attemptNo}.png`);
      await mkdir(dirname(rawFilePath), { recursive: true });
      await writeFile(rawFilePath, normalizedRawBuffer);

      const processed = await applyChromaKeyAndResize({
        inputBuffer: normalizedRawBuffer,
        targetWidth: spec.width,
        targetHeight: spec.height,
        bgKey: options.alphaBgKey,
        tolerance: options.alphaTolerance
      });

      const vision = await runVisionQa({
        apiKey,
        options,
        spec,
        imageBuffer: processed.finalBuffer
      });

      const local = {
        decodeOk: true,
        sizeMatch: processed.finalSize.width === spec.width && processed.finalSize.height === spec.height,
        transparentPixelRatio: processed.transparentPixelRatio
      };

      const quality = evaluateMasterSheetQuality({ spec, vision, local });

      attemptRecord.status = quality.qualityPassed ? "passed" : "failed_quality";
      attemptRecord.imageConfig = imageConfig;
      attemptRecord.referenceImagesUsed = usedReferencePaths;
      attemptRecord.rawPath = rawFilePath;
      attemptRecord.rawSize = toSizeString(processed.rawSize);
      attemptRecord.finalSize = toSizeString(processed.finalSize);
      attemptRecord.vision = vision;
      attemptRecord.local = local;
      attemptRecord.quality = quality;

      report.attempts.push(attemptRecord);

      if (!bestAttempt || quality.qualityScore > bestAttempt.quality.qualityScore) {
        bestAttempt = {
          finalBuffer: processed.finalBuffer,
          rawSize: toSizeString(processed.rawSize),
          quality,
          attemptNo
        };
      }

      if (quality.qualityPassed) {
        const archivedFinalPath = await writeFinalWithBackup({
          targetPath: spec.outputPath,
          nextBuffer: processed.finalBuffer,
          specId: spec.id,
          runId: options.runId
        });
        if (archivedFinalPath) {
          attemptRecord.archivedPreviousFinal = archivedFinalPath;
        }
        report.completedAt = new Date().toISOString();
        report.result = {
          qualityPassed: true,
          selectedAttempt: attemptNo
        };

        await writeJson(join(REPORT_DIR, `${spec.id}.json`), report);

        return makeManifestItem({
          spec,
          actualSize: toSizeString(processed.rawSize),
          retries: attempt,
          quality,
          options
        });
      }
    } catch (error) {
      attemptRecord.status = "failed_error";
      attemptRecord.error = error instanceof Error ? error.message : String(error);
      report.attempts.push(attemptRecord);
    }

    await sleep(options.delayMs);
  }

  if (!bestAttempt) {
    report.completedAt = new Date().toISOString();
    report.result = {
      qualityPassed: false,
      selectedAttempt: null,
      reason: "all attempts failed before producing final image"
    };
    await writeJson(join(REPORT_DIR, `${spec.id}.json`), report);
    throw new Error(`No usable output produced for ${spec.id}`);
  }

  const archivedFinalPath = await writeFinalWithBackup({
    targetPath: spec.outputPath,
    nextBuffer: bestAttempt.finalBuffer,
    specId: spec.id,
    runId: options.runId
  });
  if (archivedFinalPath) {
    report.archivedPreviousFinal = archivedFinalPath;
  }

  report.completedAt = new Date().toISOString();
  report.result = {
    qualityPassed: false,
    selectedAttempt: bestAttempt.attemptNo,
    reason: `quality gates not met: ${summarizeFailedGates(bestAttempt.quality.gates)}`
  };

  await writeJson(join(REPORT_DIR, `${spec.id}.json`), report);

  return makeManifestItem({
    spec,
    actualSize: bestAttempt.rawSize,
    retries: options.maxRetries,
    quality: bestAttempt.quality,
    options
  });
}

async function main() {
  await loadEnvFile(".env.local");
  await loadEnvFile(".env");

  validateAssetSpecUniqueness(ASSET_SPECS);

  const options = parseArgs(process.argv.slice(2), process.env);
  if (options.help) {
    printHelp();
    return;
  }

  const specs = resolveSpecs(options);
  if (specs.length === 0) {
    throw new Error("No assets selected. Check --preset / --only arguments.");
  }

  if (options.list) {
    console.log("Available specs:");
    for (const spec of specs) {
      console.log(`- ${spec.id} (${spec.width}x${spec.height}) -> ${spec.outputPath}`);
    }
    return;
  }

  if (options.dryRun) {
    console.log(
      `DRY RUN: ${specs.length} asset(s), preset=${options.preset}, reportOnly=${options.reportOnly}, maxRetries=${options.maxRetries}`
    );
    for (const spec of specs) {
      const imageConfig = resolveImageConfig({ spec, options });
      const referenceCount = Array.isArray(spec.referenceImagePaths)
        ? spec.referenceImagePaths.length
        : 0;
      console.log(
        `PLAN ${spec.id} kind=${spec.kind} qa=${spec.qaProfile} alpha=${spec.alphaRequired} refs=${referenceCount} imageConfig=${
          imageConfig ? JSON.stringify(imageConfig) : "default"
        } -> ${spec.outputPath}`
      );
    }
    return;
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY is required.");
  }

  const accepted = await confirmExecution(options, specs);
  if (!accepted) {
    console.log("Canceled by user.");
    return;
  }

  const existingManifest = await readJson(MANIFEST_PATH, {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    generator: "scripts/assets/generate-gemini-master-sheets-v2.mjs",
    preset: options.preset,
    outputDir: "public/assets/master-sheets",
    items: []
  });

  const itemById = new Map(
    Array.isArray(existingManifest.items)
      ? existingManifest.items.filter((item) => item && typeof item.id === "string").map((item) => [item.id, item])
      : []
  );

  console.log(
    `V2 generation started. assets=${specs.length} preset=${options.preset} reportOnly=${options.reportOnly} maxRetries=${options.maxRetries}`
  );

  let success = 0;
  let failed = 0;

  for (const spec of specs) {
    const finalExists = await fileExists(spec.outputPath);

    if (finalExists && !options.force && !options.reportOnly) {
      console.log(`SKIP ${spec.id} (final exists, use --force to regenerate)`);
      try {
        const item = await runReportOnly({ spec, options, apiKey });
        itemById.set(spec.id, item);
        success += 1;
      } catch (error) {
        console.log(`WARN ${spec.id} report-only on existing file failed: ${error.message}`);
      }
      continue;
    }

    try {
      console.log(`RUN  ${spec.id}`);
      const item = options.reportOnly
        ? await runReportOnly({ spec, options, apiKey })
        : await runGeneration({ spec, options, apiKey });
      itemById.set(spec.id, item);
      success += 1;
      console.log(`OK   ${spec.id} score=${item.qualityScore} passed=${item.qualityPassed}`);
    } catch (error) {
      failed += 1;
      const message = error instanceof Error ? error.message : String(error);
      console.log(`FAIL ${spec.id}: ${message}`);
      const reportPath = join(REPORT_DIR, `${spec.id}.json`);
      const hasDetailedReport = await fileExists(reportPath);
      if (!hasDetailedReport) {
        await writeJson(reportPath, {
          id: spec.id,
          failedAt: new Date().toISOString(),
          error: message
        });
      }
    }

    await sleep(options.delayMs);
  }

  const masterSpecOrder = ASSET_SPECS.filter((spec) => spec.tags.includes("master-sheet")).map((spec) => spec.id);
  const orderedItems = [];
  const seen = new Set();

  for (const id of masterSpecOrder) {
    if (itemById.has(id)) {
      orderedItems.push(itemById.get(id));
      seen.add(id);
    }
  }

  for (const [id, item] of itemById.entries()) {
    if (!seen.has(id)) orderedItems.push(item);
  }

  const nextManifest = {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    generator: "scripts/assets/generate-gemini-master-sheets-v2.mjs",
    preset: options.preset,
    outputDir: "public/assets/master-sheets",
    items: orderedItems
  };

  for (const item of nextManifest.items) {
    const validation = validateMasterSheetManifestItem(item);
    if (!validation.ok) {
      throw new Error(`Manifest validation failed for ${item?.id || "unknown"}: ${validation.errors.join("; ")}`);
    }
  }

  await writeJson(MANIFEST_PATH, nextManifest);

  console.log(`Completed. success=${success} failed=${failed} total=${specs.length}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
