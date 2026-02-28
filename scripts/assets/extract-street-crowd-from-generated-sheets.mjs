#!/usr/bin/env node

import { mkdir, readdir, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import process from "node:process";
import sharp from "sharp";

const DEFAULT_INPUT_DIR = "public/assets/characters/sprites/street-crowd";
const DEFAULT_OUTPUT_SPRITES = "public/assets/characters/sprites/street-crowd-v2";
const DEFAULT_OUTPUT_PORTRAITS = "public/assets/characters/portraits/street-crowd-v2";
const DEFAULT_MANIFEST_PATH = `${DEFAULT_OUTPUT_SPRITES}/manifest.generated.json`;

const SHEET_PATTERNS = [/^Generated Image .*\.png$/i];
const SHEET_GRID = { cols: 2, rows: 2 };
const OUTPUT_SPRITE_SIZE = { width: 64, height: 96 };
const OUTPUT_FACE_SIZE = { width: 160, height: 160 };

const POSE_KEYS = ["front", "diag_left", "diag_right"];

function parseArgs(argv) {
  const options = {
    inputDir: DEFAULT_INPUT_DIR,
    inputFiles: [],
    outputSpritesDir: DEFAULT_OUTPUT_SPRITES,
    outputPortraitsDir: DEFAULT_OUTPUT_PORTRAITS,
    manifestPath: DEFAULT_MANIFEST_PATH,
    bgKey: "#00FF00",
    tolerance: 40
  };

  for (const arg of argv) {
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg.startsWith("--input-dir=")) {
      options.inputDir = arg.slice("--input-dir=".length);
      continue;
    }
    if (arg.startsWith("--inputs=")) {
      options.inputFiles = arg
        .slice("--inputs=".length)
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean);
      continue;
    }
    if (arg.startsWith("--output-sprites-dir=")) {
      options.outputSpritesDir = arg.slice("--output-sprites-dir=".length);
      continue;
    }
    if (arg.startsWith("--output-portraits-dir=")) {
      options.outputPortraitsDir = arg.slice("--output-portraits-dir=".length);
      continue;
    }
    if (arg.startsWith("--manifest=")) {
      options.manifestPath = arg.slice("--manifest=".length);
      continue;
    }
    if (arg.startsWith("--bg-key=")) {
      options.bgKey = arg.slice("--bg-key=".length);
      continue;
    }
    if (arg.startsWith("--tolerance=")) {
      const value = Number.parseInt(arg.slice("--tolerance=".length), 10);
      if (!Number.isNaN(value) && value >= 0 && value <= 255) {
        options.tolerance = value;
      }
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  console.log(`
Extract crowd sprites/portraits from generated green-screen sheets.

Usage:
  node scripts/assets/extract-street-crowd-from-generated-sheets.mjs [options]

Options:
  --input-dir=<path>             Directory containing generated sheets
  --inputs=<file1,file2,...>     Explicit list of input files
  --output-sprites-dir=<path>    Output directory for 64x96 poses
  --output-portraits-dir=<path>  Output directory for 160x160 faces
  --manifest=<path>              Output manifest JSON path
  --bg-key=<hex>                 Chroma key color (default: #00FF00)
  --tolerance=<0-255>            Chroma-key tolerance (default: 40)
  -h, --help                     Show help
`);
}

function parseHexColor(hex) {
  const normalized = String(hex || "").trim().replace(/^#/, "");
  if (normalized.length !== 6) {
    throw new Error(`Invalid hex color: ${hex}`);
  }
  const r = Number.parseInt(normalized.slice(0, 2), 16);
  const g = Number.parseInt(normalized.slice(2, 4), 16);
  const b = Number.parseInt(normalized.slice(4, 6), 16);
  if ([r, g, b].some((value) => Number.isNaN(value))) {
    throw new Error(`Invalid hex color: ${hex}`);
  }
  return { r, g, b };
}

function isMatchingSheetFile(file) {
  return SHEET_PATTERNS.some((pattern) => pattern.test(file));
}

async function resolveInputFiles(options) {
  if (options.inputFiles.length > 0) {
    return options.inputFiles.map((file) => (file.includes("/") ? file : join(options.inputDir, file)));
  }

  const files = await readdir(options.inputDir);
  return files
    .filter(isMatchingSheetFile)
    .sort((a, b) => a.localeCompare(b))
    .map((file) => join(options.inputDir, file));
}

function classifyComponent(component) {
  const { width, height, count } = component;
  const area = width * height;
  const fillRatio = area > 0 ? count / area : 0;

  if (
    count >= 12_000 &&
    width >= 90 &&
    width <= 220 &&
    height >= 180 &&
    height <= 340 &&
    fillRatio >= 0.45
  ) {
    return "body";
  }

  if (
    count >= 7_000 &&
    width >= 80 &&
    width <= 180 &&
    height >= 100 &&
    height <= 220 &&
    fillRatio >= 0.45
  ) {
    return "face";
  }

  return "other";
}

async function detectConnectedComponents(inputBuffer, { key, tolerance }) {
  const { data, info } = await sharp(inputBuffer).ensureAlpha().raw().toBuffer({
    resolveWithObject: true
  });

  const width = info.width;
  const height = info.height;
  const visited = new Uint8Array(width * height);
  const queueX = new Int32Array(width * height);
  const queueY = new Int32Array(width * height);
  const components = [];

  const pixelIndex = (x, y) => y * width + x;

  const isForeground = (x, y) => {
    const i = pixelIndex(x, y) * 4;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    if (a < 8) return false;
    const dr = Math.abs(r - key.r);
    const dg = Math.abs(g - key.g);
    const db = Math.abs(b - key.b);
    return !(dr <= tolerance && dg <= tolerance && db <= tolerance);
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = pixelIndex(x, y);
      if (visited[p]) continue;
      visited[p] = 1;
      if (!isForeground(x, y)) continue;

      let head = 0;
      let tail = 0;
      queueX[tail] = x;
      queueY[tail] = y;
      tail += 1;

      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let count = 0;

      while (head < tail) {
        const cx = queueX[head];
        const cy = queueY[head];
        head += 1;
        count += 1;

        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        const neighbors = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1]
        ];

        for (const [nx, ny] of neighbors) {
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const np = pixelIndex(nx, ny);
          if (visited[np]) continue;
          visited[np] = 1;
          if (!isForeground(nx, ny)) continue;
          queueX[tail] = nx;
          queueY[tail] = ny;
          tail += 1;
        }
      }

      const component = {
        minX,
        maxX,
        minY,
        maxY,
        width: maxX - minX + 1,
        height: maxY - minY + 1,
        count
      };
      component.centerX = (component.minX + component.maxX) / 2;
      component.centerY = (component.minY + component.maxY) / 2;
      component.kind = classifyComponent(component);
      components.push(component);
    }
  }

  return { width, height, components };
}

async function removeGreenBackgroundFromBuffer(inputBuffer, { key, tolerance }) {
  const { data, info } = await sharp(inputBuffer).ensureAlpha().raw().toBuffer({
    resolveWithObject: true
  });
  const width = info.width;
  const height = info.height;
  const rgba = Buffer.from(data);
  const pixelCount = width * height;
  const bgMask = new Uint8Array(pixelCount);
  const queue = new Int32Array(pixelCount);

  const toleranceSq = tolerance * tolerance * 3;
  const spillTolerance = Math.min(255, tolerance + 72);
  const spillSq = spillTolerance * spillTolerance * 3;

  const pixelIndex = (x, y) => y * width + x;

  const isBgCandidate = (p) => {
    const i = p * 4;
    const r = rgba[i];
    const g = rgba[i + 1];
    const b = rgba[i + 2];
    const a = rgba[i + 3];
    if (a < 8) return true;

    const dr = r - key.r;
    const dg = g - key.g;
    const db = b - key.b;
    const distSq = dr * dr + dg * dg + db * db;
    if (distSq <= toleranceSq) return true;

    const greenDominant = g >= r + 18 && g >= b + 18;
    return greenDominant && distSq <= spillSq;
  };

  let head = 0;
  let tail = 0;

  const enqueue = (x, y) => {
    const p = pixelIndex(x, y);
    if (bgMask[p]) return;
    if (!isBgCandidate(p)) return;
    bgMask[p] = 1;
    queue[tail] = p;
    tail += 1;
  };

  for (let x = 0; x < width; x += 1) {
    enqueue(x, 0);
    enqueue(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    enqueue(0, y);
    enqueue(width - 1, y);
  }

  while (head < tail) {
    const p = queue[head];
    head += 1;
    const x = p % width;
    const y = Math.floor(p / width);

    const neighbors = [
      [x - 1, y],
      [x + 1, y],
      [x, y - 1],
      [x, y + 1]
    ];

    for (const [nx, ny] of neighbors) {
      if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
      const np = pixelIndex(nx, ny);
      if (bgMask[np]) continue;
      if (!isBgCandidate(np)) continue;
      bgMask[np] = 1;
      queue[tail] = np;
      tail += 1;
    }
  }

  for (let p = 0; p < pixelCount; p += 1) {
    if (!bgMask[p]) continue;
    const i = p * 4;
    rgba[i] = 0;
    rgba[i + 1] = 0;
    rgba[i + 2] = 0;
    rgba[i + 3] = 0;
  }

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const p = pixelIndex(x, y);
      if (bgMask[p]) continue;
      const i = p * 4;
      if (rgba[i + 3] < 8) continue;

      const nearBg =
        bgMask[pixelIndex(x - 1, y)] ||
        bgMask[pixelIndex(x + 1, y)] ||
        bgMask[pixelIndex(x, y - 1)] ||
        bgMask[pixelIndex(x, y + 1)];
      if (!nearBg) continue;

      const r = rgba[i];
      const g = rgba[i + 1];
      const b = rgba[i + 2];
      if (g > r + 8 && g > b + 8) {
        const cap = Math.max(r, b) + 4;
        if (rgba[i + 1] > cap) {
          rgba[i + 1] = cap;
        }
      }
    }
  }

  return sharp(rgba, {
    raw: {
      width,
      height,
      channels: 4
    }
  })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function neutralizeGreenEdgeFringe(inputBuffer) {
  const { data, info } = await sharp(inputBuffer).ensureAlpha().raw().toBuffer({
    resolveWithObject: true
  });
  const width = info.width;
  const height = info.height;
  const rgba = Buffer.from(data);
  const pixelIndex = (x, y) => (y * width + x) * 4;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = pixelIndex(x, y);
      const a = rgba[i + 3];
      if (a < 8) continue;

      const hasTransparentNeighbor =
        rgba[pixelIndex(x - 1, y) + 3] < 8 ||
        rgba[pixelIndex(x + 1, y) + 3] < 8 ||
        rgba[pixelIndex(x, y - 1) + 3] < 8 ||
        rgba[pixelIndex(x, y + 1) + 3] < 8;
      if (!hasTransparentNeighbor) continue;

      const r = rgba[i];
      const g = rgba[i + 1];
      const b = rgba[i + 2];
      if (g > r + 10 && g > b + 10) {
        const cap = Math.max(r, b) + 4;
        if (rgba[i + 1] > cap) {
          rgba[i + 1] = cap;
        }
      }
    }
  }

  return sharp(rgba, {
    raw: {
      width,
      height,
      channels: 4
    }
  })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function purgeResidualGreenKeyPixels(inputBuffer, { key, tolerance }) {
  const { data, info } = await sharp(inputBuffer).ensureAlpha().raw().toBuffer({
    resolveWithObject: true
  });
  const width = info.width;
  const height = info.height;
  const rgba = Buffer.from(data);
  const killTolerance = Math.min(255, Math.max(40, tolerance + 18));
  const killSq = killTolerance * killTolerance * 3;

  for (let i = 0; i < rgba.length; i += 4) {
    const a = rgba[i + 3];
    if (a < 8) continue;
    const r = rgba[i];
    const g = rgba[i + 1];
    const b = rgba[i + 2];
    if (!(g > r + 12 && g > b + 12)) continue;

    const dr = r - key.r;
    const dg = g - key.g;
    const db = b - key.b;
    const distSq = dr * dr + dg * dg + db * db;

    if (distSq <= killSq) {
      rgba[i] = 0;
      rgba[i + 1] = 0;
      rgba[i + 2] = 0;
      rgba[i + 3] = 0;
    }
  }

  return sharp(rgba, {
    raw: {
      width,
      height,
      channels: 4
    }
  })
    .png({ compressionLevel: 9 })
    .toBuffer();
}

async function exportComponent({
  sourceBuffer,
  component,
  sourceWidth,
  sourceHeight,
  targetWidth,
  targetHeight,
  outputPath,
  chromaOptions,
  padding,
  fit = "contain"
}) {
  const left = Math.max(0, Math.floor(component.minX - padding));
  const top = Math.max(0, Math.floor(component.minY - padding));
  const right = Math.min(sourceWidth - 1, Math.ceil(component.maxX + padding));
  const bottom = Math.min(sourceHeight - 1, Math.ceil(component.maxY + padding));
  const cropWidth = right - left + 1;
  const cropHeight = bottom - top + 1;

  const tileBuffer = await sharp(sourceBuffer)
    .extract({ left, top, width: cropWidth, height: cropHeight })
    .png({ compressionLevel: 9 })
    .toBuffer();

  const keyed = await removeGreenBackgroundFromBuffer(tileBuffer, chromaOptions);
  const resized = await sharp(keyed)
    .resize(targetWidth, targetHeight, {
      fit,
      background: { r: 0, g: 0, b: 0, alpha: 0 },
      kernel: sharp.kernel.nearest
    })
    .png({ compressionLevel: 9 })
    .toBuffer();

  const cleaned = await neutralizeGreenEdgeFringe(resized);
  const purged = await purgeResidualGreenKeyPixels(cleaned, chromaOptions);
  await sharp(purged).png({ compressionLevel: 9 }).toFile(outputPath);

  return {
    outputPath,
    sourceBbox: {
      left: component.minX,
      top: component.minY,
      width: component.width,
      height: component.height
    }
  };
}

function groupByCharacterCell(bodyComponents, faceComponents, sourceWidth, sourceHeight) {
  const cellWidth = sourceWidth / SHEET_GRID.cols;
  const cellHeight = sourceHeight / SHEET_GRID.rows;
  const groups = [];

  for (let row = 0; row < SHEET_GRID.rows; row += 1) {
    for (let col = 0; col < SHEET_GRID.cols; col += 1) {
      groups.push({
        row,
        col,
        key: `${row}-${col}`,
        bodies: [],
        faces: []
      });
    }
  }

  const cellIndex = (row, col) => row * SHEET_GRID.cols + col;

  for (const component of bodyComponents) {
    const row = Math.min(SHEET_GRID.rows - 1, Math.max(0, Math.floor(component.centerY / cellHeight)));
    const col = Math.min(SHEET_GRID.cols - 1, Math.max(0, Math.floor(component.centerX / cellWidth)));
    groups[cellIndex(row, col)].bodies.push(component);
  }

  for (const component of faceComponents) {
    const row = Math.min(SHEET_GRID.rows - 1, Math.max(0, Math.floor(component.centerY / cellHeight)));
    const col = Math.min(SHEET_GRID.cols - 1, Math.max(0, Math.floor(component.centerX / cellWidth)));
    groups[cellIndex(row, col)].faces.push(component);
  }

  for (const group of groups) {
    group.bodies.sort((a, b) => a.centerX - b.centerX);
    group.faces.sort((a, b) => a.centerY - b.centerY);
  }

  return groups;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const key = parseHexColor(options.bgKey);
  const inputFiles = await resolveInputFiles(options);
  if (inputFiles.length === 0) {
    throw new Error(`No input files found in ${options.inputDir}`);
  }

  await mkdir(options.outputSpritesDir, { recursive: true });
  await mkdir(options.outputPortraitsDir, { recursive: true });

  const manifest = {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    inputFiles,
    outputSpritesDir: options.outputSpritesDir,
    outputPortraitsDir: options.outputPortraitsDir,
    bgKey: options.bgKey,
    tolerance: options.tolerance,
    spriteTarget: `${OUTPUT_SPRITE_SIZE.width}x${OUTPUT_SPRITE_SIZE.height}`,
    portraitTarget: `${OUTPUT_FACE_SIZE.width}x${OUTPUT_FACE_SIZE.height}`,
    items: []
  };

  let castIndex = 1;

  for (const inputPath of inputFiles) {
    const sourceBuffer = await sharp(inputPath).ensureAlpha().png({ compressionLevel: 9 }).toBuffer();
    const { width, height, components } = await detectConnectedComponents(sourceBuffer, {
      key,
      tolerance: options.tolerance
    });

    const bodies = components.filter((component) => component.kind === "body");
    const faces = components.filter((component) => component.kind === "face");
    const groups = groupByCharacterCell(bodies, faces, width, height);

    console.log(
      `[sheet] ${basename(inputPath)} bodies=${bodies.length} faces=${faces.length} cells=${groups.length}`
    );

    for (const group of groups) {
      if (group.bodies.length < 3 || group.faces.length < 1) {
        console.warn(
          `[warn] Skip cell ${group.key} from ${basename(inputPath)} (bodies=${group.bodies.length}, faces=${group.faces.length})`
        );
        continue;
      }

      const castId = `crowd_${String(castIndex).padStart(2, "0")}`;
      const spriteOutputs = {};
      const poseComponents = group.bodies.slice(0, 3);

      for (let poseIndex = 0; poseIndex < poseComponents.length; poseIndex += 1) {
        const pose = POSE_KEYS[poseIndex];
        const fileName = `${castId}_${pose}_${OUTPUT_SPRITE_SIZE.width}x${OUTPUT_SPRITE_SIZE.height}.png`;
        const outputPath = join(options.outputSpritesDir, fileName);
        spriteOutputs[pose] = await exportComponent({
          sourceBuffer,
          component: poseComponents[poseIndex],
          sourceWidth: width,
          sourceHeight: height,
          targetWidth: OUTPUT_SPRITE_SIZE.width,
          targetHeight: OUTPUT_SPRITE_SIZE.height,
        outputPath,
        chromaOptions: { key, tolerance: options.tolerance },
        padding: 12,
        fit: "contain"
      });
      }

      const faceMainFile = `por_${castId}_face_${OUTPUT_FACE_SIZE.width}.png`;
      const faceMainPath = join(options.outputPortraitsDir, faceMainFile);
      const faceMain = await exportComponent({
        sourceBuffer,
        component: group.faces[0],
        sourceWidth: width,
        sourceHeight: height,
        targetWidth: OUTPUT_FACE_SIZE.width,
        targetHeight: OUTPUT_FACE_SIZE.height,
        outputPath: faceMainPath,
        chromaOptions: { key, tolerance: options.tolerance },
        padding: 2,
        fit: "cover"
      });

      let faceAlt = null;
      if (group.faces[1]) {
        const faceAltFile = `por_${castId}_face_alt_${OUTPUT_FACE_SIZE.width}.png`;
        const faceAltPath = join(options.outputPortraitsDir, faceAltFile);
        faceAlt = await exportComponent({
          sourceBuffer,
          component: group.faces[1],
          sourceWidth: width,
          sourceHeight: height,
          targetWidth: OUTPUT_FACE_SIZE.width,
          targetHeight: OUTPUT_FACE_SIZE.height,
          outputPath: faceAltPath,
          chromaOptions: { key, tolerance: options.tolerance },
          padding: 2,
          fit: "cover"
        });
      }

      manifest.items.push({
        castId,
        index: castIndex,
        source: {
          file: inputPath,
          cell: {
            row: group.row,
            col: group.col
          }
        },
        outputs: {
          sprites: {
            front: spriteOutputs.front.outputPath,
            diagLeft: spriteOutputs.diag_left.outputPath,
            diagRight: spriteOutputs.diag_right.outputPath
          },
          portraits: {
            face: faceMain.outputPath,
            faceAlt: faceAlt ? faceAlt.outputPath : null
          }
        },
        sourceBboxes: {
          front: spriteOutputs.front.sourceBbox,
          diagLeft: spriteOutputs.diag_left.sourceBbox,
          diagRight: spriteOutputs.diag_right.sourceBbox,
          face: faceMain.sourceBbox,
          faceAlt: faceAlt ? faceAlt.sourceBbox : null
        }
      });

      console.log(`OK   ${castId} (${basename(inputPath)} cell ${group.key})`);
      castIndex += 1;
    }
  }

  manifest.totalCasts = manifest.items.length;
  manifest.totalSpriteFiles = manifest.items.length * 3;
  manifest.totalPortraitFiles = manifest.items.length * 2;

  await mkdir(join(options.outputSpritesDir), { recursive: true });
  await writeFile(options.manifestPath, JSON.stringify(manifest, null, 2) + "\n");

  console.log(`Done. casts=${manifest.totalCasts}`);
  console.log(`Manifest: ${options.manifestPath}`);
}

main().catch((error) => {
  console.error(`Failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
