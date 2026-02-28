#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import process from "node:process";
import sharp from "sharp";

function parseArgs(argv) {
  const options = {
    input: "public/assets/master-sheets/character_master_all_cast_4096x4096.png",
    outputDir: "public/assets/characters/sprites/street-crowd",
    bgKey: "#00FF00",
    tolerance: 36,
    targetWidth: 32,
    targetHeight: 48
  };

  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") {
      options.help = true;
      continue;
    }
    if (arg.startsWith("--input=")) {
      options.input = arg.slice("--input=".length);
      continue;
    }
    if (arg.startsWith("--output-dir=")) {
      options.outputDir = arg.slice("--output-dir=".length);
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
Extract street-crowd sprites from character_master_all_cast.

Usage:
  node scripts/assets/extract-street-crowd-from-master.mjs [options]

Options:
  --input=<path>         Source master sheet
  --output-dir=<path>    Output directory
  --bg-key=<hex>         Chroma-key background color (default: #00FF00)
  --tolerance=<0-255>    Chroma-key tolerance (default: 36)
  -h, --help             Show help
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

function isLikelyBody(component) {
  const area = component.width * component.height;
  const aspectRatio = component.width / component.height;

  return (
    component.count >= 120_000 &&
    component.width >= 280 &&
    component.width <= 430 &&
    component.height >= 580 &&
    component.height <= 760 &&
    area >= 180_000 &&
    aspectRatio >= 0.42 &&
    aspectRatio <= 0.70
  );
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
      const dr = r - key.r;
      const dg = g - key.g;
      const db = b - key.b;
      const distSq = dr * dr + dg * dg + db * db;

      if (g > r + 8 && g > b + 8 && distSq <= spillSq * 2) {
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

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const key = parseHexColor(options.bgKey);
  const sourceBuffer = await sharp(options.input).ensureAlpha().png({ compressionLevel: 9 }).toBuffer();
  const { data, info } = await sharp(sourceBuffer).ensureAlpha().raw().toBuffer({
    resolveWithObject: true
  });

  const width = info.width;
  const height = info.height;
  const visited = new Uint8Array(width * height);
  const maxQueueSize = Math.min(width * height, 2_000_000);
  const queueX = new Int32Array(maxQueueSize);
  const queueY = new Int32Array(maxQueueSize);

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
    return !(dr <= options.tolerance && dg <= options.tolerance && db <= options.tolerance);
  };

  const components = [];

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
          if (tail >= maxQueueSize) continue;
          queueX[tail] = nx;
          queueY[tail] = ny;
          tail += 1;
        }
      }

      components.push({
        minX,
        maxX,
        minY,
        maxY,
        count,
        width: maxX - minX + 1,
        height: maxY - minY + 1
      });
    }
  }

  const bodies = components
    .filter(isLikelyBody)
    .sort((a, b) => (a.minY === b.minY ? a.minX - b.minX : a.minY - b.minY));

  await mkdir(options.outputDir, { recursive: true });

  const manifest = {
    generatedAt: new Date().toISOString(),
    source: options.input,
    count: bodies.length,
    targetSize: `${options.targetWidth}x${options.targetHeight}`,
    items: []
  };

  console.log(`Detected body-like components: ${bodies.length}`);

  for (let i = 0; i < bodies.length; i += 1) {
    const body = bodies[i];
    const padding = 24;
    const left = Math.max(0, body.minX - padding);
    const top = Math.max(0, body.minY - padding);
    const right = Math.min(width - 1, body.maxX + padding);
    const bottom = Math.min(height - 1, body.maxY + padding);
    const cropWidth = right - left + 1;
    const cropHeight = bottom - top + 1;

    const fileName = `crowd_${String(i + 1).padStart(2, "0")}_${options.targetWidth}x${options.targetHeight}.png`;
    const filePath = join(options.outputDir, fileName);

    const tileBuffer = await sharp(sourceBuffer)
      .extract({ left, top, width: cropWidth, height: cropHeight })
      .png({ compressionLevel: 9 })
      .toBuffer();

    const keyedTileBuffer = await removeGreenBackgroundFromBuffer(tileBuffer, {
      key,
      tolerance: options.tolerance
    });

    const resizedBuffer = await sharp(keyedTileBuffer)
      .resize(options.targetWidth, options.targetHeight, {
        fit: "contain",
        background: { r: 0, g: 0, b: 0, alpha: 0 },
        kernel: sharp.kernel.nearest
      })
      .png({ compressionLevel: 9 })
      .toBuffer();

    const cleanedBuffer = await neutralizeGreenEdgeFringe(resizedBuffer);
    await sharp(cleanedBuffer).png({ compressionLevel: 9 }).toFile(filePath);

    manifest.items.push({
      id: `crowd_${String(i + 1).padStart(2, "0")}`,
      file: fileName,
      sourceBbox: {
        left: body.minX,
        top: body.minY,
        width: body.width,
        height: body.height
      }
    });

    console.log(`OK   ${fileName}`);
  }

  const manifestPath = join(options.outputDir, "manifest.json");
  await mkdir(dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
  console.log(`Done. manifest=${manifestPath}`);
}

main().catch((error) => {
  console.error(`Failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
