#!/usr/bin/env node

import { mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import process from "node:process";
import sharp from "sharp";

const ROWS = 3;
const COLS = 4;
const SCENES = ["street", "shop", "compose"];
const TIMES = ["day", "sunset", "rain", "night"];

function parseArgs(argv) {
  const options = {
    input: "public/assets/master-sheets/bg_3x4.png",
    outputDir: "public/assets/bg",
    sliceDir: "public/assets/bg/slices/bg_3x4",
    targetWidth: 960,
    targetHeight: 540
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
    if (arg.startsWith("--slice-dir=")) {
      options.sliceDir = arg.slice("--slice-dir=".length);
      continue;
    }
    if (arg.startsWith("--target-width=")) {
      const value = Number.parseInt(arg.slice("--target-width=".length), 10);
      if (!Number.isNaN(value) && value > 0) options.targetWidth = value;
      continue;
    }
    if (arg.startsWith("--target-height=")) {
      const value = Number.parseInt(arg.slice("--target-height=".length), 10);
      if (!Number.isNaN(value) && value > 0) options.targetHeight = value;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  console.log(`
Split 3x4 background sheet into scene/time files.

Usage:
  node scripts/assets/split-bg-3x4.mjs [options]

Options:
  --input=<path>           Source sheet (default: public/assets/master-sheets/bg_3x4.png)
  --output-dir=<path>      Output dir for resized files (default: public/assets/bg)
  --slice-dir=<path>       Output dir for raw slices (default: public/assets/bg/slices/bg_3x4)
  --target-width=<n>       Resized width (default: 960)
  --target-height=<n>      Resized height (default: 540)
  -h, --help               Show help
`);
}

async function ensureDirs(paths) {
  for (const path of paths) {
    await mkdir(path, { recursive: true });
  }
}

async function writePng(buffer, outputPath) {
  await mkdir(dirname(outputPath), { recursive: true });
  await sharp(buffer).png({ compressionLevel: 9 }).toFile(outputPath);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const sourceBuffer = await sharp(options.input).png({ compressionLevel: 9 }).toBuffer();
  const metadata = await sharp(sourceBuffer).metadata();
  if (!metadata.width || !metadata.height) {
    throw new Error(`Failed to read image metadata: ${options.input}`);
  }
  if (metadata.width % COLS !== 0 || metadata.height % ROWS !== 0) {
    throw new Error(
      `Source size ${metadata.width}x${metadata.height} is not divisible by ${COLS}x${ROWS}`
    );
  }

  const tileWidth = Math.floor(metadata.width / COLS);
  const tileHeight = Math.floor(metadata.height / ROWS);

  if (SCENES.length !== ROWS || TIMES.length !== COLS) {
    throw new Error("Scene/time mapping does not match sheet grid");
  }

  await ensureDirs([options.outputDir, options.sliceDir]);

  console.log(
    `Split started: source=${options.input} tile=${tileWidth}x${tileHeight} target=${options.targetWidth}x${options.targetHeight}`
  );

  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const scene = SCENES[row];
      const time = TIMES[col];
      const left = col * tileWidth;
      const top = row * tileHeight;

      const tileBuffer = await sharp(sourceBuffer)
        .extract({ left, top, width: tileWidth, height: tileHeight })
        .png({ compressionLevel: 9 })
        .toBuffer();

      const rawName = `bg_${scene}_${time}_${tileWidth}x${tileHeight}.png`;
      const resizedName = `bg_${scene}_${time}_${options.targetWidth}x${options.targetHeight}.png`;
      const rawPath = join(options.sliceDir, rawName);
      const resizedPath = join(options.outputDir, resizedName);

      await writePng(tileBuffer, rawPath);
      await sharp(tileBuffer)
        .resize(options.targetWidth, options.targetHeight, {
          fit: "cover",
          position: "center",
          kernel: sharp.kernel.nearest
        })
        .png({ compressionLevel: 9 })
        .toFile(resizedPath);

      console.log(`OK   ${scene}/${time} -> ${resizedPath}`);
    }
  }

  console.log("Done.");
}

main().catch((error) => {
  console.error(`Failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
