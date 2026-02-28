#!/usr/bin/env node

import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import process from "node:process";
import sharp from "sharp";

const SOURCES = [
  {
    id: "anna",
    input: "public/assets/master-sheets/portrait_sheet_cust_anna_1024x1024.png",
    output: "public/assets/characters/portraits/por_cust_anna_160.png",
    crop: { left: 36, top: 430, width: 169, height: 164 }
  },
  {
    id: "ben",
    input: "public/assets/master-sheets/portrait_sheet_cust_ben_1024x1024.png",
    output: "public/assets/characters/portraits/por_cust_ben_160.png",
    crop: { left: 36, top: 36, width: 280, height: 280 }
  },
  {
    id: "cara",
    input: "public/assets/master-sheets/portrait_sheet_cust_cara_1024x1024.png",
    output: "public/assets/characters/portraits/por_cust_cara_160.png",
    crop: { left: 36, top: 36, width: 280, height: 280 }
  },
  {
    id: "dan",
    input: "public/assets/master-sheets/portrait_sheet_cust_dan_1024x1024.png",
    output: "public/assets/characters/portraits/por_cust_dan_160.png",
    crop: { left: 50, top: 380, width: 170, height: 170 }
  }
];
const OUTPUT_SIZE = 160;

async function main() {
  for (const item of SOURCES) {
    await mkdir(dirname(item.output), { recursive: true });
    await sharp(item.input)
      .extract(item.crop)
      .resize(OUTPUT_SIZE, OUTPUT_SIZE, {
        fit: "fill",
        kernel: sharp.kernel.nearest
      })
      .png({ compressionLevel: 9 })
      .toFile(item.output);
    console.log(`OK   ${item.id} -> ${item.output}`);
  }
}

main().catch((error) => {
  console.error(`Failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
