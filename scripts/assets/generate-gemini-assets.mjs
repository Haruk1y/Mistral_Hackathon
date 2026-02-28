#!/usr/bin/env node

import { mkdir, writeFile, access, readFile } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { dirname } from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import {
  ASSET_SPECS,
  PRESETS,
  STYLE_SUFFIX,
  NEGATIVE_SUFFIX,
  validateAssetSpecUniqueness
} from "./asset-specs.mjs";

function parseArgs(argv, env) {
  const options = {
    preset: "mvp",
    model: env.GEMINI_IMAGE_MODEL || "nano-banana-pro-preview",
    baseUrl: env.GEMINI_API_BASE_URL || "https://generativelanguage.googleapis.com/v1beta",
    dryRun: false,
    force: false,
    list: false,
    yes: false,
    only: new Set(),
    delayMs: 400
  };

  for (const arg of argv) {
    if (arg === "--help" || arg === "-h") {
      options.help = true;
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
    if (arg === "--yes") {
      options.yes = true;
      continue;
    }
    if (arg.startsWith("--preset=")) {
      options.preset = arg.slice("--preset=".length);
      continue;
    }
    if (arg.startsWith("--model=")) {
      options.model = arg.slice("--model=".length);
      continue;
    }
    if (arg.startsWith("--base-url=")) {
      options.baseUrl = arg.slice("--base-url=".length);
      continue;
    }
    if (arg.startsWith("--delay-ms=")) {
      const parsed = Number(arg.slice("--delay-ms=".length));
      if (!Number.isNaN(parsed) && parsed >= 0) {
        options.delayMs = parsed;
      }
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
    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function printHelp() {
  const presetList = Object.keys(PRESETS).join("|");
  console.log(`
Gemini Asset Generator for おとことば工房

Usage:
  node scripts/assets/generate-gemini-assets.mjs [options]

Options:
  --preset=<${presetList}>   Select asset preset (default: mvp)
  --only=<id1,id2,...>          Generate only specific asset IDs
  --model=<model-id>            Gemini model ID
  --base-url=<url>              Gemini API base URL
  --delay-ms=<number>           Delay between requests in ms (default: 400)
  --force                       Overwrite existing files
  --yes                         Skip confirmation prompt
  --dry-run                     Print planned actions only
  --list                        List selectable assets and exit
  -h, --help                    Show this help

Environment variables:
  GEMINI_API_KEY                Required for real generation
  GEMINI_IMAGE_MODEL            Optional default model
  GEMINI_API_BASE_URL           Optional API base URL
`);
}

function resolveSpecs(options) {
  let selected = ASSET_SPECS;

  if (options.preset !== "all") {
    const requiredTags = PRESETS[options.preset];
    if (!requiredTags) {
      throw new Error(`Unknown preset: ${options.preset}`);
    }
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

function buildPrompt(spec) {
  return [
    `Create one game asset image for a retro Japanese pixel-art game.`,
    `Target output size: ${spec.width}x${spec.height}.`,
    `Asset role: ${spec.id}.`,
    spec.prompt,
    STYLE_SUFFIX,
    NEGATIVE_SUFFIX
  ].join(" ");
}

async function callGeminiImage({ apiKey, baseUrl, model, prompt }) {
  const url = `${baseUrl}/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const payload = {
    contents: [
      {
        role: "user",
        parts: [{ text: prompt }]
      }
    ],
    generationConfig: {
      responseModalities: ["TEXT", "IMAGE"]
    }
  };

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const json = await response.json();
  if (!response.ok) {
    const message = json?.error?.message || response.statusText;
    throw new Error(`Gemini API error (${response.status}): ${message}`);
  }

  const candidates = Array.isArray(json.candidates) ? json.candidates : [];
  for (const candidate of candidates) {
    const parts = candidate?.content?.parts;
    if (!Array.isArray(parts)) continue;
    for (const part of parts) {
      const inlineData = part.inlineData || part.inline_data;
      if (!inlineData) continue;
      const mimeType = inlineData.mimeType || inlineData.mime_type;
      const data = inlineData.data;
      if (!data || typeof data !== "string") continue;
      if (!mimeType || !String(mimeType).startsWith("image/")) continue;
      return { mimeType, buffer: Buffer.from(data, "base64"), raw: json };
    }
  }

  const textHint = candidates
    .flatMap((candidate) => candidate?.content?.parts || [])
    .map((part) => part?.text)
    .filter(Boolean)
    .join("\n");

  throw new Error(`No image data returned by Gemini. ${textHint ? `Response text: ${textHint}` : ""}`);
}

async function sleep(ms) {
  if (ms <= 0) return;
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function confirmExecution(options, specs) {
  if (options.dryRun || options.yes) return true;

  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    throw new Error("Confirmation required. Re-run with --yes to execute in non-interactive mode.");
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  try {
    const answer = await rl.question(
      `Generate ${specs.length} asset(s) with model=${options.model}, preset=${options.preset}, force=${options.force}? [y/N] `
    );
    const accepted = ["y", "yes"].includes(answer.trim().toLowerCase());
    if (!accepted) {
      console.log("Canceled by user.");
      return false;
    }
    return true;
  } finally {
    rl.close();
  }
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
      if (!key) continue;
      if (process.env[key] === undefined) {
        process.env[key] = value;
      }
    }
  } catch {
    // ignore when file is absent
  }
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
    console.log("Available asset specs:");
    for (const spec of ASSET_SPECS) {
      console.log(
        `- ${spec.id} (${spec.width}x${spec.height}) -> ${spec.outputPath} [${spec.tags.join(", ")}]`
      );
    }
    return;
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey && !options.dryRun) {
    throw new Error("GEMINI_API_KEY is required. Set it in your shell or .env.local.");
  }

  const allowed = await confirmExecution(options, specs);
  if (!allowed) {
    return;
  }

  console.log(
    `Generating ${specs.length} asset(s) with model=${options.model}, preset=${options.preset}, dryRun=${options.dryRun}`
  );

  let generated = 0;
  let skipped = 0;
  for (const spec of specs) {
    const exists = await fileExists(spec.outputPath);
    if (exists && !options.force) {
      console.log(`SKIP ${spec.id} (exists): ${spec.outputPath}`);
      skipped += 1;
      continue;
    }

    const prompt = buildPrompt(spec);
    if (options.dryRun) {
      console.log(`PLAN ${spec.id} -> ${spec.outputPath}`);
      console.log(`      prompt: ${prompt}`);
      continue;
    }

    console.log(`RUN  ${spec.id} -> ${spec.outputPath}`);
    const { buffer, mimeType } = await callGeminiImage({
      apiKey,
      baseUrl: options.baseUrl,
      model: options.model,
      prompt
    });

    await mkdir(dirname(spec.outputPath), { recursive: true });
    await writeFile(spec.outputPath, buffer);
    console.log(`OK   ${spec.id} (${buffer.length} bytes, ${mimeType})`);
    generated += 1;
    await sleep(options.delayMs);
  }

  console.log(`Completed. generated=${generated}, skipped=${skipped}, total=${specs.length}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
