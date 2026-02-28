import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const ENV_FILES = [".env.local", ".env"];

const stripQuotes = (value) => {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  return value;
};

const parseLine = (line) => {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) return null;

  const eq = trimmed.indexOf("=");
  if (eq < 1) return null;

  const key = trimmed.slice(0, eq).trim();
  const rawValue = trimmed.slice(eq + 1).trim();
  return { key, value: stripQuotes(rawValue) };
};

export const loadEnvFiles = (rootDir) => {
  for (const relPath of ENV_FILES) {
    const fullPath = resolve(rootDir, relPath);
    if (!existsSync(fullPath)) continue;

    const content = readFileSync(fullPath, "utf8");
    const lines = content.split(/\r?\n/);
    for (const line of lines) {
      const pair = parseLine(line);
      if (!pair) continue;
      if (process.env[pair.key] == null || process.env[pair.key] === "") {
        process.env[pair.key] = pair.value;
      }
    }
  }
};
