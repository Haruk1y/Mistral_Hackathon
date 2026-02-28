import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { loadEnvFiles } from "../utils/load-env.mjs";

const root = resolve(new URL("../../", import.meta.url).pathname);
loadEnvFiles(root);

const scriptPath = resolve(root, "scripts/wandb/weave_eval_runner.py");
const localFallbackPath = resolve(root, "scripts/eval/run_eval_local.mjs");
const uvCacheDir = process.env.UV_CACHE_DIR || resolve(root, ".uv-cache");

const args = ["run", scriptPath];
const run = spawnSync("uv", args, {
  cwd: root,
  env: {
    ...process.env,
    UV_CACHE_DIR: uvCacheDir
  },
  encoding: "utf8",
});

if (run.stdout) process.stdout.write(run.stdout);
if (run.stderr) process.stderr.write(run.stderr);

if (run.status !== 0) {
  const allowFallback = !["0", "false", "no"].includes((process.env.EVAL_ALLOW_LOCAL_FALLBACK || "true").toLowerCase());
  if (!allowFallback) {
    console.error(`uv ${args.join(" ")} failed with code ${run.status ?? 1}`);
    process.exit(run.status ?? 1);
  }

  console.warn("Python eval runner failed. Falling back to local Node evaluator.");
  const fallbackRun = spawnSync("node", [localFallbackPath], {
    cwd: root,
    env: process.env,
    encoding: "utf8"
  });
  if (fallbackRun.stdout) process.stdout.write(fallbackRun.stdout);
  if (fallbackRun.stderr) process.stderr.write(fallbackRun.stderr);
  if (fallbackRun.status !== 0) {
    console.error(`node ${localFallbackPath} failed with code ${fallbackRun.status ?? 1}`);
    process.exit(fallbackRun.status ?? 1);
  }
}
