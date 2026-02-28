import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const apiKey = process.env.MISTRAL_API_KEY;
const baseUrl = process.env.MISTRAL_BASE_URL || "https://api.mistral.ai/v1";

const headers = () => ({
  "Content-Type": "application/json",
  Authorization: `Bearer ${apiKey}`
});

const assertApiKey = () => {
  if (!apiKey) {
    throw new Error("MISTRAL_API_KEY is required.");
  }
};

const call = async (path, init = {}) => {
  assertApiKey();
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: {
      ...headers(),
      ...(init.headers || {})
    }
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${JSON.stringify(payload)}`);
  }

  return payload;
};

const command = process.argv[2] || "help";

const pretty = (payload) => console.log(JSON.stringify(payload, null, 2));

const createJob = async () => {
  const model = process.env.MISTRAL_FT_MODEL_ID || "ministral-3b-latest";
  const trainingFiles = (process.env.MISTRAL_TRAINING_FILE_IDS || "")
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean);
  const validationFiles = (process.env.MISTRAL_VALIDATION_FILE_IDS || "")
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean);

  if (!trainingFiles.length) {
    throw new Error("MISTRAL_TRAINING_FILE_IDS is required for create.");
  }

  const autoStart = process.env.MISTRAL_FT_AUTO_START === "true";

  const payload = {
    model,
    training_files: trainingFiles,
    validation_files: validationFiles.length ? validationFiles : undefined,
    auto_start: autoStart,
    hyperparameters: {
      training_steps: Number(process.env.MISTRAL_FT_TRAINING_STEPS || 800),
      learning_rate: Number(process.env.MISTRAL_FT_LR || 0.0002)
    }
  };

  const result = await call("/fine_tuning/jobs", {
    method: "POST",
    body: JSON.stringify(payload)
  });

  pretty(result);
};

const listJobs = async () => {
  const result = await call("/fine_tuning/jobs", { method: "GET" });
  pretty(result);
};

const statusJob = async () => {
  const jobId = process.argv[3] || process.env.MISTRAL_FT_JOB_ID;
  if (!jobId) throw new Error("job id is required.");

  const result = await call(`/fine_tuning/jobs/${jobId}`, { method: "GET" });
  pretty(result);
};

const startJob = async () => {
  const jobId = process.argv[3] || process.env.MISTRAL_FT_JOB_ID;
  if (!jobId) throw new Error("job id is required.");

  const result = await call(`/fine_tuning/jobs/${jobId}/start`, { method: "POST" });
  pretty(result);
};

const cancelJob = async () => {
  const jobId = process.argv[3] || process.env.MISTRAL_FT_JOB_ID;
  if (!jobId) throw new Error("job id is required.");

  const result = await call(`/fine_tuning/jobs/${jobId}/cancel`, { method: "POST" });
  pretty(result);
};

const uploadFile = async () => {
  assertApiKey();
  const filePath = process.argv[3] || process.env.MISTRAL_FT_UPLOAD_FILE;
  if (!filePath) throw new Error("file path is required.");

  const absolutePath = resolve(filePath);
  const bytes = await readFile(absolutePath);
  const form = new FormData();
  form.append("purpose", "fine-tune");
  form.append("file", new Blob([bytes]), absolutePath.split("/").pop() || "upload.jsonl");

  const response = await fetch(`${baseUrl}/files`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`
    },
    body: form
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${JSON.stringify(payload)}`);
  }

  pretty(payload);
};

const help = () => {
  console.log(`Usage:\n  node scripts/ft/mistral_ft_jobs.mjs <command> [arg]\n\nCommands:\n  create\n  list\n  status <job_id>\n  start <job_id>\n  cancel <job_id>\n  upload <file_path>\n`);
};

const main = async () => {
  if (command === "create") return createJob();
  if (command === "list") return listJobs();
  if (command === "status") return statusJob();
  if (command === "start") return startJob();
  if (command === "cancel") return cancelJob();
  if (command === "upload") return uploadFile();
  return help();
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
