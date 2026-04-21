const fs = require("node:fs");
const path = require("node:path");

require("dotenv").config({ quiet: true });

const { GoogleGenAI } = require("@google/genai");

const { createVideoEmbeddingsStore } = require("./lib/postgres-video-store");

const DEFAULTS = {
  downloadsDir: "downloads",
  embeddingModel: "gemini-embedding-2-preview",
  geminiApiKey: "",
  fixedSku: "",
  outputDimensionality: 3072,
  upsertBatchSize: 20,
  queryChunkSize: 50,
  concurrency: 10,
  embedRetryLimit: 5,
  embedRetryBaseDelayMs: 2000,
  skipExistingEmbeddings: true,
  audioTrackExtraction: false,
  videoFps: null,
  videoStartOffsetSeconds: null,
  videoEndOffsetSeconds: null,
  maxVideos: null,
  embeddingPrompt:
    "task: clustering | query: Embed this video for grouping duplicate or near-duplicate ad creatives. Focus on the video's visual and semantic identity rather than filenames, URLs, or other metadata.",
};

function readEnvNumber(name, fallback) {
  const value = process.env[name];
  if (value === undefined || value === "") {
    return fallback;
  }

  const parsed = Number(value);
  if (Number.isNaN(parsed)) {
    throw new Error(`Environment variable ${name} must be a number, got: ${value}`);
  }

  return parsed;
}

function readEnvInteger(name, fallback) {
  const value = readEnvNumber(name, fallback);
  if (value === null) {
    return value;
  }
  return Math.trunc(value);
}

function readEnvBoolean(name, fallback) {
  const value = process.env[name];
  if (value === undefined || value === "") {
    return fallback;
  }

  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function readEnvBooleanAny(names, fallback) {
  for (const name of names) {
    if (process.env[name] !== undefined && process.env[name] !== "") {
      return readEnvBoolean(name, fallback);
    }
  }
  return fallback;
}

function resolveDownloadsDir(sku) {
  return path.resolve(process.cwd(), DEFAULTS.downloadsDir, sku);
}

function buildConfig() {
  const skuArg = process.argv[2];
  const fixedSku = String(skuArg || "").trim();

  return {
    databaseUrl: process.env.DATABASE_URL || "",
    downloadsDir: resolveDownloadsDir(fixedSku),
    embeddingModel: process.env.EMBEDDING_MODEL || DEFAULTS.embeddingModel,
    geminiApiKey:
      process.env.GOOGLE_GENERATIVE_AI_API_KEY ||
      process.env.GOOGLE_API_KEY ||
      DEFAULTS.geminiApiKey,
    fixedSku,
    outputDimensionality: readEnvInteger(
      "OUTPUT_DIMENSIONALITY",
      DEFAULTS.outputDimensionality,
    ),
    upsertBatchSize: readEnvInteger(
      "UPSERT_BATCH_SIZE",
      DEFAULTS.upsertBatchSize,
    ),
    queryChunkSize: readEnvInteger(
      "QUERY_CHUNK_SIZE",
      DEFAULTS.queryChunkSize,
    ),
    concurrency: Math.max(
      1,
      readEnvInteger("EMBED_CONCURRENCY", DEFAULTS.concurrency),
    ),
    embedRetryLimit: readEnvInteger(
      "EMBED_RETRY_LIMIT",
      DEFAULTS.embedRetryLimit,
    ),
    embedRetryBaseDelayMs: readEnvInteger(
      "EMBED_RETRY_BASE_DELAY_MS",
      DEFAULTS.embedRetryBaseDelayMs,
    ),
    skipExistingEmbeddings: readEnvBooleanAny(
      ["SKIP_EXISTING_EMBEDDINGS", "SKIP_EXISTING_IN_MILVUS"],
      DEFAULTS.skipExistingEmbeddings,
    ),
    audioTrackExtraction: readEnvBoolean(
      "AUDIO_TRACK_EXTRACTION",
      DEFAULTS.audioTrackExtraction,
    ),
    videoFps: process.env.VIDEO_FPS
      ? readEnvNumber("VIDEO_FPS", null)
      : DEFAULTS.videoFps,
    videoStartOffsetSeconds: process.env.VIDEO_START_OFFSET_SECONDS
      ? readEnvNumber("VIDEO_START_OFFSET_SECONDS", null)
      : DEFAULTS.videoStartOffsetSeconds,
    videoEndOffsetSeconds: process.env.VIDEO_END_OFFSET_SECONDS
      ? readEnvNumber("VIDEO_END_OFFSET_SECONDS", null)
      : DEFAULTS.videoEndOffsetSeconds,
    maxVideos:
      process.env.MAX_VIDEOS && process.env.MAX_VIDEOS !== ""
        ? readEnvInteger("MAX_VIDEOS", null)
        : DEFAULTS.maxVideos,
    embeddingPrompt: process.env.EMBEDDING_PROMPT || DEFAULTS.embeddingPrompt,
  };
}

function ensureLocalInputs(config) {
  if (!config.databaseUrl) {
    throw new Error(
      "Missing DATABASE_URL. Set it in .env or the shell before running.",
    );
  }
  if (!config.geminiApiKey) {
    throw new Error(
      "Missing GOOGLE_GENERATIVE_AI_API_KEY. Set it in .env or the shell before running.",
    );
  }
  if (!config.fixedSku) {
    throw new Error(
      "Missing sku. Usage: node scripts/embed-videos-to-postgres.js <sku>",
    );
  }
  if (!fs.existsSync(config.downloadsDir)) {
    throw new Error(`Downloads directory not found: ${config.downloadsDir}`);
  }
}

function loadVideoPathMap(downloadsDir) {
  const map = new Map();

  function walk(currentDir) {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
        continue;
      }
      if (!entry.isFile()) {
        continue;
      }

      const ext = path.extname(entry.name).toLowerCase();
      if (!ext) {
        continue;
      }

      const base = path.basename(entry.name, ext);
      if (!map.has(base)) {
        map.set(base, fullPath);
      }
    }
  }

  walk(downloadsDir);
  return map;
}

function loadRecords(config) {
  const videoPathMap = loadVideoPathMap(config.downloadsDir);
  let records = Array.from(videoPathMap.entries()).map(([md5, localPath]) => ({
    md5,
    sku: config.fixedSku,
    localPath,
  }));

  records.sort((a, b) => a.md5.localeCompare(b.md5));
  if (typeof config.maxVideos === "number" && config.maxVideos > 0) {
    records = records.slice(0, config.maxVideos);
  }

  return records;
}

function getMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case ".mp4":
      return "video/mp4";
    case ".mov":
      return "video/mov";
    case ".avi":
      return "video/avi";
    case ".webm":
      return "video/webm";
    case ".wmv":
      return "video/wmv";
    case ".mpeg":
    case ".mpg":
      return "video/mpeg";
    default:
      return "video/mp4";
  }
}

function buildVideoPart(record, config) {
  const data = fs.readFileSync(record.localPath);
  const part = {
    inlineData: {
      data: data.toString("base64"),
      mimeType: getMimeType(record.localPath),
    },
  };

  const videoMetadata = {};
  if (typeof config.videoFps === "number") {
    videoMetadata.fps = config.videoFps;
  }
  if (typeof config.videoStartOffsetSeconds === "number") {
    videoMetadata.startOffset = `${config.videoStartOffsetSeconds}s`;
  }
  if (typeof config.videoEndOffsetSeconds === "number") {
    videoMetadata.endOffset = `${config.videoEndOffsetSeconds}s`;
  }
  if (Object.keys(videoMetadata).length > 0) {
    part.videoMetadata = videoMetadata;
  }

  return part;
}

async function embedRecord(ai, record, config) {
  const response = await ai.models.embedContent({
    model: config.embeddingModel,
    contents: {
      role: "user",
      parts: [
        { text: config.embeddingPrompt },
        buildVideoPart(record, config),
      ],
    },
    config: {
      outputDimensionality: config.outputDimensionality,
    },
  });

  const values = response?.embeddings?.[0]?.values;
  if (!Array.isArray(values) || values.length !== config.outputDimensionality) {
    throw new Error(
      `Embedding response for ${record.md5} does not contain a ${config.outputDimensionality}-dimensional vector.`,
    );
  }

  return values;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableEmbeddingError(error) {
  const status =
    error?.status ||
    error?.error?.status ||
    error?.error?.code ||
    error?.code ||
    error?.cause?.code ||
    "";
  const message = String(error?.message || error || "");
  return (
    String(status).includes("503") ||
    String(status).includes("500") ||
    String(status).includes("429") ||
    String(status).includes("ECONNRESET") ||
    String(status).includes("ETIMEDOUT") ||
    String(status).includes("EAI_AGAIN") ||
    String(status).includes("UND_ERR_CONNECT_TIMEOUT") ||
    String(status).includes("UNAVAILABLE") ||
    String(status).includes("RESOURCE_EXHAUSTED") ||
    message.includes('"code":503') ||
    message.includes('"code":500') ||
    message.includes('"code":429') ||
    message.includes("fetch failed") ||
    message.includes("ECONNRESET") ||
    message.includes("ETIMEDOUT") ||
    message.includes("EAI_AGAIN") ||
    message.includes("UND_ERR_CONNECT_TIMEOUT") ||
    message.includes("socket hang up") ||
    message.includes("UNAVAILABLE") ||
    message.includes("RESOURCE_EXHAUSTED")
  );
}

async function embedRecordWithRetry(ai, record, config) {
  let lastError = null;

  for (let attempt = 1; attempt <= config.embedRetryLimit; attempt += 1) {
    try {
      return await embedRecord(ai, record, config);
    } catch (error) {
      lastError = error;
      if (!isRetryableEmbeddingError(error) || attempt === config.embedRetryLimit) {
        throw error;
      }

      const delayMs = config.embedRetryBaseDelayMs * attempt;
      console.warn(
        `[retry ${attempt}/${config.embedRetryLimit}] ${record.md5} after ${delayMs}ms because ${error.message || error}`,
      );
      await sleep(delayMs);
    }
  }

  throw lastError;
}

async function main() {
  const { default: pMap } = await import("p-map");

  const config = buildConfig();
  ensureLocalInputs(config);

  const records = loadRecords(config);
  const withFiles = records;
  const missingFiles = [];
  const store = createVideoEmbeddingsStore(config);

  try {
    console.log(
      JSON.stringify(
        {
          stage: "preflight",
          downloadsDir: config.downloadsDir,
          totalUniqueVideoMd5: records.length,
          localFilesFound: withFiles.length,
          localFilesMissing: missingFiles.length,
          sku: config.fixedSku,
          embeddingModel: config.embeddingModel,
          concurrency: config.concurrency,
          storageBackend: "postgres",
        },
        null,
        2,
      ),
    );

    const databaseDescription = await store.ensureSchema();
    const existingByMd5 = config.skipExistingEmbeddings
      ? await store.fetchExistingRecords(withFiles, config.queryChunkSize)
      : new Map();
    const recordsToEmbed = withFiles.filter((record) => !existingByMd5.has(record.md5));

    const ai = new GoogleGenAI({
      apiKey: config.geminiApiKey,
    });

    const embeddedRows = await pMap(
      recordsToEmbed,
      async (record, index) => {
        console.log(
          `[embed ${index + 1}/${recordsToEmbed.length}] ${record.md5} ${path.basename(record.localPath)}`,
        );

        const embedding = await embedRecordWithRetry(ai, record, config);
        return {
          md5: record.md5,
          sku: record.sku,
          embedding,
        };
      },
      { concurrency: config.concurrency },
    );

    if (embeddedRows.length > 0) {
      await store.upsertEmbeddings(embeddedRows, config.upsertBatchSize);
    }

    console.log(
      JSON.stringify(
        {
          stage: "done",
          sku: config.fixedSku,
          embeddingModel: config.embeddingModel,
          outputDimensionality: config.outputDimensionality,
          concurrency: config.concurrency,
          totalUniqueVideoMd5: records.length,
          localFilesFound: withFiles.length,
          reusedFromPostgres: existingByMd5.size,
          embeddedThisRun: embeddedRows.length,
          missingFiles,
          databaseDescription,
        },
        null,
        2,
      ),
    );
  } finally {
    await store.close();
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
