const fs = require("node:fs");
const path = require("node:path");

require("dotenv").config({ quiet: true });

const { parse } = require("csv-parse/sync");
const { GoogleGenAI } = require("@google/genai");

const { createVideoEmbeddingsStore } = require("./lib/postgres-video-store");

const DEFAULTS = {
  csvPath: "LF10333042去重素材.csv",
  downloadsDir: "downloads",
  outputsDir: "outputs",
  embeddingModel: "gemini-embedding-2-preview",
  geminiApiKey: "",
  fixedSku: "LF10333042",
  skuField: "material_id",
  md5Field: "video_md5",
  outputDimensionality: 3072,
  upsertBatchSize: 10,
  queryChunkSize: 50,
  concurrency: 1,
  embedRetryLimit: 5,
  embedRetryBaseDelayMs: 2000,
  clusterThreshold: 0.95,
  pairPreviewLimit: 50,
  dryRun: false,
  skipExistingEmbeddings: true,
  clusterAlgorithm: "complete-linkage",
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

function buildConfig() {
  return {
    databaseUrl: process.env.DATABASE_URL || "",
    csvPath: path.resolve(process.cwd(), process.env.CSV_PATH || DEFAULTS.csvPath),
    downloadsDir: path.resolve(
      process.cwd(),
      process.env.DOWNLOADS_DIR || DEFAULTS.downloadsDir,
    ),
    outputsDir: path.resolve(
      process.cwd(),
      process.env.OUTPUTS_DIR || DEFAULTS.outputsDir,
    ),
    embeddingModel: process.env.EMBEDDING_MODEL || DEFAULTS.embeddingModel,
    geminiApiKey:
      process.env.GOOGLE_GENERATIVE_AI_API_KEY ||
      process.env.GOOGLE_API_KEY ||
      DEFAULTS.geminiApiKey,
    fixedSku: process.env.FIXED_SKU || DEFAULTS.fixedSku,
    skuField: process.env.SKU_FIELD || DEFAULTS.skuField,
    md5Field: process.env.MD5_FIELD || DEFAULTS.md5Field,
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
    clusterThreshold: readEnvNumber(
      "CLUSTER_THRESHOLD",
      DEFAULTS.clusterThreshold,
    ),
    pairPreviewLimit: readEnvInteger(
      "PAIR_PREVIEW_LIMIT",
      DEFAULTS.pairPreviewLimit,
    ),
    clusterAlgorithm:
      process.env.CLUSTER_ALGORITHM || DEFAULTS.clusterAlgorithm,
    dryRun: readEnvBoolean("DRY_RUN", DEFAULTS.dryRun),
    skipExistingEmbeddings: readEnvBooleanAny(
      ["SKIP_EXISTING_EMBEDDINGS"],
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
  if (!fs.existsSync(config.csvPath)) {
    throw new Error(`CSV file not found: ${config.csvPath}`);
  }
  if (!fs.existsSync(config.downloadsDir)) {
    throw new Error(`Downloads directory not found: ${config.downloadsDir}`);
  }
  fs.mkdirSync(config.outputsDir, { recursive: true });
}

function loadVideoPathMap(downloadsDir) {
  const map = new Map();
  const entries = fs.readdirSync(downloadsDir, { withFileTypes: true });

  for (const entry of entries) {
    if (!entry.isFile()) {
      continue;
    }
    const ext = path.extname(entry.name).toLowerCase();
    if (!ext) {
      continue;
    }
    const base = path.basename(entry.name, ext);
    map.set(base, path.join(downloadsDir, entry.name));
  }

  return map;
}

function loadRecords(config) {
  const csvText = fs.readFileSync(config.csvPath, "utf8");
  const rows = parse(csvText, {
    columns: true,
    bom: true,
    skip_empty_lines: true,
  });

  const videoPathMap = loadVideoPathMap(config.downloadsDir);
  const unique = new Map();

  for (const row of rows) {
    const md5 = String(row[config.md5Field] || "").trim();
    if (!md5 || unique.has(md5)) {
      continue;
    }

    const sku = String(
      config.fixedSku ||
        row[config.skuField] ||
        row.material_id ||
        row.sku ||
        row.link_md5 ||
        md5,
    ).trim();
    const localPath = videoPathMap.get(md5) || null;

    unique.set(md5, {
      md5,
      sku,
      localPath,
      source: row,
    });
  }

  let records = Array.from(unique.values());
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
  const embedConfig = {
    outputDimensionality: config.outputDimensionality,
  };

  const response = await ai.models.embedContent({
    model: config.embeddingModel,
    contents: {
      role: "user",
      parts: [
        { text: config.embeddingPrompt },
        buildVideoPart(record, config),
      ],
    },
    config: embedConfig,
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
    "";
  const message = String(error?.message || error || "");
  return (
    String(status).includes("503") ||
    String(status).includes("500") ||
    String(status).includes("429") ||
    String(status).includes("UNAVAILABLE") ||
    String(status).includes("RESOURCE_EXHAUSTED") ||
    message.includes('"code":503') ||
    message.includes('"code":500') ||
    message.includes('"code":429') ||
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

function normalizeVector(values) {
  let sum = 0;
  for (const value of values) {
    sum += value * value;
  }
  const magnitude = Math.sqrt(sum);
  if (!magnitude) {
    return values.map(() => 0);
  }
  return values.map((value) => value / magnitude);
}

function cosineSimilarity(normalizedA, normalizedB) {
  let total = 0;
  for (let index = 0; index < normalizedA.length; index += 1) {
    total += normalizedA[index] * normalizedB[index];
  }
  return total;
}

function clusterEmbeddings(records, config) {
  const normalized = records.map((record) => ({
    ...record,
    normalizedEmbedding: normalizeVector(record.embedding),
  }));
  const scoreMatrix = Array.from({ length: normalized.length }, () =>
    Array.from({ length: normalized.length }, () => 0),
  );
  const candidatePairs = [];

  for (let i = 0; i < normalized.length; i += 1) {
    for (let j = i + 1; j < normalized.length; j += 1) {
      const score = cosineSimilarity(
        normalized[i].normalizedEmbedding,
        normalized[j].normalizedEmbedding,
      );
      scoreMatrix[i][j] = score;
      scoreMatrix[j][i] = score;
      if (score >= config.clusterThreshold) {
        candidatePairs.push({
          md5A: normalized[i].md5,
          skuA: normalized[i].sku,
          md5B: normalized[j].md5,
          skuB: normalized[j].sku,
          score: Number(score.toFixed(6)),
        });
      }
    }
  }

  function buildCompleteLinkageClusters() {
    const clusters = normalized.map((_, index) => [index]);

    function clusterScore(clusterA, clusterB) {
      let minScore = Infinity;
      for (const indexA of clusterA) {
        for (const indexB of clusterB) {
          const score = scoreMatrix[indexA][indexB];
          if (score < minScore) {
            minScore = score;
          }
        }
      }
      return minScore;
    }

    while (clusters.length > 1) {
      let bestI = -1;
      let bestJ = -1;
      let bestScore = -Infinity;

      for (let i = 0; i < clusters.length; i += 1) {
        for (let j = i + 1; j < clusters.length; j += 1) {
          const score = clusterScore(clusters[i], clusters[j]);
          if (score > bestScore) {
            bestScore = score;
            bestI = i;
            bestJ = j;
          }
        }
      }

      if (bestScore < config.clusterThreshold || bestI === -1 || bestJ === -1) {
        break;
      }

      const merged = [...clusters[bestI], ...clusters[bestJ]];
      clusters[bestI] = merged;
      clusters.splice(bestJ, 1);
    }

    return clusters;
  }

  if (config.clusterAlgorithm !== "complete-linkage") {
    throw new Error(
      `Unsupported cluster algorithm: ${config.clusterAlgorithm}. Use complete-linkage.`,
    );
  }
  const clusters = buildCompleteLinkageClusters();

  const formattedClusters = clusters
    .map((memberIndexes, index) => ({
      clusterId: index + 1,
      size: memberIndexes.length,
      representativeMd5: normalized[memberIndexes[0]].md5,
      members: memberIndexes
        .map((memberIndex) => ({
          md5: normalized[memberIndex].md5,
          sku: normalized[memberIndex].sku,
          localPath: normalized[memberIndex].localPath || null,
          source: normalized[memberIndex].source || "embedded",
        }))
        .sort((a, b) => a.md5.localeCompare(b.md5)),
    }))
    .sort((a, b) => b.size - a.size || a.representativeMd5.localeCompare(b.representativeMd5));

  candidatePairs.sort((a, b) => b.score - a.score || a.md5A.localeCompare(b.md5A));

  return {
    clusters: formattedClusters,
    candidatePairs: candidatePairs.slice(0, config.pairPreviewLimit),
  };
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

async function main() {
  const config = buildConfig();
  ensureLocalInputs(config);

  const records = loadRecords(config);
  const withFiles = records.filter((record) => record.localPath);
  const missingFiles = records
    .filter((record) => !record.localPath)
    .map((record) => record.md5);

  console.log(
    JSON.stringify(
      {
        stage: "preflight",
        totalUniqueVideoMd5: records.length,
        localFilesFound: withFiles.length,
        localFilesMissing: missingFiles.length,
        dryRun: config.dryRun,
        clusterAlgorithm: config.clusterAlgorithm,
        hasGeminiApiKey: Boolean(config.geminiApiKey),
        storageBackend: "postgres",
      },
      null,
      2,
    ),
  );

  const store = createVideoEmbeddingsStore(config);

  try {
    const databaseDescription = await store.ensureSchema();

    if (!config.dryRun && !config.geminiApiKey) {
      throw new Error(
        "Missing Gemini authentication. Set GOOGLE_GENERATIVE_AI_API_KEY before running the pipeline.",
      );
    }

    const existingByMd5 = config.skipExistingEmbeddings
      ? await store.fetchExistingRecords(withFiles, config.queryChunkSize)
      : new Map();
    const recordsToEmbed = withFiles.filter((record) => !existingByMd5.has(record.md5));

    let embeddedRows = [];
    if (!config.dryRun && recordsToEmbed.length > 0) {
      const ai = new GoogleGenAI({
        apiKey: config.geminiApiKey,
      });

      const pendingUpserts = [];
      for (let index = 0; index < recordsToEmbed.length; index += 1) {
        const record = recordsToEmbed[index];
        console.log(
          `[embed ${index + 1}/${recordsToEmbed.length}] ${record.md5} ${path.basename(record.localPath)}`,
        );
        const embedding = await embedRecordWithRetry(ai, record, config);
        const row = {
          md5: record.md5,
          sku: record.sku,
          localPath: record.localPath,
          source: "vertex",
          embedding,
        };
        embeddedRows.push(row);
        pendingUpserts.push(row);

        if (pendingUpserts.length >= config.upsertBatchSize) {
          await store.upsertEmbeddings(
            pendingUpserts.splice(0, pendingUpserts.length),
            config.upsertBatchSize,
          );
        }
      }

      if (pendingUpserts.length > 0) {
        await store.upsertEmbeddings(pendingUpserts, config.upsertBatchSize);
      }
    }

    const embeddedByMd5 = new Map(
      embeddedRows.map((item) => [item.md5, item]),
    );
    const clusterInput = [];
    for (const record of withFiles) {
      const fromPostgres = existingByMd5.get(record.md5);
      if (fromPostgres) {
        clusterInput.push({
          md5: record.md5,
          sku: fromPostgres.sku || record.sku,
          localPath: record.localPath,
          source: "postgres",
          embedding: fromPostgres.embedding,
        });
        continue;
      }

      const fromEmbed = embeddedByMd5.get(record.md5);
      if (fromEmbed) {
        clusterInput.push(fromEmbed);
      }
    }

    const timestamp = new Date().toISOString();
    const summary = {
      generatedAt: timestamp,
      embeddingModel: config.embeddingModel,
      outputDimensionality: config.outputDimensionality,
      clusterAlgorithm: config.clusterAlgorithm,
      clusterThreshold: config.clusterThreshold,
      totalUniqueVideoMd5: records.length,
      localFilesFound: withFiles.length,
      localFilesMissing: missingFiles.length,
      reusedFromPostgres: clusterInput.filter((item) => item.source === "postgres").length,
      embeddedThisRun: clusterInput.filter((item) => item.source === "vertex").length,
      clusteredVideos: clusterInput.length,
      dryRun: config.dryRun,
      hasGeminiApiKey: Boolean(config.geminiApiKey),
      databaseDescription,
      missingFiles,
    };

    if (clusterInput.length === 0) {
      const summaryPath = path.join(config.outputsDir, "video-cluster-summary.json");
      writeJson(summaryPath, summary);
      console.log(JSON.stringify({ stage: "done", summaryPath, summary }, null, 2));
      return;
    }

    const clusterResult = clusterEmbeddings(clusterInput, config);
    summary.estimatedUniqueVideos = clusterResult.clusters.length;
    summary.clusterSizes = clusterResult.clusters.map((cluster) => cluster.size);

    const summaryPath = path.join(config.outputsDir, "video-cluster-summary.json");
    const clustersPath = path.join(config.outputsDir, "video-clusters.json");
    const pairsPath = path.join(config.outputsDir, "video-similar-pairs.json");

    writeJson(summaryPath, summary);
    writeJson(clustersPath, clusterResult.clusters);
    writeJson(pairsPath, clusterResult.candidatePairs);

    console.log(
      JSON.stringify(
        {
          stage: "done",
          summaryPath,
          clustersPath,
          pairsPath,
          summary,
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
