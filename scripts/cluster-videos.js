const fs = require("node:fs");
const path = require("node:path");

require("dotenv").config({ quiet: true });

const { createVideoEmbeddingsStore } = require("./lib/postgres-video-store");

const DEFAULTS = {
  downloadsDir: "downloads",
  resultsDir: "results",
  outputDimensionality: 3072,
  clusterThreshold: 0.98,
  clusterAlgorithm: "complete-linkage",
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

function formatThreshold(value) {
  return String(value);
}

function buildConfig() {
  const sku = String(process.argv[2] || "").trim();
  const thresholdArg = process.argv[3];
  const threshold = thresholdArg
    ? Number(thresholdArg)
    : readEnvNumber("CLUSTER_THRESHOLD", DEFAULTS.clusterThreshold);

  if (Number.isNaN(threshold)) {
    throw new Error(`Invalid threshold: ${thresholdArg}`);
  }

  return {
    databaseUrl: process.env.DATABASE_URL || "",
    sku,
    clusterThreshold: threshold,
    clusterAlgorithm:
      process.env.CLUSTER_ALGORITHM || DEFAULTS.clusterAlgorithm,
    downloadsDir: path.resolve(process.cwd(), DEFAULTS.downloadsDir, sku),
    resultsDir: path.resolve(
      process.cwd(),
      DEFAULTS.resultsDir,
      sku,
      formatThreshold(threshold),
    ),
    outputDimensionality: readEnvNumber(
      "OUTPUT_DIMENSIONALITY",
      DEFAULTS.outputDimensionality,
    ),
  };
}

function ensureLocalInputs(config) {
  if (!config.databaseUrl) {
    throw new Error(
      "Missing DATABASE_URL. Set it in .env or the shell before running.",
    );
  }
  if (!config.sku) {
    throw new Error(
      "Missing sku. Usage: node scripts/cluster-videos.js <sku> [threshold]",
    );
  }
  if (!fs.existsSync(config.downloadsDir)) {
    throw new Error(`Downloads directory not found: ${config.downloadsDir}`);
  }
  fs.mkdirSync(config.resultsDir, { recursive: true });
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

      const ext = path.extname(entry.name);
      if (!ext) {
        continue;
      }

      const md5 = path.basename(entry.name, ext);
      if (!map.has(md5)) {
        map.set(md5, fullPath);
      }
    }
  }

  walk(downloadsDir);
  return map;
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

  for (let i = 0; i < normalized.length; i += 1) {
    for (let j = i + 1; j < normalized.length; j += 1) {
      const score = cosineSimilarity(
        normalized[i].normalizedEmbedding,
        normalized[j].normalizedEmbedding,
      );
      scoreMatrix[i][j] = score;
      scoreMatrix[j][i] = score;
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

      clusters[bestI] = [...clusters[bestI], ...clusters[bestJ]];
      clusters.splice(bestJ, 1);
    }

    return clusters;
  }

  function buildSingleLinkageClusters() {
    const parents = Array.from({ length: normalized.length }, (_, index) => index);

    function find(index) {
      if (parents[index] !== index) {
        parents[index] = find(parents[index]);
      }
      return parents[index];
    }

    function union(a, b) {
      const rootA = find(a);
      const rootB = find(b);
      if (rootA !== rootB) {
        parents[rootB] = rootA;
      }
    }

    for (let i = 0; i < normalized.length; i += 1) {
      for (let j = i + 1; j < normalized.length; j += 1) {
        if (scoreMatrix[i][j] >= config.clusterThreshold) {
          union(i, j);
        }
      }
    }

    const groups = new Map();
    for (let i = 0; i < normalized.length; i += 1) {
      const root = find(i);
      if (!groups.has(root)) {
        groups.set(root, []);
      }
      groups.get(root).push(i);
    }

    return Array.from(groups.values());
  }

  let clusters;
  if (config.clusterAlgorithm === "single-linkage") {
    clusters = buildSingleLinkageClusters();
  } else if (config.clusterAlgorithm === "complete-linkage") {
    clusters = buildCompleteLinkageClusters();
  } else {
    throw new Error(
      `Unsupported cluster algorithm: ${config.clusterAlgorithm}. Use complete-linkage or single-linkage.`,
    );
  }

  return clusters
    .map((memberIndexes, index) => ({
      clusterId: String(index + 1),
      size: memberIndexes.length,
      representativeMd5: normalized[memberIndexes[0]].md5,
      members: memberIndexes
        .map((memberIndex) => ({
          md5: normalized[memberIndex].md5,
          sku: normalized[memberIndex].sku,
        }))
        .sort((a, b) => a.md5.localeCompare(b.md5)),
    }))
    .sort((a, b) => b.size - a.size || a.representativeMd5.localeCompare(b.representativeMd5))
    .map((cluster, index) => ({
      ...cluster,
      clusterId: String(index + 1),
    }));
}

function resetDirectoryContents(dirPath) {
  fs.rmSync(dirPath, { recursive: true, force: true });
  fs.mkdirSync(dirPath, { recursive: true });
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf8");
}

function materializeResults(config, clusters, videoPathMap) {
  resetDirectoryContents(config.resultsDir);

  const missingFiles = [];

  for (const cluster of clusters) {
    const clusterDir = path.join(config.resultsDir, cluster.clusterId);
    fs.mkdirSync(clusterDir, { recursive: true });

    for (const member of cluster.members) {
      const sourcePath = videoPathMap.get(member.md5);
      if (!sourcePath) {
        missingFiles.push(member.md5);
        continue;
      }

      const extension = path.extname(sourcePath) || ".mp4";
      const targetPath = path.join(clusterDir, `${member.md5}${extension}`);
      if (fs.existsSync(targetPath)) {
        fs.rmSync(targetPath, { force: true });
      }
      fs.linkSync(sourcePath, targetPath);
    }
  }

  return missingFiles.sort((a, b) => a.localeCompare(b));
}

async function main() {
  const config = buildConfig();
  ensureLocalInputs(config);

  console.log(
    JSON.stringify(
      {
        stage: "preflight",
        sku: config.sku,
        clusterThreshold: config.clusterThreshold,
        clusterAlgorithm: config.clusterAlgorithm,
        downloadsDir: config.downloadsDir,
        resultsDir: config.resultsDir,
        storageBackend: "postgres",
      },
      null,
      2,
    ),
  );

  const store = createVideoEmbeddingsStore(config);

  try {
    const databaseDescription = await store.ensureSchema();
    const records = await store.fetchSkuRecords(config.sku);
    const videoPathMap = loadVideoPathMap(config.downloadsDir);

    if (records.length === 0) {
      throw new Error(`No embeddings found in PostgreSQL for sku ${config.sku}.`);
    }

    const clusters = clusterEmbeddings(records, config);
    const missingFiles = materializeResults(config, clusters, videoPathMap);

    const summary = {
      generatedAt: new Date().toISOString(),
      sku: config.sku,
      clusterThreshold: config.clusterThreshold,
      clusterAlgorithm: config.clusterAlgorithm,
      clusteredVideos: records.length,
      estimatedUniqueVideos: clusters.length,
      clusterSizes: clusters.map((cluster) => cluster.size),
      resultsDir: config.resultsDir,
      downloadsDir: config.downloadsDir,
      missingFiles,
      databaseDescription,
    };

    writeJson(path.join(config.resultsDir, "summary.json"), summary);
    writeJson(path.join(config.resultsDir, "clusters.json"), clusters);

    console.log(
      JSON.stringify(
        {
          stage: "done",
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
