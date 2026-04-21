const path = require("node:path");
const { spawn } = require("node:child_process");

const DEFAULT_THRESHOLD = "0.99";
const DOWNLOAD_FULL_PASSES = 2;

function parseArgs(argv) {
  const skus = [];
  let threshold = process.env.CLUSTER_THRESHOLD || DEFAULT_THRESHOLD;

  for (const arg of argv) {
    if (arg.startsWith("--threshold=")) {
      threshold = arg.slice("--threshold=".length);
      continue;
    }
    skus.push(arg);
  }

  if (skus.length === 0) {
    throw new Error(
      "Missing sku list. Usage: node scripts/process-skus.js [--threshold=0.99] <sku> <sku> ...",
    );
  }

  const parsedThreshold = Number(threshold);
  if (Number.isNaN(parsedThreshold)) {
    throw new Error(`Invalid threshold: ${threshold}`);
  }

  return {
    skus,
    threshold: String(parsedThreshold),
  };
}

function runNodeScript(scriptName, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [path.join(__dirname, scriptName), ...args],
      {
        cwd: path.resolve(__dirname, ".."),
        stdio: "inherit",
        env: process.env,
      },
    );

    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve();
        return;
      }
      const error = new Error(`${scriptName} exited with code ${code}`);
      error.exitCode = code;
      reject(error);
    });
  });
}

async function runDownloadWithRetry(sku) {
  const csvPath = `sku/${sku}.csv`;
  const outputDir = `downloads/${sku}`;
  let lastError = null;

  for (let attempt = 1; attempt <= DOWNLOAD_FULL_PASSES; attempt += 1) {
    try {
      await runNodeScript("download-videos-from-csv.js", [csvPath, outputDir]);
      return { attempts: attempt };
    } catch (error) {
      lastError = error;
      if (attempt === DOWNLOAD_FULL_PASSES) {
        break;
      }
      console.warn(
        `[batch retry ${attempt}/${DOWNLOAD_FULL_PASSES}] download ${sku} because ${error.message || error}`,
      );
    }
  }

  throw lastError;
}

async function processSku(sku, threshold, index, total) {
  console.log(
    JSON.stringify(
      {
        stage: "batch-start",
        sku,
        position: `${index + 1}/${total}`,
        threshold,
      },
      null,
      2,
    ),
  );

  const startedAt = Date.now();
  const downloadResult = await runDownloadWithRetry(sku);
  await runNodeScript("embed-videos-to-postgres.js", [sku]);
  await runNodeScript("cluster-videos.js", [sku, threshold]);

  return {
    sku,
    status: "ok",
    threshold,
    downloadPasses: downloadResult.attempts,
    durationSeconds: Number(((Date.now() - startedAt) / 1000).toFixed(1)),
  };
}

async function main() {
  const { skus, threshold } = parseArgs(process.argv.slice(2));
  const results = [];

  for (let index = 0; index < skus.length; index += 1) {
    const sku = skus[index];
    try {
      const result = await processSku(sku, threshold, index, skus.length);
      results.push(result);
    } catch (error) {
      results.push({
        sku,
        status: "failed",
        threshold,
        error: error.message || String(error),
      });
      console.error(
        JSON.stringify(
          {
            stage: "batch-failed",
            sku,
            error: error.message || String(error),
          },
          null,
          2,
        ),
      );
      break;
    }
  }

  console.log(
    JSON.stringify(
      {
        stage: "batch-done",
        threshold,
        results,
      },
      null,
      2,
    ),
  );

  if (results.some((item) => item.status === "failed")) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
