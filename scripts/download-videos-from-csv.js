const fs = require("node:fs");
const path = require("node:path");
const { pipeline } = require("node:stream/promises");

require("dotenv").config({ quiet: true });

const { parse } = require("csv-parse/sync");
const XLSX = require("xlsx");

const DEFAULTS = {
  csvPath: "",
  outputDir: "",
  md5Field: "md5",
  skuField: "sku",
  urlField: "url",
  spreadsheetDir: "sku",
  requestTimeoutMs: 60000,
  retryLimit: 3,
  retryBaseDelayMs: 2000,
};
const MD5_FIELD_CANDIDATES = ["md5", "material_md5", "video_md5", "link_md5"];
const URL_FIELD_CANDIDATES = ["url", "replace", "video_url", "video", "link"];

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

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildConfig() {
  const csvArg = process.argv[2] || process.env.CSV_PATH || DEFAULTS.csvPath;
  const outputArg = process.argv[3] || process.env.DOWNLOADS_DIR || DEFAULTS.outputDir;

  if (!csvArg) {
    throw new Error(
      "Missing CSV path. Usage: node scripts/download-videos-from-csv.js <csvPath> <outputDir>",
    );
  }

  if (!outputArg) {
    throw new Error(
      "Missing output directory. Usage: node scripts/download-videos-from-csv.js <csvPath> <outputDir>",
    );
  }

  return {
    csvPath: path.resolve(process.cwd(), csvArg),
    outputDir: path.resolve(process.cwd(), outputArg),
    md5Field: process.env.MD5_FIELD || DEFAULTS.md5Field,
    skuField: process.env.SKU_FIELD || DEFAULTS.skuField,
    urlField: process.env.URL_FIELD || DEFAULTS.urlField,
    spreadsheetDir: path.resolve(
      process.cwd(),
      process.env.SPREADSHEET_DIR || DEFAULTS.spreadsheetDir,
    ),
    requestTimeoutMs: readEnvNumber(
      "REQUEST_TIMEOUT_MS",
      DEFAULTS.requestTimeoutMs,
    ),
    retryLimit: readEnvNumber("DOWNLOAD_RETRY_LIMIT", DEFAULTS.retryLimit),
    retryBaseDelayMs: readEnvNumber(
      "DOWNLOAD_RETRY_BASE_DELAY_MS",
      DEFAULTS.retryBaseDelayMs,
    ),
  };
}

function ensurePaths(config) {
  if (!fs.existsSync(config.csvPath)) {
    throw new Error(`CSV file not found: ${config.csvPath}`);
  }

  fs.mkdirSync(config.outputDir, { recursive: true });
}

function normalizeHeader(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
}

function inferSkuFromCsvPath(csvPath) {
  return path.basename(csvPath, path.extname(csvPath));
}

function findSpreadsheetForSku(config, sku) {
  const directCandidates = [
    path.join(config.spreadsheetDir, `${sku}.xlsx`),
    path.join(config.spreadsheetDir, `${sku}.xls`),
  ];
  for (const candidate of directCandidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  const entries = fs.readdirSync(config.spreadsheetDir, { withFileTypes: true });
  const matched = entries
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .filter((name) =>
      new RegExp(`^${sku}(?:_[0-9]+)?\\.(xlsx|xls)$`, "i").test(name),
    )
    .sort();

  if (matched.length === 0) {
    throw new Error(
      `Could not find a matching spreadsheet for sku ${sku} in ${config.spreadsheetDir}.`,
    );
  }

  return path.join(config.spreadsheetDir, matched[0]);
}

function findColumnIndex(headerRow, candidates) {
  const normalizedHeader = headerRow.map((value) => normalizeHeader(value));
  for (const candidate of candidates) {
    const index = normalizedHeader.indexOf(candidate);
    if (index !== -1) {
      return index;
    }
  }
  return -1;
}

function loadUrlMapFromSpreadsheet(spreadsheetPath) {
  const workbook = XLSX.readFile(spreadsheetPath, { cellDates: false });
  const firstSheetName = workbook.SheetNames[0];
  if (!firstSheetName) {
    throw new Error(`Spreadsheet has no sheets: ${spreadsheetPath}`);
  }

  const sheet = workbook.Sheets[firstSheetName];
  const rows = XLSX.utils.sheet_to_json(sheet, {
    header: 1,
    blankrows: false,
    defval: "",
  });

  if (rows.length === 0) {
    return new Map();
  }

  const headerRow = rows[0];
  const md5Index = findColumnIndex(headerRow, MD5_FIELD_CANDIDATES);
  const urlIndex = findColumnIndex(headerRow, URL_FIELD_CANDIDATES);

  if (md5Index === -1) {
    throw new Error(`Could not find an md5 column in spreadsheet ${spreadsheetPath}.`);
  }
  if (urlIndex === -1) {
    throw new Error(`Could not find a url column in spreadsheet ${spreadsheetPath}.`);
  }

  const map = new Map();
  for (const row of rows.slice(1)) {
    const md5 = String(row[md5Index] || "").trim();
    const url = String(row[urlIndex] || "").trim();
    if (!md5 || !url || map.has(md5)) {
      continue;
    }
    const normalizedUrl = /^https?:\/\//i.test(url) ? url : `https://${url}`;
    map.set(md5, normalizedUrl);
  }

  return map;
}

function inferExtension(urlString) {
  try {
    const pathname = new URL(urlString).pathname.toLowerCase();
    const base = path.basename(pathname);
    if (base.includes(".")) {
      const ext = `.${base.split(".").pop()}`;
      if (ext.length > 1 && ext.length <= 10) {
        return ext;
      }
    }
  } catch {
    // Fall through to the default extension.
  }

  return ".mp4";
}

function loadRows(config) {
  const csvText = fs.readFileSync(config.csvPath, "utf8");
  const parsedRows = parse(csvText, {
    columns: true,
    bom: true,
    skip_empty_lines: true,
  });

  const rowsWithUrls = [];
  const unique = new Map();
  let inferredSku = "";
  for (const row of parsedRows) {
    const md5 = String(row[config.md5Field] || "").trim();
    const sku = String(row[config.skuField] || "").trim();
    const url = String(row[config.urlField] || "").trim();
    if (!md5 || unique.has(md5)) {
      continue;
    }
    unique.set(md5, { md5 });
    if (!inferredSku && sku) {
      inferredSku = sku;
    }
    if (url) {
      rowsWithUrls.push({
        md5,
        url,
      });
    }
  }

  if (rowsWithUrls.length === unique.size) {
    return {
      rows: rowsWithUrls,
      sourceType: "csv",
      sourcePath: config.csvPath,
      missingUrlMd5s: [],
    };
  }

  const sku = inferredSku || inferSkuFromCsvPath(config.csvPath);
  const spreadsheetPath = findSpreadsheetForSku(config, sku);
  const urlByMd5 = loadUrlMapFromSpreadsheet(spreadsheetPath);
  const missingUrlMd5s = [];
  const rows = [];

  for (const { md5 } of unique.values()) {
    const url = urlByMd5.get(md5);
    if (!url) {
      missingUrlMd5s.push(md5);
      continue;
    }
    rows.push({ md5, url });
  }

  return {
    rows,
    sourceType: "spreadsheet",
    sourcePath: spreadsheetPath,
    missingUrlMd5s,
  };
}

async function downloadFile(url, destinationPath, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      headers: { "User-Agent": "Mozilla/5.0" },
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error("Empty response body");
    }

    await pipeline(response.body, fs.createWriteStream(destinationPath));
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const config = buildConfig();
  ensurePaths(config);

  const { rows, sourceType, sourcePath, missingUrlMd5s } = loadRows(config);
  console.log(
    JSON.stringify(
      {
        stage: "preflight",
        csvPath: config.csvPath,
        outputDir: config.outputDir,
        md5Field: config.md5Field,
        urlField: config.urlField,
        sourceType,
        sourcePath,
        totalUniqueRows: rows.length,
        missingUrlMd5Count: missingUrlMd5s.length,
      },
      null,
      2,
    ),
  );

  let downloadedNow = 0;
  let skippedExisting = 0;
  const failed = [];

  for (let index = 0; index < rows.length; index += 1) {
    const { md5, url } = rows[index];
    const extension = inferExtension(url);
    const destinationPath = path.join(config.outputDir, `${md5}${extension}`);
    const tempPath = `${destinationPath}.part`;

    if (fs.existsSync(destinationPath) && fs.statSync(destinationPath).size > 0) {
      skippedExisting += 1;
      console.log(`[${index + 1}/${rows.length}] skip ${path.basename(destinationPath)}`);
      continue;
    }

    let lastError = null;
    for (let attempt = 1; attempt <= config.retryLimit; attempt += 1) {
      try {
        await downloadFile(url, tempPath, config.requestTimeoutMs);
        fs.renameSync(tempPath, destinationPath);
        downloadedNow += 1;
        const sizeMb = fs.statSync(destinationPath).size / 1024 / 1024;
        console.log(
          `[${index + 1}/${rows.length}] ok   ${path.basename(destinationPath)} ${sizeMb.toFixed(2)} MB`,
        );
        lastError = null;
        break;
      } catch (error) {
        lastError = error;
        if (fs.existsSync(tempPath)) {
          fs.rmSync(tempPath, { force: true });
        }

        if (attempt < config.retryLimit) {
          const delayMs = config.retryBaseDelayMs * attempt;
          console.warn(
            `[retry ${attempt}/${config.retryLimit}] ${md5} after ${delayMs}ms because ${error.message || error}`,
          );
          await sleep(delayMs);
        }
      }
    }

    if (lastError) {
      failed.push({
        md5,
        url,
        error: lastError.message || String(lastError),
      });
      console.log(
        `[${index + 1}/${rows.length}] fail ${path.basename(destinationPath)} ${lastError.message || lastError}`,
      );
    }
  }

  console.log(
    JSON.stringify(
      {
        stage: "done",
        totalUniqueRows: rows.length,
        downloadedNow,
        skippedExisting,
        failedCount: failed.length,
        missingUrlMd5Count: missingUrlMd5s.length,
      },
      null,
      2,
    ),
  );

  if (missingUrlMd5s.length > 0) {
    const missingPath = path.join(config.outputDir, "_missing_urls.json");
    fs.writeFileSync(missingPath, JSON.stringify(missingUrlMd5s, null, 2), "utf8");
    console.error(`Wrote missing-url md5 list to ${missingPath}`);
  }

  if (failed.length > 0) {
    const failuresPath = path.join(config.outputDir, "_download_failures.json");
    fs.writeFileSync(failuresPath, JSON.stringify(failed, null, 2), "utf8");
    console.error(`Wrote failure details to ${failuresPath}`);
    process.exitCode = 2;
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
