const fs = require("node:fs");
const path = require("node:path");

const { parse } = require("csv-parse/sync");

const DEFAULTS = {
  threshold: "0.99",
  resultsDir: "results",
  outputsDir: "outputs",
  clusterColumn: "cluster",
};

const MD5_FIELD_CANDIDATES = ["md5", "material_md5", "video_md5", "link_md5"];

function normalizeHeader(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_");
}

function inferSkuFromCsvPath(csvPath) {
  return path.basename(csvPath, path.extname(csvPath));
}

function formatThreshold(value) {
  return String(Number(value));
}

function parseArgs(argv) {
  let csvArg = "";
  let clustersArg = "";
  let outputArg = "";
  let threshold = process.env.CLUSTER_THRESHOLD || DEFAULTS.threshold;
  let sku = "";
  let clusterColumn = DEFAULTS.clusterColumn;

  for (const arg of argv) {
    if (arg.startsWith("--clusters=")) {
      clustersArg = arg.slice("--clusters=".length);
      continue;
    }
    if (arg.startsWith("--output=")) {
      outputArg = arg.slice("--output=".length);
      continue;
    }
    if (arg.startsWith("--threshold=")) {
      threshold = arg.slice("--threshold=".length);
      continue;
    }
    if (arg.startsWith("--sku=")) {
      sku = arg.slice("--sku=".length).trim();
      continue;
    }
    if (arg.startsWith("--column=")) {
      clusterColumn = arg.slice("--column=".length).trim() || DEFAULTS.clusterColumn;
      continue;
    }
    if (!csvArg) {
      csvArg = arg;
      continue;
    }

    throw new Error(
      `Unexpected argument: ${arg}\nUsage: node scripts/write-clusters-to-csv.js <csvPath> [--threshold=0.99] [--clusters=/path/to/clusters.json] [--output=/path/to/output.csv] [--sku=CP69988] [--column=cluster]`,
    );
  }

  if (!csvArg) {
    throw new Error(
      "Missing csvPath. Usage: node scripts/write-clusters-to-csv.js <csvPath> [--threshold=0.99] [--clusters=/path/to/clusters.json] [--output=/path/to/output.csv] [--sku=CP69988] [--column=cluster]",
    );
  }

  const parsedThreshold = Number(threshold);
  if (Number.isNaN(parsedThreshold)) {
    throw new Error(`Invalid threshold: ${threshold}`);
  }

  return {
    csvPath: path.resolve(process.cwd(), csvArg),
    outputPath: path.resolve(process.cwd(), outputArg || csvArg),
    threshold: formatThreshold(parsedThreshold),
    clustersPath: clustersArg ? path.resolve(process.cwd(), clustersArg) : "",
    sku,
    clusterColumn,
  };
}

function buildConfig(argv) {
  const parsed = parseArgs(argv);
  const sku = parsed.sku || inferSkuFromCsvPath(parsed.csvPath);

  const clusterCandidates = parsed.clustersPath
    ? [parsed.clustersPath]
    : [
        path.resolve(
          process.cwd(),
          DEFAULTS.resultsDir,
          sku,
          parsed.threshold,
          "clusters.json",
        ),
        path.resolve(
          process.cwd(),
          DEFAULTS.outputsDir,
          `threshold_${parsed.threshold}`,
          "video-clusters.json",
        ),
        path.resolve(
          process.cwd(),
          DEFAULTS.outputsDir,
          `complete-linkage_${parsed.threshold}`,
          "video-clusters.json",
        ),
        path.resolve(process.cwd(), DEFAULTS.outputsDir, "video-clusters.json"),
      ];

  const clustersPath = clusterCandidates.find((candidate) => fs.existsSync(candidate));

  if (!clustersPath) {
    throw new Error(
      `Could not find a clusters file for sku ${sku}. Checked:\n${clusterCandidates.join("\n")}`,
    );
  }

  return {
    ...parsed,
    sku,
    clustersPath,
  };
}

function ensurePaths(config) {
  if (!fs.existsSync(config.csvPath)) {
    throw new Error(`CSV file not found: ${config.csvPath}`);
  }
}

function loadCsv(config) {
  const text = fs.readFileSync(config.csvPath, "utf8");
  const hasBom = text.charCodeAt(0) === 0xfeff;
  const headerRows = parse(text, {
    bom: true,
    to_line: 1,
    skip_empty_lines: true,
  });

  if (!headerRows.length) {
    throw new Error(`CSV file is empty: ${config.csvPath}`);
  }

  const headers = headerRows[0].map((value) => String(value));
  const rows = parse(text, {
    columns: true,
    bom: true,
    skip_empty_lines: true,
  });

  return {
    hasBom,
    headers,
    rows,
  };
}

function findMd5Field(headers) {
  const normalizedHeaders = headers.map((header) => normalizeHeader(header));
  for (const candidate of MD5_FIELD_CANDIDATES) {
    const index = normalizedHeaders.indexOf(candidate);
    if (index !== -1) {
      return headers[index];
    }
  }

  throw new Error(
    `Could not find an md5 column. Expected one of: ${MD5_FIELD_CANDIDATES.join(", ")}`,
  );
}

function loadClusterMap(config) {
  const clusters = JSON.parse(fs.readFileSync(config.clustersPath, "utf8"));
  if (!Array.isArray(clusters)) {
    throw new Error(`Clusters file must contain an array: ${config.clustersPath}`);
  }

  const clusterMap = new Map();
  let totalMembers = 0;
  for (const cluster of clusters) {
    const clusterId = String(cluster?.clusterId ?? "").trim();
    if (!clusterId) {
      continue;
    }

    for (const member of cluster.members || []) {
      totalMembers += 1;
      if (member?.sku && config.sku && String(member.sku).trim() !== config.sku) {
        continue;
      }

      const md5 = String(member?.md5 || "").trim();
      if (!md5) {
        continue;
      }
      if (clusterMap.has(md5) && clusterMap.get(md5) !== clusterId) {
        throw new Error(
          `Duplicate md5 ${md5} found in multiple clusters (${clusterMap.get(md5)} and ${clusterId}).`,
        );
      }
      clusterMap.set(md5, clusterId);
    }
  }

  if (config.sku && totalMembers > 0 && clusterMap.size === 0) {
    throw new Error(
      `No cluster members matched sku ${config.sku} in ${config.clustersPath}. Pass --clusters=/path/to/clusters.json or --sku=<actual-sku> if needed.`,
    );
  }

  return clusterMap;
}

function escapeCsv(value) {
  const stringValue = String(value ?? "");
  if (/[",\n\r]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, "\"\"")}"`;
  }
  return stringValue;
}

function stringifyCsv(headers, rows, hasBom) {
  const lines = [
    headers.map((header) => escapeCsv(header)).join(","),
    ...rows.map((row) =>
      headers.map((header) => escapeCsv(row[header] ?? "")).join(","),
    ),
  ];

  return `${hasBom ? "\ufeff" : ""}${lines.join("\n")}\n`;
}

function writeClusterColumn(csvData, clusterMap, config) {
  const md5Field = findMd5Field(csvData.headers);
  const headers = csvData.headers.includes(config.clusterColumn)
    ? [...csvData.headers]
    : [...csvData.headers, config.clusterColumn];

  let matchedRows = 0;
  let blankRows = 0;

  const rows = csvData.rows.map((row) => {
    const md5 = String(row[md5Field] || "").trim();
    const clusterId = md5 ? clusterMap.get(md5) || "" : "";
    if (clusterId) {
      matchedRows += 1;
    } else {
      blankRows += 1;
    }

    return {
      ...row,
      [config.clusterColumn]: clusterId,
    };
  });

  return {
    headers,
    rows,
    md5Field,
    matchedRows,
    blankRows,
  };
}

function main() {
  const config = buildConfig(process.argv.slice(2));
  ensurePaths(config);

  const csvData = loadCsv(config);
  const clusterMap = loadClusterMap(config);
  const output = writeClusterColumn(csvData, clusterMap, config);
  const csvText = stringifyCsv(output.headers, output.rows, csvData.hasBom);

  fs.writeFileSync(config.outputPath, csvText, "utf8");

  console.log(
    JSON.stringify(
      {
        stage: "cluster-column-written",
        sku: config.sku,
        csvPath: config.csvPath,
        outputPath: config.outputPath,
        clustersPath: config.clustersPath,
        md5Field: output.md5Field,
        clusterColumn: config.clusterColumn,
        rows: output.rows.length,
        matchedRows: output.matchedRows,
        blankRows: output.blankRows,
        clustersLoaded: clusterMap.size,
      },
      null,
      2,
    ),
  );
}

main();
