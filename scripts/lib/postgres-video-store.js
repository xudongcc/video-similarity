const { Pool } = require("pg");

const { asc, eq, inArray, sql } = require("drizzle-orm");
const { drizzle } = require("drizzle-orm/node-postgres");
const {
  index,
  integer,
  jsonb,
  pgTable,
  text,
  timestamp,
} = require("drizzle-orm/pg-core");

const VIDEO_EMBEDDINGS_TABLE_NAME = "video_embeddings";

const videoEmbeddings = pgTable(
  VIDEO_EMBEDDINGS_TABLE_NAME,
  {
    md5: text("md5").primaryKey(),
    sku: text("sku").notNull(),
    embedding: jsonb("embedding").notNull(),
    embeddingDim: integer("embedding_dim").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => [index("video_embeddings_sku_idx").on(table.sku)],
);

function chunkArray(items, chunkSize) {
  const chunks = [];
  for (let index = 0; index < items.length; index += chunkSize) {
    chunks.push(items.slice(index, index + chunkSize));
  }
  return chunks;
}

function normalizeEmbedding(md5, embedding, expectedDim) {
  if (!Array.isArray(embedding)) {
    throw new Error(`Embedding for ${md5} is not an array.`);
  }

  const normalized = embedding.map((value) => Number(value));
  if (normalized.some((value) => Number.isNaN(value))) {
    throw new Error(`Embedding for ${md5} contains non-numeric values.`);
  }

  if (normalized.length !== expectedDim) {
    throw new Error(
      `Embedding for ${md5} has dimension ${normalized.length}, expected ${expectedDim}.`,
    );
  }

  return normalized;
}

function createVideoEmbeddingsStore(config) {
  const databaseUrl = String(config.databaseUrl || "").trim();
  if (!databaseUrl) {
    throw new Error(
      "Missing DATABASE_URL. Set it in .env or the shell before running.",
    );
  }

  const expectedDim = Math.trunc(Number(config.outputDimensionality));
  if (!Number.isInteger(expectedDim) || expectedDim <= 0) {
    throw new Error(
      `outputDimensionality must be a positive integer, got: ${config.outputDimensionality}`,
    );
  }

  const pool = new Pool({
    connectionString: databaseUrl,
  });
  const db = drizzle({ client: pool });
  let schemaEnsured = false;

  function describeStore() {
    return {
      backend: "drizzle-orm/node-postgres",
      tableName: VIDEO_EMBEDDINGS_TABLE_NAME,
      md5PrimaryKey: true,
      skuIndexed: true,
      embeddingColumnType: "jsonb",
      embeddingDimensions: expectedDim,
    };
  }

  async function ensureSchema() {
    if (schemaEnsured) {
      return describeStore();
    }

    await db.execute(sql`
      CREATE TABLE IF NOT EXISTS video_embeddings (
        md5 text PRIMARY KEY,
        sku text NOT NULL,
        embedding jsonb NOT NULL,
        embedding_dim integer NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT video_embeddings_embedding_is_array
          CHECK (jsonb_typeof(embedding) = 'array'),
        CONSTRAINT video_embeddings_embedding_dim_positive
          CHECK (embedding_dim > 0),
        CONSTRAINT video_embeddings_embedding_dim_matches_json
          CHECK (jsonb_array_length(embedding) = embedding_dim)
      )
    `);
    await db.execute(sql`
      CREATE INDEX IF NOT EXISTS video_embeddings_sku_idx
      ON video_embeddings (sku)
    `);

    schemaEnsured = true;
    return describeStore();
  }

  async function fetchExistingRecords(records, queryChunkSize) {
    await ensureSchema();

    const md5Values = Array.from(
      new Set(
        records
          .map((record) =>
            typeof record === "string" ? record : String(record?.md5 || "").trim(),
          )
          .filter(Boolean),
      ),
    );

    if (md5Values.length === 0) {
      return new Map();
    }

    const found = new Map();
    const chunks = chunkArray(md5Values, Math.max(1, queryChunkSize || 50));

    for (const chunk of chunks) {
      const rows = await db
        .select({
          md5: videoEmbeddings.md5,
          sku: videoEmbeddings.sku,
          embedding: videoEmbeddings.embedding,
          embeddingDim: videoEmbeddings.embeddingDim,
        })
        .from(videoEmbeddings)
        .where(inArray(videoEmbeddings.md5, chunk))
        .orderBy(asc(videoEmbeddings.md5));

      for (const row of rows) {
        const embedding = normalizeEmbedding(
          row.md5,
          row.embedding,
          expectedDim,
        );
        if (row.embeddingDim !== expectedDim) {
          throw new Error(
            `Stored embedding dimension for ${row.md5} is ${row.embeddingDim}, expected ${expectedDim}.`,
          );
        }

        found.set(row.md5, {
          md5: row.md5,
          sku: row.sku,
          embedding,
        });
      }
    }

    return found;
  }

  async function fetchSkuRecords(sku) {
    await ensureSchema();

    const rows = await db
      .select({
        md5: videoEmbeddings.md5,
        sku: videoEmbeddings.sku,
        embedding: videoEmbeddings.embedding,
        embeddingDim: videoEmbeddings.embeddingDim,
      })
      .from(videoEmbeddings)
      .where(eq(videoEmbeddings.sku, sku))
      .orderBy(asc(videoEmbeddings.md5));

    return rows.map((row) => {
      if (row.embeddingDim !== expectedDim) {
        throw new Error(
          `Stored embedding dimension for ${row.md5} is ${row.embeddingDim}, expected ${expectedDim}.`,
        );
      }

      return {
        md5: row.md5,
        sku: row.sku,
        embedding: normalizeEmbedding(row.md5, row.embedding, expectedDim),
      };
    });
  }

  async function upsertEmbeddings(rows, upsertBatchSize) {
    await ensureSchema();

    const chunks = chunkArray(rows, Math.max(1, upsertBatchSize || 20));

    for (const chunk of chunks) {
      await db.transaction(async (tx) => {
        for (const row of chunk) {
          const md5 = String(row.md5 || "").trim();
          const sku = String(row.sku || "").trim();
          if (!md5) {
            throw new Error("Cannot upsert an embedding without md5.");
          }
          if (!sku) {
            throw new Error(`Cannot upsert ${md5} without sku.`);
          }

          const embedding = normalizeEmbedding(md5, row.embedding, expectedDim);
          const payload = {
            md5,
            sku,
            embedding,
            embeddingDim: embedding.length,
          };

          await tx
            .insert(videoEmbeddings)
            .values(payload)
            .onConflictDoUpdate({
              target: videoEmbeddings.md5,
              set: {
                sku: payload.sku,
                embedding: payload.embedding,
                embeddingDim: payload.embeddingDim,
                updatedAt: sql`now()`,
              },
            });
        }
      });
    }
  }

  async function close() {
    await pool.end();
  }

  return {
    close,
    describeStore,
    ensureSchema,
    fetchExistingRecords,
    fetchSkuRecords,
    upsertEmbeddings,
  };
}

module.exports = {
  VIDEO_EMBEDDINGS_TABLE_NAME,
  createVideoEmbeddingsStore,
};
