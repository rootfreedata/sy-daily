import OpenAI from "openai";

/**
 * Embedding client — works with any **OpenAI-compatible** `/embeddings` endpoint.
 *
 * Defaults preserve upstream behaviour (OpenAI `text-embedding-3-small`), but the three
 * env vars below let you point it at whichever provider you already pay for — no OpenAI
 * account required:
 *
 *   EMBEDDING_API_URL   e.g. https://dashscope.aliyuncs.com/compatible-mode/v1  (Qwen / Bailian)
 *                            https://open.bigmodel.cn/api/paas/v4                 (Zhipu GLM)
 *   EMBEDDING_API_KEY   provider key (falls back to OPENAI_API_KEY)
 *   EMBEDDING_MODEL     e.g. text-embedding-v4 (Qwen) / embedding-3 (Zhipu)
 *
 * NOTE: vector length differs per model (OpenAI small = 1536, Qwen v4 = 1024, Zhipu = 2048).
 * That is fine — clustering only uses cosine similarity — but tune
 * EMBEDDING_DEDUP_THRESHOLD per provider (see embeddingDedup.ts).
 */
let _client: OpenAI | null = null;

function apiKey(): string {
  return process.env.EMBEDDING_API_KEY || process.env.OPENAI_API_KEY || "";
}

function model(): string {
  return process.env.EMBEDDING_MODEL || "text-embedding-3-small";
}

function client(): OpenAI {
  const key = apiKey();
  if (!key) {
    throw new Error(
      "Embedding API key is not set (set EMBEDDING_API_KEY, or OPENAI_API_KEY for the default OpenAI endpoint)"
    );
  }
  if (!_client) {
    const baseURL = process.env.EMBEDDING_API_URL;
    _client = baseURL ? new OpenAI({ apiKey: key, baseURL }) : new OpenAI({ apiKey: key });
  }
  return _client;
}

/** Batch embedding — a single shared client and one API request for many texts. */
export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const response = await client().embeddings.create({
    model: model(),
    input: texts,
    // Pin float explicitly. Since openai-node v5 the SDK defaults `encoding_format` to
    // "base64" AND then base64-decodes the response unconditionally — so any
    // OpenAI-compatible provider that ignores the flag and returns float arrays
    // (Qwen/DashScope, Zhipu) yields silently garbage vectors with no error.
    encoding_format: "float",
  });
  // OpenAI-compatible responses carry an `index`; sort defensively so the returned
  // vectors always line up with the input order.
  return response.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

export async function getEmbedding(text: string): Promise<number[]> {
  const emb = await getEmbeddings([text]);
  return emb[0];
}
