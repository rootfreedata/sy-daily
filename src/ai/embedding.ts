import OpenAI from "openai";

/**
 * Generates an embedding vector for the given text using the configured AI provider.
 * Supports both Arabic and English text.
 *
 * @param text - The text to generate an embedding for
 * @returns A promise that resolves to the embedding vector (array of numbers)
 */
let _client: OpenAI | null = null;
function client(): OpenAI {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  if (!_client) {
    _client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  return _client;
}

/** Batch embedding — a single shared client and one API request for many texts. */
export async function getEmbeddings(texts: string[]): Promise<number[][]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  if (texts.length === 0) return [];
  const response = await client().embeddings.create({
    model: "text-embedding-3-small",
    input: texts,
  });
  // OpenAI returns embeddings in the same order as the input (index field is 0..n-1).
  return response.data
    .slice()
    .sort((a, b) => a.index - b.index)
    .map((d) => d.embedding);
}

export async function getEmbedding(text: string): Promise<number[]> {
  const emb = await getEmbeddings([text]);
  return emb[0];
}
