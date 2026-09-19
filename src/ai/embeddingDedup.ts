/**
 * Embedding-based deduplication — replaces the (very expensive) multi-round gpt-4.1
 * reasoning-high LLM merge with cheap `text-embedding-3-small` vectors + cosine clustering.
 *
 * For each raw post we embed once, cluster by cosine similarity, then within each cluster
 * keep the most comprehensive post as the representative and union all source URLs. This
 * yields the same `SimplifiedNewsItem[]` shape the downstream Summarize stage expects, at a
 * tiny fraction of the token cost. Any failure propagates to the caller, which falls back to
 * the original LLM dedup path.
 */
import { getEmbeddings } from "./embedding";
import { SimplifiedNewsItem } from "../types";

const URL_RE = /https?:\/\/[^\s)\]<>]+/gi;

/** Pull every URL out of a post's text (used to populate SimplifiedNewsItem.sources). */
export function extractSources(text: string): string[] {
  const matches = text.match(URL_RE) ?? [];
  const cleaned = matches.map((m) => m.replace(/[.,;:]+$/, ""));
  return Array.from(new Set(cleaned));
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

interface Cluster {
  repIndex: number; // index of the (longest) representative in `items`
  members: number[];
  vec: number[]; // representative vector used for future comparisons
}

export async function embeddingDeduplicate(
  items: string[]
): Promise<SimplifiedNewsItem[]> {
  if (items.length === 0) return [];

  const threshold = Number(process.env.EMBEDDING_DEDUP_THRESHOLD ?? 0.85);
  // Throws if OPENAI_API_KEY is missing → caller falls back to LLM dedup.
  const embeddings = await getEmbeddings(items);

  const clusters: Cluster[] = [];
  for (let i = 0; i < items.length; i++) {
    let best = -1;
    let bestSim = -1;
    for (let c = 0; c < clusters.length; c++) {
      const sim = cosine(embeddings[i], clusters[c].vec);
      if (sim > bestSim) {
        bestSim = sim;
        best = c;
      }
    }
    if (best >= 0 && bestSim >= threshold) {
      clusters[best].members.push(i);
      // Promote the longer post to representative so the final text is the most comprehensive.
      if (items[i].length > items[clusters[best].repIndex].length) {
        clusters[best].repIndex = i;
        clusters[best].vec = embeddings[i];
      }
    } else {
      clusters.push({ repIndex: i, members: [i], vec: embeddings[i] });
    }
  }

  return clusters.map((cl): SimplifiedNewsItem => {
    const rep = items[cl.repIndex];
    const sources = Array.from(
      new Set(cl.members.flatMap((m) => extractSources(items[m])))
    );
    return {
      text: rep,
      sources: sources.length ? sources : ["unknown-source"],
    };
  });
}
