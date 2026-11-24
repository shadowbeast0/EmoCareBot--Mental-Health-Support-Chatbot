import { readFileSync, writeFileSync } from "node:fs";
import { env, pipeline } from "@huggingface/transformers";

if (process.env.HF_CACHE) env.cacheDir = process.env.HF_CACHE;
import { HELD_OUT } from "./held_out.mjs";
import { classify, decodeVectors, encodeVectors } from "../web/semantic.js";

export const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const root = new URL("../", import.meta.url);
const intents = JSON.parse(readFileSync(new URL("model/intents.json", root), "utf8")).intents;
const extra = JSON.parse(readFileSync(new URL("model/intents_extra.json", root), "utf8")).patterns;

const patterns = [];
for (const intent of intents) {
  const all = [...intent.patterns, ...(extra[intent.tag] ?? []).filter((p) => !intent.patterns.includes(p))];
  for (const text of all) patterns.push({ text, tag: intent.tag });
}

const embed = await pipeline("feature-extraction", MODEL_ID, { dtype: "q8" });
async function vectors(texts) {
  const out = [];
  for (let i = 0; i < texts.length; i += 64) {
    const batch = await embed(texts.slice(i, i + 64), { pooling: "mean", normalize: true });
    out.push(...batch.tolist().map((row) => Float32Array.from(row)));
  }
  return out;
}

const patternVectors = await vectors(patterns.map((p) => p.text));
const index = { tags: patterns.map((p) => p.tag), vectors: decodeVectors(encodeVectors(patternVectors), patternVectors[0].length) };

let loo = 0;
patterns.forEach((pattern, i) => {
  const masked = { tags: index.tags.filter((_, j) => j !== i), vectors: index.vectors.filter((_, j) => j !== i) };
  if (classify(masked, patternVectors[i])[0].tag === pattern.tag) loo += 1;
});

const heldVectors = await vectors(HELD_OUT.map(([text]) => text));
let top1 = 0;
let top2 = 0;
const misses = [];
HELD_OUT.forEach(([text, tag], i) => {
  const ranked = classify(index, heldVectors[i]);
  if (ranked[0].tag === tag) top1 += 1;
  else misses.push(`${text}  →  ${ranked[0].tag} (${ranked[0].score.toFixed(2)}), wanted ${tag}`);
  if (ranked.slice(0, 2).some((r) => r.tag === tag)) top2 += 1;
});

console.log(`${patterns.length} patterns embedded with ${MODEL_ID}`);
console.log(`leave-one-out accuracy: ${(loo / patterns.length).toFixed(3)}`);
console.log(`held-out everyday sentences: top-1 ${(top1 / HELD_OUT.length).toFixed(2)}, top-2 ${(top2 / HELD_OUT.length).toFixed(2)}`);
if (misses.length) console.log("misses:\n  " + misses.join("\n  "));

writeFileSync(new URL("web/embeddings.json", root), JSON.stringify({
  model: MODEL_ID,
  dim: patternVectors[0].length,
  tags: index.tags,
  vectors: encodeVectors(patternVectors),
}));
