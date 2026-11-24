import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { env, pipeline } from "@huggingface/transformers";
import { HELD_OUT } from "../training/held_out.mjs";
import { classify, decodeVectors } from "../web/semantic.js";

if (process.env.HF_CACHE) env.cacheDir = process.env.HF_CACHE;
const data = JSON.parse(readFileSync(new URL("../web/embeddings.json", import.meta.url), "utf8"));
const index = { tags: data.tags, vectors: decodeVectors(data.vectors, data.dim) };
const embed = await pipeline("feature-extraction", data.model, { dtype: "q8" });
const rank = async (text) => classify(index, (await embed(text, { pooling: "mean", normalize: true })).data);

test("held-out everyday sentences: right intent first for at least 85%", async () => {
  let hits = 0;
  for (const [text, tag] of HELD_OUT) if ((await rank(text))[0].tag === tag) hits += 1;
  assert.ok(hits / HELD_OUT.length >= 0.85, `top-1 accuracy ${hits}/${HELD_OUT.length}`);
});

test("off-topic text scores below the reply threshold", async () => {
  for (const text of ["asdkjh qwpoe zzzz", "how do I install python", "recommend a movie"]) {
    assert.ok((await rank(text))[0].score < 0.5, text);
  }
});
