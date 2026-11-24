import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { tokenize } from "../web/text.js";
import { EmoCareModel, isCrisis } from "../web/bot.js";

const fixture = JSON.parse(readFileSync(new URL("./fixture.json", import.meta.url), "utf8"));
const model = new EmoCareModel(JSON.parse(readFileSync(new URL("../web/model.json", import.meta.url), "utf8")));

test("tokenizer matches training/text.py on every sentence", () => {
  for (const row of fixture) assert.deepEqual(tokenize(row.text), row.tokens, row.text);
});

test("predictions match the Python model", () => {
  for (const row of fixture) {
    const [best] = model.predict(row.text);
    assert.equal(best.tag, row.top, row.text);
    assert.ok(Math.abs(best.p - row.p) < 1e-3, `${row.text}: ${best.p} vs ${row.p}`);
  }
});

test("crisis phrases are always caught", () => {
  for (const text of ["I want to kill myself", "thinking about suicide", "I don't want to live anymore", "everyone would be better off without me",
    "I want to end my life", "i keep self-harming", "I’m going to hurt myself", "I want to die"]) {
    assert.ok(isCrisis(text), text);
    assert.equal(model.respond(text).kind, "crisis", text);
  }
});

test("ordinary sentences are not flagged as crisis", () => {
  for (const text of ["this exam is killing me", "I'm dying to see the movie", "I want to quit my job", "I hate losing at chess"]) {
    assert.equal(isCrisis(text), false, text);
  }
});

test("gibberish gets a gentle 'didn't understand' instead of a random intent", () => {
  assert.equal(model.respond("asdkjh qwpoe zzzz").kind, "unknown");
});
