import { normalize, tokenize } from "./text.js";

const CRISIS_PATTERNS = [
  /\b(kill|hurt|harm|cut|end)\s+(my\s*self|myself)\b/,
  /\bsuicid/,
  /\bend\s+(my|it)\s+(life|all)\b/,
  /\b(want|wanna|going|plan(ning)?)\s+to\s+die\b/,
  /\bdon'?t\s+want\s+to\s+(live|be\s+alive|exist|wake\s+up)\b/,
  /\bno\s+(reason|point)\s+(to|in)\s+(live|living)\b/,
  /\bbetter\s+off\s+(dead|without\s+me)\b/,
  /\bself[\s-]?harm/,
  /\boverdose\b/,
];

export const TOPIC_LABELS = {
  "gratitude": "Saying thanks",
  "greeting": "Just saying hi",
  "i am a victim of bullying": "Being bullied",
  "i am afraid to file a case against bullying": "Scared to report bullying",
  "i am sad": "Feeling sad",
  "i am stressed out": "Feeling stressed",
  "i hate losing.": "Losing hurts",
  "i hate myself!": "Hating myself",
  "i let everyone down": "Letting people down",
  "i think i'm losing my mind": "Feeling overwhelmed",
  "i want a break": "Needing a break",
  "i want to leave the country and run away": "Wanting to run away",
  "i will never succeed in life": "Feeling like a failure",
  "i wish i could've been a winner": "Wishing I'd won",
  "i wish i was better than them": "Comparing myself to others",
  "i wish to quit": "Wanting to give up",
  "relationship-issues": "Relationship trouble",
  "thanks": "Saying thanks",
};

const SUPPORT_TOPICS = new Set(["i wish to quit", "i hate myself!", "i think i'm losing my mind"]);

export function isCrisis(text) {
  const clean = normalize(text);
  return CRISIS_PATTERNS.some((pattern) => pattern.test(clean));
}

export class EmoCareModel {
  constructor(data) {
    this.data = data;
    this.index = new Map(data.vocab.map((word, i) => [word, i]));
  }

  static async load(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load the model (${response.status}).`);
    return new EmoCareModel(await response.json());
  }

  predict(text) {
    let activations = new Float64Array(this.data.vocab.length);
    for (const word of tokenize(text)) {
      const i = this.index.get(word);
      if (i !== undefined) activations[i] = 1;
    }
    const layers = this.data.layers;
    layers.forEach((layer, depth) => {
      const out = new Float64Array(layer.b.length);
      for (let j = 0; j < out.length; j += 1) out[j] = layer.b[j];
      for (let i = 0; i < activations.length; i += 1) {
        const value = activations[i];
        if (value === 0) continue;
        const row = layer.W[i];
        for (let j = 0; j < out.length; j += 1) out[j] += value * row[j];
      }
      if (depth < layers.length - 1) for (let j = 0; j < out.length; j += 1) out[j] = Math.max(0, out[j]);
      activations = out;
    });
    const max = Math.max(...activations);
    const exps = Array.from(activations, (value) => Math.exp(value - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return this.data.tags.map((tag, i) => ({ tag, p: exps[i] / sum })).sort((a, b) => b.p - a.p);
  }

  knownWords(text) {
    return tokenize(text).filter((word) => this.index.has(word)).length;
  }

  respond(text, rng = Math.random) {
    if (isCrisis(text)) return { kind: "crisis" };
    const ranked = this.predict(text);
    const pick = (tag) => {
      const options = this.data.responses[tag];
      return options[Math.floor(rng() * options.length)];
    };
    if (this.knownWords(text) === 0) return { kind: "unknown" };
    if (ranked[0].p >= this.data.threshold) {
      return { kind: "reply", tag: ranked[0].tag, text: pick(ranked[0].tag), confidence: ranked[0].p, support: SUPPORT_TOPICS.has(ranked[0].tag) };
    }
    const options = ranked.slice(0, 3).filter((item, index, list) =>
      TOPIC_LABELS[item.tag] && list.findIndex((other) => TOPIC_LABELS[other.tag] === TOPIC_LABELS[item.tag]) === index).slice(0, 2);
    return { kind: "clarify", options: options.map((item) => ({ tag: item.tag, label: TOPIC_LABELS[item.tag] })) };
  }

  replyFor(tag, rng = Math.random) {
    const options = this.data.responses[tag];
    return { kind: "reply", tag, text: options[Math.floor(rng() * options.length)], confidence: 1, support: SUPPORT_TOPICS.has(tag) };
  }
}
