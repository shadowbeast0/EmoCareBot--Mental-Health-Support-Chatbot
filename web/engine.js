import { EmoCareModel, TOPIC_LABELS, isCrisis } from "./bot.js";
import { classify, decodeVectors } from "./semantic.js";
import { normalize } from "./text.js";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1";
const REPLY_SCORE = 0.5;
const REPLY_MARGIN = 0.05;
const CLARIFY_SCORE = 0.4;

const SUPPORT_TOPICS = new Set(["i wish to quit", "i hate myself!", "i think i'm losing my mind"]);

const FOLLOW_UPS = {
  "i am sad": ["Do you want to tell me what's been weighing on you?", "Has something happened recently, or has it been building up?"],
  "i am stressed out": ["What's the part that feels heaviest right now?", "Is there one thing on your list we could make smaller together?"],
  "relationship-issues": ["How long has it been feeling like this between you two?", "What do you wish they understood about how you feel?"],
  "i want a break": ["What would a real break look like for you, even a short one?"],
  "i will never succeed in life": ["What happened that made you feel this way?"],
  "i let everyone down": ["Who do you feel you've let down? Sometimes saying it out loud helps."],
  "i am a victim of bullying": ["Is there an adult or someone you trust who knows this is happening?"],
  "i am afraid to file a case against bullying": ["Would it help to think through who could support you if you reported it?"],
  "i hate myself!": ["What's making you feel this way about yourself today?"],
  "i wish to quit": ["What's making things feel this heavy right now?"],
  "i think i'm losing my mind": ["When did things start feeling this overwhelming?"],
  "i want to leave the country and run away": ["What would you most want to get away from?"],
  "i wish i was better than them": ["Who are you comparing yourself to lately?"],
  "i hate losing.": ["What did you lose? It's okay to be upset about it."],
  "i wish i could've been a winner": ["What were you hoping would happen?"],
};

const SMALL_TALK = [
  { test: /^(ok(ay)?|k|sure|alright|fine|yeah|yes|yep|no|nope|nah|hmm+|mm+|idk|i don'?t know|maybe)[.!?]*$/, replies: [
    "I'm listening. Take your time. What's been on your mind?",
    "That's okay. Is there something you'd like to talk about?",
    "No pressure. You can tell me as much or as little as you want.",
  ] },
  { test: /\b(your name|who are you|what are you)\b/, replies: ["I'm EmoCareBot, a small chatbot that listens. I'm not a person or a therapist, but I'm here to hear you out. How are you feeling?"] },
  { test: /\b(joke|funny)\b/, replies: ["I'm better at listening than at jokes. How's your day going, honestly?"] },
  { test: /\b(bye|goodbye|see you|good night|gn)\b/, replies: ["Take care of yourself. I'm here whenever you want to talk again.", "Goodbye for now. Be gentle with yourself today."] },
];

export class Engine {
  constructor() {
    this.keyword = null;
    this.semantic = null;
    this.lastResponse = new Map();
    this.turns = 0;
    this.status = "loading";
  }

  async loadKeywordModel(url) {
    this.keyword = await EmoCareModel.load(url);
    this.status = "keyword";
  }

  async loadSemanticModel(embeddingsUrl, onProgress) {
    const [{ pipeline }, data] = await Promise.all([
      import(TRANSFORMERS_URL),
      fetch(embeddingsUrl).then((response) => { if (!response.ok) throw new Error("embeddings"); return response.json(); }),
    ]);
    const embed = await pipeline("feature-extraction", data.model, { dtype: "q8", progress_callback: onProgress });
    this.semantic = { embed, index: { tags: data.tags, vectors: decodeVectors(data.vectors, data.dim) } };
    this.status = "semantic";
  }

  pick(tag, options) {
    const last = this.lastResponse.get(tag);
    const pool = options.length > 1 ? options.filter((option) => option !== last) : options;
    const choice = pool[Math.floor(Math.random() * pool.length)];
    this.lastResponse.set(tag, choice);
    return choice;
  }

  replyFor(tag) {
    this.turns += 1;
    const text = this.pick(tag, this.keyword.data.responses[tag]);
    const followUps = FOLLOW_UPS[tag];
    const followUp = followUps && this.turns % 2 === 1 ? this.pick(`${tag}:follow`, followUps) : null;
    return { kind: "reply", tag, text: followUp ? `${text} ${followUp}` : text, support: SUPPORT_TOPICS.has(tag) };
  }

  async respond(message) {
    if (isCrisis(message)) return { kind: "crisis" };
    const clean = normalize(message).trim();
    for (const rule of SMALL_TALK) {
      if (rule.test.test(clean)) return { kind: "reply", tag: "small-talk", text: this.pick(rule.test.source, rule.replies) };
    }

    if (this.semantic) {
      const output = await this.semantic.embed(message, { pooling: "mean", normalize: true });
      const ranked = classify(this.semantic.index, output.data);
      const [best, second] = ranked;
      if (best.score >= REPLY_SCORE && best.score - second.score >= REPLY_MARGIN) return this.replyFor(best.tag);
      if (best.score >= CLARIFY_SCORE) return clarify(ranked.map((item) => item.tag));
      return { kind: "unknown" };
    }

    const result = this.keyword.respond(message);
    if (result.kind === "reply") return this.replyFor(result.tag);
    return result;
  }
}

function clarify(tags) {
  const seen = new Set();
  const options = [];
  for (const tag of tags) {
    const label = TOPIC_LABELS[tag];
    if (!label || seen.has(label)) continue;
    seen.add(label);
    options.push({ tag, label });
    if (options.length === 2) break;
  }
  return { kind: "clarify", options };
}
