const CONTRACTIONS = {
  "can't": "can not", cannot: "can not", "won't": "will not", "shan't": "shall not",
  "i'm": "i am", im: "i am", dont: "do not", cant: "can not", wont: "will not",
  didnt: "did not", doesnt: "does not", isnt: "is not", arent: "are not",
  wasnt: "was not", werent: "were not", couldnt: "could not", shouldnt: "should not",
  wouldnt: "would not", ive: "i have", youre: "you are", theyre: "they are",
  thats: "that is", whats: "what is", its: "it is", idk: "i do not know",
};
const SUFFIXES = [["n't", " not"], ["'re", " are"], ["'ve", " have"], ["'ll", " will"], ["'d", " would"], ["'m", " am"], ["'s", ""]];
const KEEP_S = new Set(["is", "was", "has", "this", "us", "yes", "less", "unless", "always", "stress", "process", "mess", "loss", "boss",
  "class", "pass", "kiss", "miss", "yours", "ours", "hers", "his", "its", "thanks", "nervous", "anxious",
  "serious", "jealous", "famous", "useless", "worthless", "hopeless", "helpless", "pointless", "endless", "careless",
  "access", "success", "progress", "business", "illness", "sadness", "loneliness", "happiness", "weakness", "kindness",
  "darkness", "emptiness", "tiredness", "numbness", "as", "does", "bus", "plus", "focus", "virus", "crisis", "basis"]);

export function normalize(text) {
  return text.toLowerCase().replace(/[‘’ʼ`]/g, "'").replace(/[“”]/g, '"');
}

export function lemma(word) {
  if (word.length <= 3 || KEEP_S.has(word) || !word.endsWith("s")) return word;
  if (word.endsWith("ies") && word.length > 4) return word.slice(0, -3) + "y";
  if (["sses", "shes", "ches", "xes"].some((ending) => word.endsWith(ending))) return word.slice(0, -2);
  if (word.endsWith("ss") || word.endsWith("us")) return word;
  return word.slice(0, -1);
}

export function tokenize(text) {
  const words = [];
  for (let raw of normalize(text).match(/[a-z]+(?:'[a-z]+)*/g) ?? []) {
    if (Object.hasOwn(CONTRACTIONS, raw)) { words.push(...CONTRACTIONS[raw].split(" ")); continue; }
    for (const [suffix, replacement] of SUFFIXES) {
      if (raw.endsWith(suffix) && raw.length > suffix.length) { raw = raw.slice(0, -suffix.length) + replacement; break; }
    }
    words.push(...raw.replace(/'/g, "").split(" ").filter(Boolean));
  }
  return words.filter(Boolean).map(lemma);
}
