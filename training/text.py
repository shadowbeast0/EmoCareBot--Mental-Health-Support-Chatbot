
import re

CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "shan't": "shall not",
    "i'm": "i am", "im": "i am", "dont": "do not", "cant": "can not", "wont": "will not",
    "didnt": "did not", "doesnt": "does not", "isnt": "is not", "arent": "are not",
    "wasnt": "was not", "werent": "were not", "couldnt": "could not", "shouldnt": "should not",
    "wouldnt": "would not", "ive": "i have", "youre": "you are", "theyre": "they are",
    "thats": "that is", "whats": "what is", "its": "it is", "idk": "i do not know",
}
SUFFIXES = (("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would"), ("'m", " am"), ("'s", ""))
KEEP_S = {"is", "was", "has", "this", "us", "yes", "less", "unless", "always", "stress", "process", "mess", "loss", "boss",
          "class", "pass", "kiss", "miss", "yours", "ours", "hers", "his", "its", "thanks", "always", "nervous", "anxious",
          "serious", "jealous", "famous", "useless", "worthless", "hopeless", "helpless", "pointless", "endless", "careless",
          "access", "success", "progress", "business", "illness", "sadness", "loneliness", "happiness", "weakness", "kindness",
          "darkness", "emptiness", "tiredness", "numbness", "as", "does", "bus", "plus", "focus", "virus", "crisis", "basis"}

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[‘’ʼ`]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    return text

def lemma(word: str) -> str:
    if len(word) <= 3 or word in KEEP_S or not word.endswith("s"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if word.endswith("ss") or word.endswith("us"):
        return word
    return word[:-1]

def tokenize(text: str) -> list[str]:
    words = []
    for raw in re.findall(r"[a-z]+(?:'[a-z]+)*", normalize(text)):
        if raw in CONTRACTIONS:
            words.extend(CONTRACTIONS[raw].split())
            continue
        for suffix, replacement in SUFFIXES:
            if raw.endswith(suffix) and len(raw) > len(suffix):
                raw = raw[: -len(suffix)] + replacement
                break
        words.extend(raw.replace("'", "").split())
    return [lemma(word) for word in words if word]
