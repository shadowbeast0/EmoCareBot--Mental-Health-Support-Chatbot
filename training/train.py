
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
from text import tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
intents = json.loads((ROOT / "model" / "intents.json").read_text(encoding="utf-8"))["intents"]
extra = json.loads((ROOT / "model" / "intents_extra.json").read_text(encoding="utf-8"))["patterns"]
for intent in intents:
    intent["patterns"] = intent["patterns"] + [p for p in extra.get(intent["tag"], []) if p not in intent["patterns"]]

samples = [(tokenize(pattern), intent["tag"]) for intent in intents for pattern in intent["patterns"]]
vocab = sorted({word for words, _ in samples for word in words})
tags = sorted({intent["tag"] for intent in intents})
index = {word: i for i, word in enumerate(vocab)}

def bag(words: list[str]) -> np.ndarray:
    vector = np.zeros(len(vocab), dtype=np.float32)
    for word in words:
        if word in index:
            vector[index[word]] = 1.0
    return vector

X = np.stack([bag(words) for words, _ in samples])
y = np.array([tags.index(tag) for _, tag in samples])

def make_model(seed: int = 7) -> MLPClassifier:
    return MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-3, learning_rate_init=4e-3,
                         max_iter=600, random_state=seed)

scores = cross_val_score(make_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1))
print(f"{len(samples)} patterns, {len(vocab)} words, {len(tags)} intents")
print(f"5-fold accuracy on unseen patterns: {scores.mean():.3f} ± {scores.std():.3f}")

model = make_model().fit(X, y)
print(f"training accuracy: {model.score(X, y):.3f}")

HELD_OUT = [
    ("my fiance hasn't spoken to me in days", "relationship-issues"), ("she cheated on me", "relationship-issues"),
    ("he never makes time for me anymore", "relationship-issues"), ("I have three tests tomorrow and I'm panicking", "i am stressed out"),
    ("my workload is crushing me", "i am stressed out"), ("so much homework and no time", "i am stressed out"),
    ("I've felt miserable all week", "i am sad"), ("I feel like crying for no reason", "i am sad"),
    ("I'm completely drained and need to rest", "i want a break"), ("I need to get away from my desk for a bit", "i want a break"),
    ("I bombed the interview", "i will never succeed in life"), ("nothing ever works out for me", "i will never succeed in life"),
    ("my dad is ashamed of my grades", "i let everyone down"), ("I keep disappointing the people I love", "i let everyone down"),
    ("kids at school push me around", "i am a victim of bullying"), ("people keep posting mean comments about me", "i am a victim of bullying"),
    ("if I tell the teacher they'll make it worse", "i am afraid to file a case against bullying"),
    ("I'm scared of what happens if I report them", "i am afraid to file a case against bullying"),
    ("I can't stand the person I've become", "i hate myself!"), ("I'm disgusted with myself", "i hate myself!"),
    ("I want to move abroad and never come back", "i want to leave the country and run away"),
    ("I just want to pack a bag and disappear", "i want to leave the country and run away"),
    ("I give up on everything", "i wish to quit"), ("I don't see the point of trying anymore", "i wish to quit"),
    ("my brain won't shut off at night", "i think i'm losing my mind"), ("I feel like I'm going mad", "i think i'm losing my mind"),
    ("all my friends are doing better than me", "i wish i was better than them"),
    ("we lost the final and it's eating me up", "i hate losing."), ("hey there, how's it going", "greeting"),
    ("thanks, that actually helped", "thanks"),
]
held = model.predict_proba(np.stack([bag(tokenize(text)) for text, _ in HELD_OUT]))
top1 = np.mean([tags[int(p.argmax())] == tag for p, (_, tag) in zip(held, HELD_OUT)])
top2 = np.mean([tag in [tags[i] for i in np.argsort(-p)[:2]] for p, (_, tag) in zip(held, HELD_OUT)])
print(f"held-out everyday sentences: top-1 {top1:.2f}, top-2 {top2:.2f} ({len(HELD_OUT)} sentences)")

responses = {intent["tag"]: intent["responses"] for intent in intents}
export = {
    "version": 2,
    "vocab": vocab,
    "tags": tags,
    "threshold": 0.55,
    "layers": [{"W": np.round(w, 4).tolist(), "b": np.round(b, 4).tolist()} for w, b in zip(model.coefs_, model.intercepts_)],
    "responses": responses,
}
out = ROOT / "web" / "model.json"
out.write_text(json.dumps(export, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

extra = [
    "I'm feeling really low today", "i cant take this anymore", "Nobody at school leaves me alone",
    "my girlfriend keeps ignoring my texts", "I just need some rest", "I failed my exams and my parents are disappointed",
    "Everyone's better than me at everything", "thank u so much!!", "I’m scared they’ll come after me if I report it",
    "I wish I could just disappear to another country", "it’s 3am and my thoughts won’t stop",
]
sentences = [pattern for intent in intents for pattern in intent["patterns"]] + extra
rounded = MLPClassifier(hidden_layer_sizes=(64,))
rounded.coefs_ = [np.round(w, 4) for w in model.coefs_]
rounded.intercepts_ = [np.round(b, 4) for b in model.intercepts_]
rounded.n_layers_, rounded.out_activation_, rounded.activation = model.n_layers_, model.out_activation_, model.activation
rounded.classes_, rounded.n_outputs_ = model.classes_, model.n_outputs_
rounded._label_binarizer = model._label_binarizer
probs = rounded.predict_proba(np.stack([bag(tokenize(s)) for s in sentences]))
fixture = [{"text": s, "tokens": tokenize(s), "top": tags[int(p.argmax())], "p": round(float(p.max()), 5)} for s, p in zip(sentences, probs)]
(ROOT / "tests").mkdir(exist_ok=True)
(ROOT / "tests" / "fixture.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=0), encoding="utf-8")
