<p align="center">
  <img src="web/brain-heart-logo.png" alt="EmoCareBot logo: a cartoon brain hugging a heart" width="120">
</p>

<h1 align="center">EmoCareBot</h1>
<p align="center"><b>An NLP chatbot that listens, recognises what you're going through, and responds with empathy.</b></p>

<p align="center">
  <a href="https://shadowbeast0.github.io/EmoCareBot--Mental-Health-Support-Chatbot/"><img alt="Open the live demo" src="https://img.shields.io/badge/%E2%96%B6%20Live%20demo-open%20EmoCareBot-ffcf8a?style=for-the-badge&labelColor=1b1f3b"></a>
  <a href="https://github.com/shadowbeast0/EmoCareBot--Mental-Health-Support-Chatbot/actions/workflows/pages.yml"><img alt="Tests and deploy" src="https://img.shields.io/github/actions/workflow/status/shadowbeast0/EmoCareBot--Mental-Health-Support-Chatbot/pages.yml?branch=main&style=for-the-badge&label=tests%20%2B%20deploy&labelColor=1b1f3b"></a>
</p>

**Live demo: https://shadowbeast0.github.io/EmoCareBot--Mental-Health-Support-Chatbot/**

> EmoCareBot is a student project, not a therapist, and it can't respond to emergencies.
> If you might hurt yourself, please contact a crisis line: India **Tele-MANAS 14416**, US **988**,
> UK & Ireland **Samaritans 116 123**, or find one anywhere at [findahelpline.com](https://findahelpline.com).

---

## What it does

- Understands the *meaning* of what you type, not just keywords, using a sentence-embedding model (all-MiniLM-L6-v2) running in your browser. It recognises **18 emotional intents**: sadness, stress, relationship trouble, bullying, feeling like a failure, wanting to run away, and more.
- Replies with a supportive message and, every other turn, a gentle follow-up question so it feels like a conversation. It also handles small talk ("ok", "what's your name", "bye") naturally.
- **Crisis safety net:** messages about suicide or self-harm are caught by dedicated rules *before* the model runs, and always answered with helpline numbers. Heavier topics ("I want to give up", "I hate myself") show support lines alongside the reply.
- **Asks instead of guessing:** when the model isn't confident, it offers the two most likely topics as buttons instead of a random reply.
- **Private by design:** the model runs entirely in your browser. Nothing you type is sent anywhere, and the conversation is cleared when you close the tab.

## How it works

```
your message ─┬─► crisis rules ───────────────────────────────► helplines (always wins)
              ├─► small talk ("ok", "who are you", "bye") ─────► conversational reply
              └─► sentence embedding (MiniLM, in the browser)
                    └─► closest of ~400 example sentences per intent
                          ├─ clear match ─────► supportive reply + follow-up question
                          ├─ close call ──────► "Is it closer to A or B?"
                          └─ no match ────────► "Tell me a little more…"
```

The embedding model (~23 MB) downloads once and is cached by the browser. Until it's ready, a small keyword model (421 → 64 → 18 network, under 1 MB) answers instantly.

| Piece | File |
|---|---|
| Conversation engine (crisis → small talk → semantic → keyword fallback) | [`web/engine.js`](web/engine.js) |
| Semantic matcher + example-sentence embeddings | [`web/semantic.js`](web/semantic.js), [`training/embed.mjs`](training/embed.mjs) → `web/embeddings.json` |
| Training data (Kaggle dataset + extra everyday phrasings) | [`model/intents.json`](model/intents.json), [`model/intents_extra.json`](model/intents_extra.json) |
| Keyword fallback model (scikit-learn MLP → JSON weights) | [`training/train.py`](training/train.py) |
| Tokenizer, identical in Python and JavaScript | [`training/text.py`](training/text.py) ↔ [`web/text.js`](web/text.js) |
| In-browser model + crisis rules | [`web/bot.js`](web/bot.js) |
| Chat interface | [`web/index.html`](web/index.html), [`web/app.js`](web/app.js), [`web/style.css`](web/style.css) |
| Tests: browser matches Python, crisis phrases caught, semantic accuracy ≥ 85% on unseen sentences | [`tests/`](tests) |
| Original Keras training notebook | [`EmoCareBot-training.ipynb`](EmoCareBot-training.ipynb) |

### Accuracy (honest numbers)

| Test | Semantic engine | Keyword fallback |
|---|---|---|
| 30 held-out everyday sentences, right intent first | **90%** | 57% |
| Same sentences, right intent in the top two | **97%** | 77% |
| Dataset's own patterns (leave-one-out / 5-fold) | **80%** | 66% |

Close calls are never stated as fact: the bot asks which of the two topics you mean. To teach it new phrasings, add them to `model/intents_extra.json` and run `npm run embed` (and `python training/train.py` for the fallback).

## Dataset

Built from the Kaggle [Intent-based Mental Health Chatbot Data](https://www.kaggle.com/datasets/rishabhpancholi1302/intent-based-mental-health-chatbot-data), with customisations, plus extra everyday phrasings (girlfriend/boyfriend, exams, deadlines, work) kept separately in `model/intents_extra.json`.

## Run it locally

```bash
pip install -r requirements.txt
python app.py              # open http://127.0.0.1:5000
```

Rebuild and re-test after changing the data:

```bash
npm ci
npm run embed              # sentence embeddings → web/embeddings.json (+ accuracy report)
pip install -r training/requirements.txt
python training/train.py   # keyword fallback → web/model.json, tests/fixture.json
npm test
```

Every push to `main` runs the tests and deploys `web/` to GitHub Pages.

## What changed in this version

- **Runs in the browser:** the model runs client-side and is hosted free on GitHub Pages. There's no Flask + TensorFlow server to keep alive.
- **Crisis detection and helplines:** added, and they always take priority over the model.
- **Tokenizer:** the old pipeline used NLTK at training time but different text cleaning at reply time. One shared tokenizer now runs in both places.
- **Replies keep their punctuation:** the old server stripped every non-ASCII character, so "It’s okay" was shown as "Its okay".
- **Much better understanding:** a sentence-embedding model now matches meaning instead of keywords. Accuracy on unseen everyday sentences rose from 57% to 90%.
- **Conversation:** follow-up questions, small-talk handling, no repeated replies, and a clarifying question instead of a wrong guess.
- **Removed outdated dependencies:** TensorFlow 2.18, legacy `.h5` loading and NLTK runtime downloads, which also broke on newer NLTK versions that need `punkt_tab`.
- **New interface:** calm, mobile-friendly, keyboard-accessible, with starter prompts and a typing indicator.
