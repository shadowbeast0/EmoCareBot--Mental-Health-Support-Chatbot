import { Engine } from "./engine.js";

const messages = document.getElementById("messages");
const chips = document.getElementById("chips");
const form = document.getElementById("composer");
const input = document.getElementById("input_message");
const sendButton = document.getElementById("send_button");
const help = document.getElementById("help");
const STORE_KEY = "emocarebot:conversation";

const STARTERS = ["I feel really low today", "I'm stressed about exams", "Things with my partner are hard", "I just need a break"];

const safeStorage = {
  get() { try { return JSON.parse(sessionStorage.getItem(STORE_KEY) || "[]"); } catch { return []; } },
  set(value) { try { sessionStorage.setItem(STORE_KEY, JSON.stringify(value)); } catch { /* private mode */ } },
};
let history = safeStorage.get();
const engine = new Engine();
const statusLine = document.getElementById("model_status");
let ready = false;

function addMessage(entry, save = true) {
  const item = document.createElement("li");
  item.className = `msg ${entry.from}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = entry.text;
  item.append(bubble);

  if (entry.crisis || entry.support) {
    const card = document.createElement("div");
    card.className = entry.crisis ? "crisis-card" : "support-card";
    card.innerHTML = `
      <strong>${entry.crisis ? "You don't have to go through this alone." : "If it ever feels like too much:"}</strong>
      <a href="tel:14416">India · Tele-MANAS 14416</a>
      <a href="tel:988">US · 988</a>
      <a href="tel:116123">UK &amp; Ireland · 116 123</a>
      <a href="https://findahelpline.com" target="_blank" rel="noopener">Find a line in your country</a>`;
    item.append(card);
  }
  if (entry.options?.length) {
    const row = document.createElement("div");
    row.className = "options";
    for (const option of entry.options) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = option.label;
      button.addEventListener("click", () => {
        row.remove();
        entry.options = [];
        say({ from: "user", text: option.label });
        botSays(Promise.resolve(engine.replyFor(option.tag)));
      });
      row.append(button);
    }
    item.append(row);
  }
  messages.append(item);
  item.scrollIntoView({ behavior: "smooth", block: "end" });
  if (save) { history.push(entry); safeStorage.set(history.slice(-60)); }
}

const say = (entry) => addMessage(entry);

async function botSays(pending) {
  const typing = document.createElement("li");
  typing.className = "msg bot typing";
  typing.innerHTML = '<div class="bubble"><span></span><span></span><span></span></div>';
  messages.append(typing);
  typing.scrollIntoView({ behavior: "smooth", block: "end" });
  const started = performance.now();
  let result;
  try { result = await pending; } catch { result = { kind: "unknown" }; }
  const delay = Math.max(0, 450 + Math.min(900, (result.text?.length ?? 60) * 8) - (performance.now() - started));
  setTimeout(() => {
    typing.remove();
    if (result.kind === "crisis") {
      say({ from: "bot", crisis: true, text: "I'm really glad you told me. What you're feeling matters, and you deserve support from a real person right now. Please reach out to one of these lines, or to someone you trust nearby. If you're in immediate danger, call your local emergency number." });
    } else if (result.kind === "unknown") {
      say({ from: "bot", text: "I didn't quite catch that. Could you tell me a little more about how you're feeling?" });
    } else if (result.kind === "clarify") {
      say({ from: "bot", text: "I want to make sure I understand. Is it closer to one of these?", options: result.options });
    } else {
      say({ from: "bot", text: result.text, support: result.support });
    }
  }, delay);
}

function send(text) {
  const message = text.trim();
  if (!message || !ready) return;
  chips.hidden = true;
  say({ from: "user", text: message });
  botSays(engine.respond(message));
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  send(input.value);
  input.value = "";
  input.style.height = "";
  input.focus();
});
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});
input.addEventListener("input", () => {
  input.style.height = "";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});
document.getElementById("help_link").addEventListener("click", (event) => { event.preventDefault(); help.showModal(); });
document.getElementById("clear_button").addEventListener("click", () => {
  history = [];
  safeStorage.set(history);
  messages.replaceChildren();
  greet();
});

function greet() {
  chips.hidden = false;
  chips.replaceChildren(...STARTERS.map((text) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => send(text));
    return button;
  }));
  say({ from: "bot", text: "Hi, I'm EmoCareBot. I'm here to listen. How are you feeling today?" });
}

async function start() {
  sendButton.disabled = true;
  try {
    await engine.loadKeywordModel(new URL("model.json", import.meta.url));
    ready = true;
    sendButton.disabled = false;
  } catch (error) {
    say({ from: "bot", text: `${error.message} Please refresh the page.` }, false);
    return;
  }
  statusLine.textContent = "Loading better understanding…";
  let lastShown = 0;
  engine.loadSemanticModel(new URL("embeddings.json", import.meta.url), (event) => {
    if (event.status === "progress" && event.file?.endsWith(".onnx") && event.progress - lastShown >= 5) {
      lastShown = event.progress;
      statusLine.textContent = `Loading better understanding… ${Math.round(event.progress)}%`;
    }
  }).then(() => {
    statusLine.textContent = "";
  }).catch(() => {
    statusLine.textContent = "Using basic understanding (couldn't download the full model).";
  });

  if (history.length) {
    chips.hidden = true;
    history.forEach((entry) => addMessage(entry, false));
  } else if (!messages.children.length) {
    greet();
  }
}

start();
