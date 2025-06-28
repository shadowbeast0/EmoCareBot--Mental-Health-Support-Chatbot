# EmoCareBot – AI-Powered Mental Health Support Chatbot

A conversational AI that listens, understands, and responds with empathy.

---

## Overview

**EmoCareBot** is an NLP-powered mental health chatbot designed to engage with users in moments of emotional distress — offering comfort, support, and a safe space to express their feelings.

Using a custom-trained intent recognition model built on a labeled dataset of mental health expressions, the chatbot identifies what the user is going through — from sadness and anxiety to relationship issues and feelings of worthlessness — and responds with relevant, empathetic messages.

---

## What It Does

- Uses a **deep learning classifier** to recognize **user intent** from emotional language.
- Matches input text to intents like:
  - `"i am sad"`
  - `"i want to quit"`
  - `"i lost someone"`
  - `"i'm being bullied"`
- Returns supportive responses aimed at **comforting the user**, not solving their problems directly.
- Designed for web integration with a responsive chatbot interface and a symbolic logo: a **cartoon brain hugging a heart** 🧠❤️.

---

## Intent Recognition Model

The chatbot's intelligence is based on **intent classification**, where emotional user messages are mapped to predefined **intent tags** like:

| Intent Tag                  | Example Message                             |
|----------------------------|---------------------------------------------|
| `i am sad`                 | "I'm feeling really low today"              |
| `relationship-issues`      | "I broke up and can't move on"              |
| `i hate myself`            | "I don’t think I deserve to be here"        |
| `gratitude`                | "Thank you for always listening"            |
| `i want to quit`           | "I want to disappear"                       |
| `i am a victim of bullying`| "They harass me every day at school"        |

Each tag is associated with varied **patterns** and **empathetic response templates**.

---

## Dataset

The training dataset contains a curated set of user phrases and response mappings focused on **mental health expressions**. It is made by a combination of a kaggle dataset whose link is attached below along with some customisations. The entire dataset is present in model/intents.json

🔗 **Dataset Link:** https://www.kaggle.com/datasets/rishabhpancholi1302/intent-based-mental-health-chatbot-data

---

## How to Run

1. **Clone the repo**
2. **Type** ```pip install -r requirements.txt``` **in terminal**
3. **Type** ```python app.py``` **in terminal and follow the steps to the website**
4. **Click the button on the right side and type your prompts**
