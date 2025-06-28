from flask import Flask, render_template, request, jsonify
import unicodedata
import numpy as np
import json
import random
import pickle
from tensorflow.keras.models import load_model
from nltk_utils import tokenize, bag_of_words
import nltk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

def normalize_text(text: str) -> str:
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False            # ensure UTF‑8 in all JSON responses

with open('model/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

with open('model/chat_vocab.pkl', 'rb') as f:
    all_words = pickle.load(f)

with open('model/chat_labels.pkl', 'rb') as f:
    tags = pickle.load(f)

model = load_model('model/chatbot_model.h5')

def get_response(msg: str) -> str:
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = np.array([X])

    try:
        prediction = model.predict(X, verbose=0)[0]
    except Exception:
        return "Sorry, something went wrong internally."

    idx   = np.argmax(prediction)
    tag   = tags[idx]
    prob  = prediction[idx]

    if prob > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return normalize_text(random.choice(intent['responses']))

    return "I'm not sure I understand. Can you rephrase?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chatbot_response():
    data = request.get_json()

    if not data or not data.get('message', '').strip():
        return jsonify({'reply': "Please say something."})

    user_msg = data['message']
    reply    = get_response(user_msg)
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)
