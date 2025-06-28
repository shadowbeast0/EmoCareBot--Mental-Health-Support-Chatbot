from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import random
import pickle
from tensorflow.keras.models import load_model
from nltk_utils import tokenize, bag_of_words

app = Flask(__name__)

with open('model/intents.json', 'r') as f:
    intents = json.load(f)

with open('model/chat_vocab.pkl', 'rb') as f:
    all_words = pickle.load(f)

with open('model/chat_labels.pkl', 'rb') as f:
    tags = pickle.load(f)

model = load_model('model/chatbot_model.h5')

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = np.array([X])
    prediction = model.predict(X)[0]
    predicted_index = np.argmax(prediction)
    tag = tags[predicted_index]
    prob = prediction[predicted_index]

    if prob > 0.75:
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm not sure I understand. Can you rephrase?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    user_msg = request.json.get('msg')
    response = get_response(user_msg)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
