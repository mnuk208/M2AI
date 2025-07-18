from flask import Flask, render_template, request, jsonify
import math
import nltk
import torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

nltk.download('punkt')

# Load language model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('input_text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = ai_detection_analysis(text)
    return jsonify(result)

def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return math.exp(loss)

def measure_burstiness(text):
    sentences = nltk.sent_tokenize(text)
    lengths = [len(sentence.split()) for sentence in sentences]
    return round(np.std(lengths) / np.mean(lengths), 2) if lengths else 0

def measure_entropy(text):
    text = text.replace(" ", "")
    freq = Counter(text)
    total = sum(freq.values())
    entropy = -sum((f / total) * math.log2(f / total) for f in freq.values())
    return round(entropy, 2)

def ai_detection_analysis(text):
    result = {}
    try:
        result['perplexity'] = round(calculate_perplexity(text), 2)
    except:
        result['perplexity'] = None

    result['burstiness'] = measure_burstiness(text)
    result['entropy'] = measure_entropy(text)

    score = 0
    checks = 0

    if result['perplexity'] is not None:
        checks += 1
        if result['perplexity'] < 50:
            score += 1

    checks += 1
    if result['burstiness'] < 0.5:
        score += 1

    checks += 1
    if 3.5 <= result['entropy'] <= 4.5:
        score += 1

    result['ai_likelihood_percent'] = round((score / checks) * 100, 1) if checks > 0 else 0
    return result

if __name__ == '__main__':
    app.run(debug=True)
