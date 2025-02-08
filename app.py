from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Inisialisasi model NLP
nlp_model = pipeline('sentiment-analysis')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    result = nlp_model(text)[0]
    return jsonify({
        'sentiment': result['label'],
        'confidence': result['score']
    })

    # Di app.py
text_generator = pipeline('text-generation', model='gpt2')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    generated_text = text_generator(prompt, max_length=50)[0]['generated_text']
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)