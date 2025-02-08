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

if __name__ == '__main__':
    app.run(debug=True)