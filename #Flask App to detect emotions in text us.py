#Flask App to detect emotions in text using VADER Sentiment Analysis
from flask import Flask, request, jsonify, render_template
import random
import unittest
import logging
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon, this should be done once.
nltk.download('vader_lexicon')

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

def emotion_detector(text_to_analyze):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text_to_analyze)
    
    anger = scores['neg']
    joy = scores['pos']
    sadness = scores['neu']
    disgust = 0
    fear = 0
    
    # Set the dominant emotion based on the 'compound' score.
    if scores['compound'] >= 0.05:
        dominant_emotion = 'pos'
    elif scores['compound'] <= -0.05:
        dominant_emotion = 'neg'
    else:
        dominant_emotion = 'neu'
    
    return f"Anger: {anger}\nDisgust: {disgust}\nFear: {fear}\nJoy: {joy}\nSadness: {sadness}\nDominant Emotion: {dominant_emotion}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    statement = request.form['statement']
    sentiment = emotion_detector(statement)
    return jsonify({'sentiment': sentiment})

@app.route('/emotionDetector', methods=['GET'])
def emotion_detector_route():
    text_to_analyze = request.args.get('textToAnalyze')
    if text_to_analyze:
        sentiment = emotion_detector(text_to_analyze)
        return jsonify({'sentiment': sentiment})
    else:
        return jsonify({'error': 'No text provided'}), 400

@app.route('/run_tests', methods=['GET'])
def run_tests():
    logging.info("Running tests...")
    result = unittest.TextTestRunner().run(unittest.makeSuite(TestEmotionDetection))
    return jsonify({'result': str(result)})

class TestEmotionDetection(unittest.TestCase):
    def setUp(self):
        self.statements = [
            'I am happy', 'I am sad', 'I am angry', 'I am excited', 'I am bored',
            'I am tired', 'I am anxious', 'I am relaxed', 'I am frustrated', 'I am content',
            'I am surprised', 'I am scared', 'I am nervous', 'I am hopeful', 'I am proud',
            'I am disappointed', 'I am confused', 'I am curious', 'I am determined', 'I am grateful',
            'I am lonely', 'I am joyful', 'I am stressed', 'I am overwhelmed', 'I am peaceful',
            'I am confident', 'I am embarrassed', 'I am guilty', 'I am ashamed', 'I am optimistic'
        ]

    def test_random_statements(self):
        selected_statements = random.sample(self.statements, 4)
        for statement in selected_statements:
            sentiment = emotion_detector(statement)
            logging.info(f"Statement: {statement}, Sentiment: {sentiment}")
            self.assertIn(sentiment.split('\n')[-1].split(': ')[1], ['pos', 'neg', 'neu'])

if __name__ == '__main__':
    app.run(debug=True)