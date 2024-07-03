import joblib
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')

model_path = os.getcwd() + r'/models/model'
classifier = joblib.load(model_path + r'/classifier.pkl')

def getPrediction(review):
    prediction = classifier.predict(review)

    if prediction[0] == 1:
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return prediction, sentiment


@app.route("/")
def home():
    return "Hello from Flask"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")

    review = pd.Series([review])
    prediction, sentiment = getPrediction(review)
    
    return jsonify({"sentiment": sentiment, "prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, port=8080)