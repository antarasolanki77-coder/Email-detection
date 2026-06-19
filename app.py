"""
Email Spam Detection -- Flask Web Application
Beautiful UI/UX with real-time predictions, dashboard, and analytics.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from preprocess import clean_text

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Load Model & Vectorizer ---
ROOT = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(ROOT, "models", "spam_model.pkl"))
vectorizer = joblib.load(os.path.join(ROOT, "models", "tfidf.pkl"))

# --- Load Metrics ---
def load_metrics():
    report_path = os.path.join(ROOT, "results", "accuracy_report.txt")
    metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Accuracy:"):
                    metrics["accuracy"] = float(line.split()[1])
                elif line.startswith("Precision:"):
                    metrics["precision"] = float(line.split()[1])
                elif line.startswith("Recall:"):
                    metrics["recall"] = float(line.split()[1])
                elif line.startswith("F1 Score:"):
                    metrics["f1"] = float(line.split()[2])
    return metrics

# --- Load Dataset Stats ---
def load_dataset_stats():
    csv_path = os.path.join(ROOT, "dataset", "spam.csv")
    stats = {"total": 0, "spam": 0, "ham": 0}
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        stats["total"] = len(df)
        stats["spam"] = int((df["label"] == "spam").sum())
        stats["ham"] = int((df["label"] == "ham").sum())
    return stats

METRICS = load_metrics()
DATASET_STATS = load_dataset_stats()

# --- Prediction history (in-memory) ---
prediction_history = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter email text."}), 400

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    spam_prob = round(float(probabilities[1]) * 100, 2)
    ham_prob = round(float(probabilities[0]) * 100, 2)
    label = "Spam" if prediction == 1 else "Not Spam"

    # Get top contributing words
    feature_names = vectorizer.get_feature_names_out()
    feature_array = features.toarray()[0]
    top_indices = feature_array.argsort()[-5:][::-1]
    top_words = [
        {"word": feature_names[i], "score": round(float(feature_array[i]), 3)}
        for i in top_indices if feature_array[i] > 0
    ]

    result = {
        "prediction": label,
        "is_spam": bool(prediction == 1),
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "cleaned_text": cleaned,
        "top_words": top_words,
        "original_length": len(text),
        "cleaned_length": len(cleaned),
    }

    # Save to history
    prediction_history.insert(0, {
        "text": text[:80] + ("..." if len(text) > 80 else ""),
        "prediction": label,
        "confidence": spam_prob if prediction == 1 else ham_prob,
        "is_spam": bool(prediction == 1),
    })
    if len(prediction_history) > 20:
        prediction_history.pop()

    return jsonify(result)


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/stats")
def stats():
    return jsonify(DATASET_STATS)


@app.route("/api/history")
def history():
    return jsonify(prediction_history)


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    data = request.get_json()
    emails = data.get("emails", [])
    results = []

    for text in emails[:50]:  # Limit to 50
        cleaned = clean_text(text.strip())
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        spam_prob = round(float(probabilities[1]) * 100, 2)
        ham_prob = round(float(probabilities[0]) * 100, 2)

        results.append({
            "text": text[:80] + ("..." if len(text) > 80 else ""),
            "prediction": "Spam" if prediction == 1 else "Not Spam",
            "is_spam": bool(prediction == 1),
            "spam_probability": spam_prob,
            "ham_probability": ham_prob,
        })

    return jsonify(results)


@app.route("/api/sample-emails")
def sample_emails():
    samples = {
        "spam": [
            "Congratulations! You have won a $1000 gift card! Click here to claim now!",
            "URGENT: Your bank account needs verification. Click here immediately.",
            "Buy cheap pills online at 80% discount! No prescription needed.",
            "You've been selected to receive a free iPhone! Act fast!",
            "Make $5000 per week working from home! No experience needed. Sign up now.",
            "WARNING: Your computer has 5 viruses! Download antivirus now!",
        ],
        "ham": [
            "Hi team, the meeting has been rescheduled to 3 PM tomorrow.",
            "Please find attached the quarterly report for your review.",
            "Your interview is scheduled for tomorrow at 10 AM.",
            "Don't forget to call mom today. It's her anniversary.",
            "The project deadline has been extended to March 15. Please plan accordingly.",
            "Your Amazon order #456789 has been shipped. Estimated delivery: Friday.",
        ]
    }
    return jsonify(samples)


if __name__ == "__main__":
    print()
    print("  ============================================================")
    print("    Email Spam Detection -- Web Application")
    print("    Open your browser at: http://localhost:5000")
    print("  ============================================================")
    print()
    app.run(debug=False, host="0.0.0.0", port=5000)
