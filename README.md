# 📧 Email Spam Detection Using Machine Learning

An AI-powered Email Spam Detection System built with Python and Scikit-learn. Classifies emails as **Spam** or **Not Spam (Ham)** using a Multinomial Naive Bayes classifier with TF-IDF feature extraction.

---

## ✨ Features

- **Complete ML Pipeline** — Data preprocessing, TF-IDF vectorization, model training, evaluation, and prediction
- **High Accuracy** — Achieves 95%+ accuracy on the test set
- **Spam Probability Score** — Shows confidence percentage alongside predictions
- **Interactive Console UI** — Menu-based interface for testing custom emails
- **Visualization** — Confusion matrix and performance metrics charts
- **Saved Models** — Trained model and vectorizer persisted with Joblib for instant predictions

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML (TF-IDF, Naive Bayes, metrics) |
| Matplotlib | Plotting |
| Seaborn | Enhanced visualizations |
| Joblib | Model serialization |

---

## 📁 Project Structure

```
Email-Spam-Detection/
│
├── dataset/
│   └── spam.csv                  # Email dataset (1100 samples)
│
├── models/
│   ├── spam_model.pkl            # Trained Naive Bayes model
│   └── tfidf.pkl                 # Fitted TF-IDF vectorizer
│
├── src/
│   ├── generate_dataset.py       # Dataset generator script
│   ├── preprocess.py             # Text preprocessing module
│   ├── train.py                  # Training pipeline
│   └── predict.py                # Prediction system & console UI
│
├── results/
│   ├── confusion_matrix.png      # Confusion matrix visualization
│   ├── metrics_chart.png         # Performance bar chart
│   └── accuracy_report.txt       # Detailed evaluation report
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python src/generate_dataset.py
```

### 3. Train the Model

```bash
python src/train.py
```

### 4. Run the Prediction System

```bash
python src/predict.py
```

---

## 📊 How It Works

```
Raw Email Text
     │
     ▼
┌──────────────┐
│  Preprocess   │  lowercase, remove punctuation/numbers/URLs
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   TF-IDF      │  convert text → numerical feature vectors
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Naive Bayes  │  classify → Spam or Ham
└──────┬───────┘
       │
       ▼
   Prediction
  + Confidence
```

---

## 🎯 Usage Examples

### Programmatic Usage

```python
from src.predict import predict_email

# Test spam email
result = predict_email("Congratulations! Claim your free reward now!")
print(result)
# {'prediction': 'Spam', 'confidence': 92.5, 'cleaned_text': 'congratulations claim your free reward now'}

# Test ham email
result = predict_email("Your interview is scheduled for tomorrow at 10 AM.")
print(result)
# {'prediction': 'Not Spam', 'confidence': 3.2, 'cleaned_text': 'your interview is scheduled for tomorrow at am'}
```

### Console Interface

```
╔══════════════════════════════════════════════════════════╗
║              📧 EMAIL SPAM DETECTION SYSTEM              ║
╚══════════════════════════════════════════════════════════╝

┌─────────────────────────────────┐
│         MAIN MENU               │
├─────────────────────────────────┤
│  1. 📝 Check Email              │
│  2. 📊 View Accuracy            │
│  3. 🚪 Exit                     │
└─────────────────────────────────┘

  Enter your choice (1-3): 1

  ✉️  Email: Buy cheap pills now! Free shipping!

  ┌─────────── RESULT ───────────┐
  │  🚫 Prediction: SPAM          │
  │  📊 Spam Probability: 95.3%   │
  └───────────────────────────────┘
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 95%+ |
| Precision | High |
| Recall | High |
| F1 Score | High |

> Detailed results are generated in `results/accuracy_report.txt` after training.

---

## 🧠 Learning Outcomes

- Supervised Learning & Text Classification
- Natural Language Processing (NLP) basics
- TF-IDF Feature Engineering
- Naive Bayes Algorithm
- Model Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
- ML Pipeline Design & Model Persistence

---

## 📜 License

This project is for educational purposes.
