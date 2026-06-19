"""
Email Spam Detection -- Model Training Pipeline

Steps:
    1. Load dataset
    2. Preprocess text
    3. TF-IDF feature extraction
    4. Train/test split (80/20)
    5. Train Multinomial Naive Bayes
    6. Evaluate (accuracy, precision, recall, F1, confusion matrix)
    7. Save model, vectorizer, and results
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_dataset


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(path):
    """Load the email dataset from CSV."""
    print("=" * 60)
    print("  STEP 1: Loading Dataset")
    print("=" * 60)

    df = pd.read_csv(path)

    print("\n[*] Dataset loaded from: {}".format(path))
    print("[*] Shape: {} rows x {} columns".format(df.shape[0], df.shape[1]))

    print("\n--- First 5 Records ---")
    print(df.head().to_string())

    print("\n--- Missing Values ---")
    print(df.isnull().sum().to_string())

    print("\n--- Class Distribution ---")
    print(df["label"].value_counts().to_string())

    spam_pct = (df["label"] == "spam").mean() * 100
    ham_pct = (df["label"] == "ham").mean() * 100
    print("\n  Spam: {:.1f}%  |  Ham: {:.1f}%".format(spam_pct, ham_pct))

    return df


def preprocess(df):
    """Preprocess the dataset."""
    print("\n" + "=" * 60)
    print("  STEP 2: Data Preprocessing")
    print("=" * 60)

    df = preprocess_dataset(df)

    print("\n[OK] Preprocessing complete.")
    print("   Records after cleaning: {}".format(len(df)))
    print("   Spam: {}  |  Ham: {}".format((df['label'] == 1).sum(), (df['label'] == 0).sum()))
    print("\n--- Sample Cleaned Text ---")
    for i in range(min(3, len(df))):
        label = "SPAM" if df.iloc[i]["label"] == 1 else "HAM"
        print("   [{}] {}...".format(label, df.iloc[i]['text'][:80]))

    return df


def extract_features(df):
    """Extract TF-IDF features from text."""
    print("\n" + "=" * 60)
    print("  STEP 3: Feature Extraction (TF-IDF)")
    print("=" * 60)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    print("\n[OK] TF-IDF vectorization complete.")
    print("   Feature matrix shape: {}".format(X.shape))
    print("   Vocabulary size: {}".format(len(vectorizer.vocabulary_)))
    print("   Sample features: {}".format(list(vectorizer.get_feature_names_out()[:10])))

    return X, y, vectorizer


def split_data(X, y):
    """Split into training and testing sets."""
    print("\n" + "=" * 60)
    print("  STEP 4: Train-Test Split")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[OK] Data split complete.")
    print("   Training set: {} samples".format(X_train.shape[0]))
    print("   Testing set:  {} samples".format(X_test.shape[0]))

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Train the Multinomial Naive Bayes classifier."""
    print("\n" + "=" * 60)
    print("  STEP 5: Model Training (Multinomial Naive Bayes)")
    print("=" * 60)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)
    print("\n[OK] Model training complete.")
    print("   Training accuracy: {:.4f} ({:.2f}%)".format(train_accuracy, train_accuracy*100))

    return model


def evaluate_model(model, X_test, y_test, results_dir):
    """Evaluate the model and save results."""
    print("\n" + "=" * 60)
    print("  STEP 6: Model Evaluation")
    print("=" * 60)

    y_pred = model.predict(X_test)

    # --- Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "-" * 40)
    print("  EVALUATION RESULTS")
    print("-" * 40)
    print("  Accuracy:  {:.4f}  ({:.2f}%)".format(accuracy, accuracy*100))
    print("  Precision: {:.4f}  ({:.2f}%)".format(precision, precision*100))
    print("  Recall:    {:.4f}  ({:.2f}%)".format(recall, recall*100))
    print("  F1 Score:  {:.4f}  ({:.2f}%)".format(f1, f1*100))
    print("-" * 40)

    print("\n--- Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    print(report)

    # --- Confusion Matrix Plot ---
    os.makedirs(results_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        linewidths=1,
        linecolor="gray",
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.title("Confusion Matrix - Email Spam Detection", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Actual Label", fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print("\n[PLOT] Confusion matrix saved: {}".format(cm_path))

    # --- Bar chart of metrics ---
    plt.figure(figsize=(8, 5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metrics_values = [accuracy, precision, recall, f1]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    bars = plt.bar(metrics_names, metrics_values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 "{:.1f}%".format(val*100), ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.ylim(0, 1.15)
    plt.title("Model Performance Metrics", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()

    metrics_path = os.path.join(results_dir, "metrics_chart.png")
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print("[PLOT] Metrics chart saved: {}".format(metrics_path))

    # --- Save accuracy report ---
    report_path = os.path.join(results_dir, "accuracy_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("  EMAIL SPAM DETECTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("Accuracy:  {:.4f}  ({:.2f}%)\n".format(accuracy, accuracy*100))
        f.write("Precision: {:.4f}  ({:.2f}%)\n".format(precision, precision*100))
        f.write("Recall:    {:.4f}  ({:.2f}%)\n".format(recall, recall*100))
        f.write("F1 Score:  {:.4f}  ({:.2f}%)\n\n".format(f1, f1*100))
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

    print("[FILE] Accuracy report saved: {}".format(report_path))

    return accuracy


def save_model(model, vectorizer, models_dir):
    """Save the trained model and TF-IDF vectorizer."""
    print("\n" + "=" * 60)
    print("  STEP 7: Saving Model & Vectorizer")
    print("=" * 60)

    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "spam_model.pkl")
    tfidf_path = os.path.join(models_dir, "tfidf.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, tfidf_path)

    print("\n[OK] Model saved:      {}".format(model_path))
    print("[OK] Vectorizer saved: {}".format(tfidf_path))


def main():
    """Run the full training pipeline."""
    print()
    print("+" + "=" * 58 + "+")
    print("|" + "  EMAIL SPAM DETECTION -- TRAINING PIPELINE".center(58) + "|")
    print("+" + "=" * 58 + "+")
    print()

    root = get_project_root()
    dataset_path = os.path.join(root, "dataset", "spam.csv")
    models_dir = os.path.join(root, "models")
    results_dir = os.path.join(root, "results")

    # Check dataset exists
    if not os.path.exists(dataset_path):
        print("[ERROR] Dataset not found: {}".format(dataset_path))
        print("   Run generate_dataset.py first to create the dataset.")
        sys.exit(1)

    # Pipeline
    df = load_dataset(dataset_path)
    df = preprocess(df)
    X, y, vectorizer = extract_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test, results_dir)
    save_model(model, vectorizer, models_dir)

    # Summary
    print()
    print("+" + "=" * 58 + "+")
    print("|" + "  TRAINING COMPLETE".center(58) + "|")
    print("+" + "=" * 58 + "+")

    if accuracy >= 0.95:
        print("\n>> Model accuracy: {:.2f}% -- TARGET MET (>= 95%)!".format(accuracy*100))
    else:
        print("\n>> Model accuracy: {:.2f}% -- Below 95% target.".format(accuracy*100))

    print("\nFiles saved:")
    print("  [MODEL]  models/spam_model.pkl")
    print("  [MODEL]  models/tfidf.pkl")
    print("  [PLOT]   results/confusion_matrix.png")
    print("  [PLOT]   results/metrics_chart.png")
    print("  [REPORT] results/accuracy_report.txt")
    print()


if __name__ == "__main__":
    main()
