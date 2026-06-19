"""
Email Spam Detection -- Prediction System & Console Interface

Loads the trained model and TF-IDF vectorizer, then provides:
    1. predict_email(text) function for programmatic use
    2. Interactive menu-based console UI
"""

import os
import sys
import joblib

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text


def get_project_root():
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model_and_vectorizer():
    """Load the saved model and TF-IDF vectorizer."""
    root = get_project_root()
    model_path = os.path.join(root, "models", "spam_model.pkl")
    tfidf_path = os.path.join(root, "models", "tfidf.pkl")

    if not os.path.exists(model_path):
        print("[ERROR] Model not found: {}".format(model_path))
        print("   Run train.py first to train the model.")
        sys.exit(1)

    if not os.path.exists(tfidf_path):
        print("[ERROR] Vectorizer not found: {}".format(tfidf_path))
        print("   Run train.py first to train the model.")
        sys.exit(1)

    model = joblib.load(model_path)
    vectorizer = joblib.load(tfidf_path)
    return model, vectorizer


# Load once at module level
_model, _vectorizer = None, None


def _ensure_loaded():
    """Lazy-load model and vectorizer."""
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model, _vectorizer = load_model_and_vectorizer()


def predict_email(text):
    """
    Predict whether an email is Spam or Not Spam.

    Args:
        text (str): The email text to classify.

    Returns:
        dict: {
            "prediction": "Spam" or "Not Spam",
            "confidence": float (spam probability),
            "cleaned_text": str
        }
    """
    _ensure_loaded()

    cleaned = clean_text(text)
    features = _vectorizer.transform([cleaned])
    prediction = _model.predict(features)[0]
    probabilities = _model.predict_proba(features)[0]

    spam_prob = probabilities[1]  # Probability of spam
    label = "Spam" if prediction == 1 else "Not Spam"

    return {
        "prediction": label,
        "confidence": round(spam_prob * 100, 2),
        "cleaned_text": cleaned,
    }


def display_banner():
    """Display the application banner."""
    print()
    print("+" + "=" * 58 + "+")
    print("|" + "  EMAIL SPAM DETECTION SYSTEM".center(58) + "|")
    print("|" + "  Powered by Machine Learning".center(58) + "|")
    print("+" + "=" * 58 + "+")


def display_menu():
    """Display the main menu."""
    print("\n+---------------------------------+")
    print("|         MAIN MENU               |")
    print("+---------------------------------+")
    print("|  1. Check Email                 |")
    print("|  2. View Accuracy               |")
    print("|  3. Exit                        |")
    print("+---------------------------------+")


def check_email():
    """Interactive email checking."""
    print("\n" + "-" * 50)
    print("  CHECK EMAIL")
    print("-" * 50)
    print("  Enter the email text below (or 'back' to return):\n")

    text = input("  Email: ").strip()

    if text.lower() == "back" or not text:
        return

    result = predict_email(text)

    print("\n  +----------- RESULT -----------+")
    if result["prediction"] == "Spam":
        print("  |  >>> Prediction: SPAM          |")
    else:
        print("  |  >>> Prediction: NOT SPAM      |")
    print("  |  Spam Probability: {:>5.1f}%     |".format(result['confidence']))
    print("  +-------------------------------+")
    if len(result['cleaned_text']) > 60:
        print("\n  Cleaned text: \"{}...\"".format(result['cleaned_text'][:60]))
    else:
        print("\n  Cleaned text: \"{}\"".format(result['cleaned_text']))


def view_accuracy():
    """Display the saved accuracy report."""
    print("\n" + "-" * 50)
    print("  MODEL ACCURACY REPORT")
    print("-" * 50)

    root = get_project_root()
    report_path = os.path.join(root, "results", "accuracy_report.txt")

    if not os.path.exists(report_path):
        print("\n  [ERROR] Accuracy report not found.")
        print("  Run train.py first to generate the report.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        print("\n" + f.read())


def main():
    """Run the interactive console interface."""
    display_banner()

    # Pre-load the model
    print("\n  Loading model...")
    _ensure_loaded()
    print("  [OK] Model loaded successfully!")

    while True:
        display_menu()

        try:
            choice = input("\n  Enter your choice (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if choice == "1":
            check_email()
        elif choice == "2":
            view_accuracy()
        elif choice == "3":
            print("\n  Thank you for using Email Spam Detection System!")
            print("  Goodbye!\n")
            break
        else:
            print("\n  [!] Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
