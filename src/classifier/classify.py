import joblib
from pathlib import Path
from scipy.sparse import hstack

MODELS_DIR = Path(__file__).parent / "models"


def load_classifier():
    """
    Load the production model (Calibrated SVC + Word/Char TF-IDF) from disk.
    Returns a dict with everything needed to run classify().
    """
    return {
        "model":      joblib.load(MODELS_DIR / "svc_calibrated.joblib"),
        "tfidf_word": joblib.load(MODELS_DIR / "tfidf_word.joblib"),
        "tfidf_char": joblib.load(MODELS_DIR / "tfidf_char.joblib"),
    }


def classify(text, classifier):
    """
    Predict the CEFR level of a text string.

    Args:
        text:       any string (sentence, paragraph, or full episode)
        classifier: dict returned by load_classifier()

    Returns:
        dict with keys:
            level       — predicted CEFR level string e.g. 'B1'
            confidence  — probability of the predicted class (0.0–1.0)
            probs       — dict of all class probabilities
    """
    X_word = classifier["tfidf_word"].transform([text])
    X_char = classifier["tfidf_char"].transform([text])
    X = hstack([X_word, X_char])

    predicted = classifier["model"].predict(X)[0]
    proba     = classifier["model"].predict_proba(X)[0]
    classes   = classifier["model"].classes_

    return {
        "level":      predicted,
        "confidence": float(max(proba)),
        "probs":      {cls: float(p) for cls, p in zip(classes, proba)},
    }
