import joblib
from pathlib import Path
from classifier.features import load_spacy, load_ngsl, extract_features

MODELS_DIR = Path(__file__).parent / "models"
NGSL_PATH = Path(__file__).parent.parent.parent / "data" / "ngsl" / "NGSL_12_stats.csv"


def load_classifier():
    """
    Load the trained LR model, scaler, and NGSL sets from disk.
    Returns a dict with everything needed to run classify().
    """
    return {
        "model":   joblib.load(MODELS_DIR / "lr_classifier.joblib"),
        "scaler":  joblib.load(MODELS_DIR / "lr_scaler.joblib"),
        "ngsl":    joblib.load(MODELS_DIR / "ngsl_sets.joblib"),
        "nlp":     load_spacy(),
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
    features = extract_features(text, classifier["nlp"], classifier["ngsl"])
    if features is None:
        return None

    feature_cols = [
        'avg_sent_len', 'ngsl_800', 'ngsl_1000', 'ngsl_2000', 'ngsl_3000',
        'ttr', 'avg_tree_depth', 'subord_ratio', 'avg_word_len', 'out_of_ngsl'
    ]
    X = [[features[col] for col in feature_cols]]
    X_scaled = classifier["scaler"].transform(X)

    predicted = classifier["model"].predict(X_scaled)[0]
    proba = classifier["model"].predict_proba(X_scaled)[0]
    classes = classifier["model"].classes_

    return {
        "level":      predicted,
        "confidence": float(max(proba)),
        "probs":      {cls: float(p) for cls, p in zip(classes, proba)},
    }
