import joblib
from pathlib import Path
from scipy.sparse import hstack
from classifier.features import load_spacy, load_ngsl, extract_features

MODELS_DIR = Path(__file__).parent / "models"
NGSL_PATH   = Path(__file__).parent.parent.parent / "data" / "ngsl" / "NGSL_12_stats.csv"


def load_classifier():
    """
    Load the production model (Calibrated SVC + Word/Char TF-IDF) and
    supporting resources (reference bands, spaCy, NGSL) from disk.
    Returns a dict with everything needed to run classify() and diagnose().
    """
    return {
        "model":       joblib.load(MODELS_DIR / "svc_calibrated.joblib"),
        "tfidf_word":  joblib.load(MODELS_DIR / "tfidf_word.joblib"),
        "tfidf_char":  joblib.load(MODELS_DIR / "tfidf_char.joblib"),
        "bands":       joblib.load(MODELS_DIR / "reference_bands.joblib"),
        "nlp":         load_spacy(),
        "ngsl":        load_ngsl(NGSL_PATH),
    }


def classify(text, classifier):
    """
    Predict the CEFR level of a text string.

    Returns:
        dict with keys:
            level       — predicted CEFR level string e.g. 'B1'
            confidence  — probability of the predicted class (0.0–1.0)
            probs       — dict of all class probabilities
    """
    X_word = classifier["tfidf_word"].transform([text])
    X_char = classifier["tfidf_char"].transform([text])
    X      = hstack([X_word, X_char])

    predicted = classifier["model"].predict(X)[0]
    proba     = classifier["model"].predict_proba(X)[0]
    classes   = classifier["model"].classes_

    return {
        "level":      predicted,
        "confidence": float(max(proba)),
        "probs":      {cls: float(p) for cls, p in zip(classes, proba)},
    }


def should_accept(result, target_level, min_confidence=0.40, min_margin=0.15):
    """
    Return True only if the predicted level matches the target AND the model
    is sufficiently confident (top prob >= min_confidence and gap to runner-up
    >= min_margin). Prevents accepting borderline predictions.
    """
    if result["level"] != target_level:
        return False
    sorted_probs = sorted(result["probs"].values(), reverse=True)
    top, runner_up = sorted_probs[0], sorted_probs[1]
    return top >= min_confidence and (top - runner_up) >= min_margin


def diagnose(text, target_level, classifier, top_n=3):
    """
    Extract hand-crafted features from text, compare against the target
    level's reference bands, and return a plain-English diagnostic string
    listing the top_n biggest deviations.

    Used to give the Writer specific correction hints when a draft is rejected.
    """
    features = extract_features(text, classifier["nlp"], classifier["ngsl"])
    if features is None:
        return "Could not extract features from text."

    bands = classifier["bands"][target_level]
    deviations = []

    for feat, value in features.items():
        if feat not in bands:
            continue
        b = bands[feat]
        median = b["median"]
        iqr    = b["q75"] - b["q25"]
        # Normalise deviation by IQR (how many IQRs away from median)
        if iqr > 0:
            deviation = abs(value - median) / iqr
        else:
            deviation = abs(value - median)
        direction = "high" if value > median else "low"
        deviations.append((deviation, feat, value, median, b["q25"], b["q75"], direction))

    deviations.sort(reverse=True)
    top = deviations[:top_n]

    parts = []
    for _, feat, value, median, q25, q75, direction in top:
        parts.append(f"{feat}={value:.2f} (target {target_level} typical: {q25:.2f}–{q75:.2f}, {direction})")

    return f"Predicted level does not match target {target_level}. Biggest deviations: " + "; ".join(parts) + "."
