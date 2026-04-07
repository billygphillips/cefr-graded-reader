import json
import joblib
import torch
from pathlib import Path
from scipy.sparse import hstack
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from classifier.features import load_spacy, load_ngsl, extract_features

MODELS_DIR = Path(__file__).parent / "models"
NGSL_PATH = Path(__file__).parent.parent.parent / "data" / "ngsl" / "NGSL_12_stats.csv"
TRANSFORMER_DIR = MODELS_DIR / "cefr-distilbert-final"


def load_classifier(use_transformer=False):
    """
    Load classifier and supporting resources from disk.
    use_transformer=True loads the fine-tuned DistilBERT model;
    otherwise loads the Calibrated SVC + TF-IDF baseline.
    """
    base = {
        "bands": joblib.load(MODELS_DIR / "reference_bands.joblib"),
        "nlp": load_spacy(),
        "ngsl": load_ngsl(NGSL_PATH),
    }

    if use_transformer:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base["type"] = "transformer"
        base["model"] = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR).to(device)
        base["tokenizer"] = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
        base["device"] = device
        with open(TRANSFORMER_DIR / "label_config.json") as f:
            label_config = json.load(f)
        base["id2label"] = {int(k): v for k, v in label_config["id2label"].items()}
    else:
        base["type"] = "svm"
        base["model"] = joblib.load(MODELS_DIR / "svc_calibrated.joblib")
        base["tfidf_word"] = joblib.load(MODELS_DIR / "tfidf_word.joblib")
        base["tfidf_char"] = joblib.load(MODELS_DIR / "tfidf_char.joblib")

    return base


def classify(text, classifier):
    """
    Predict the CEFR level of a text string.

    Returns:
        dict with keys:
            level       — predicted CEFR level string e.g. 'B1'
            confidence  — probability of the predicted class (0.0–1.0)
            probs       — dict of all class probabilities
    """
    if classifier["type"] == "transformer":
        return _classify_transformer(text, classifier)
    return _classify_svm(text, classifier)


def _classify_svm(text, classifier):
    X_word = classifier["tfidf_word"].transform([text])
    X_char = classifier["tfidf_char"].transform([text])
    X = hstack([X_word, X_char])

    predicted = classifier["model"].predict(X)[0]
    proba = classifier["model"].predict_proba(X)[0]
    classes = classifier["model"].classes_

    return {
        "level": predicted,
        "confidence": float(max(proba)),
        "probs": {cls: float(p) for cls, p in zip(classes, proba)},
    }


def _classify_transformer(text, classifier):
    tokenizer = classifier["tokenizer"]
    model = classifier["model"]
    device = classifier["device"]

    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

    id2label = classifier["id2label"]
    predicted_id = int(probs.argmax())

    return {
        "level": id2label[predicted_id],
        "confidence": float(probs[predicted_id]),
        "probs": {id2label[i]: float(p) for i, p in enumerate(probs)},
    }


def should_accept(result, target_level, min_confidence=0.40, min_margin=0.05):
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


CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]

# Features where higher values = more complex text
_COMPLEXITY_UP = {
    "avg_sent_len",
    "avg_word_len",
    "out_of_ngsl",
    "subord_ratio",
    "relative_clause_rate",
    "passive_rate",
    "perfect_rate",
    "modal_diversity",
    "conditional_rate",
    "avg_tree_depth",
    "connector_tier",
    "ttr",
}
# Features where higher values = simpler text
_COMPLEXITY_DOWN = {
    "ngsl_800",
    "ngsl_1000",
    "ngsl_2000",
    "ngsl_3000",
}


def diagnose(text, target_level, classifier, predicted_level=None, top_n=3):
    """
    Extract hand-crafted features from text, compare against the target
    level's reference bands, and return a plain-English diagnostic string
    listing the top_n biggest deviations that push the text AWAY from the
    target level.

    Only flags features deviating toward the predicted level — e.g. if text
    is too complex for A2, only flags features that are above the A2 band.
    Features that are below the A2 band (simpler than typical A2) are fine.
    """
    features = extract_features(text, classifier["nlp"], classifier["ngsl"])
    if features is None:
        return "Could not extract features from text."

    bands = classifier["bands"][target_level]

    # Is the text too complex or too simple relative to target?
    too_complex = True  # default assumption for graded readers
    if predicted_level and predicted_level in CEFR_ORDER and target_level in CEFR_ORDER:
        too_complex = CEFR_ORDER.index(predicted_level) >= CEFR_ORDER.index(
            target_level
        )

    deviations = []
    for feat, value in features.items():
        if feat not in bands:
            continue
        b = bands[feat]
        iqr = b["q75"] - b["q25"]
        guard = max(iqr, 0.01)

        if too_complex:
            # Only flag features pushing toward MORE complexity
            if feat in _COMPLEXITY_UP and value > b["q75"]:
                severity = (value - b["q75"]) / guard
                deviations.append(
                    (severity, feat, value, b["q25"], b["q75"], "too high — simplify")
                )
            elif feat in _COMPLEXITY_DOWN and value < b["q25"]:
                severity = (b["q25"] - value) / guard
                deviations.append(
                    (
                        severity,
                        feat,
                        value,
                        b["q25"],
                        b["q75"],
                        "too low — use simpler words",
                    )
                )
        else:
            # Only flag features pushing toward LESS complexity
            if feat in _COMPLEXITY_UP and value < b["q25"]:
                severity = (b["q25"] - value) / guard
                deviations.append(
                    (severity, feat, value, b["q25"], b["q75"], "too low — increase")
                )
            elif feat in _COMPLEXITY_DOWN and value > b["q75"]:
                severity = (value - b["q75"]) / guard
                deviations.append(
                    (
                        severity,
                        feat,
                        value,
                        b["q25"],
                        b["q75"],
                        "too high — use more varied words",
                    )
                )

    deviations.sort(reverse=True)
    top = deviations[:top_n]

    if not top:
        return (
            f"Classified as {predicted_level or '?'} (target {target_level}), "
            "but no clear feature-level issues found — the TF-IDF model may be "
            "responding to vocabulary or n-gram patterns not captured by these features."
        )

    direction_word = "too complex" if too_complex else "too simple"
    parts = []
    for _, feat, value, q25, q75, hint in top:
        parts.append(
            f"{feat}={value:.2f} ({target_level} typical: {q25:.2f}–{q75:.2f}, {hint})"
        )

    return (
        f"Your draft was classified as {predicted_level or '?'} — {direction_word} for {target_level}. "
        "Key issues: " + "; ".join(parts) + "."
    )


# NGSL band thresholds per target level
_LEVEL_BAND = {"A2": 1000, "B1": 2000, "B2": 3000}


def lexical_diagnostic(text, target_level, classifier, max_words=10):
    """
    Return a list of the most frequent content words in the prose that are
    above the target level's NGSL vocabulary band.

    Uses lemmatisation for matching; reports surface forms for readability.
    Returns an empty list if all content words are within the band.
    """
    band = _LEVEL_BAND.get(target_level, 1000)
    nlp  = classifier["nlp"]
    ngsl = classifier["ngsl"]

    doc = nlp(text)
    above_band: dict[str, int] = {}

    for token in doc:
        if token.is_punct or token.is_space:
            continue
        # Content words only — skip proper nouns (character names) and function words
        if token.pos_ not in {"NOUN", "VERB", "ADJ", "ADV"}:
            continue
        if token.pos_ == "PROPN":
            continue
        lemma = token.lemma_.lower()
        if len(lemma) <= 2:
            continue
        if lemma not in ngsl[band]:
            surface = token.text.lower()
            above_band[surface] = above_band.get(surface, 0) + 1

    sorted_words = sorted(above_band.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_words]]
