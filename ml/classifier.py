import os
import pickle

import numpy as np
import streamlit as st

from ml.preprocess import preprocess_pipeline

MODEL_DIR = "models"

CATEGORY_META = {
    "work":       {"emoji": "💼", "color": "#4F8EF7", "label": "Work"},
    "spam":       {"emoji": "🚨", "color": "#FF4444", "label": "Spam"},
    "newsletter": {"emoji": "📰", "color": "#9B59B6", "label": "Newsletter"},
    "personal":   {"emoji": "💬", "color": "#2ECC71", "label": "Personal"},
    "finance":    {"emoji": "💰", "color": "#F1C40F", "label": "Finance"},
    "alerts":     {"emoji": "⚡", "color": "#E74C3C", "label": "Alert"},
}


@st.cache_resource(show_spinner=False)
def load_classifier():
    clf_path = os.path.join(MODEL_DIR, "classifier.pkl")
    vec_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

    if not os.path.exists(clf_path):
        raise FileNotFoundError("Classifier not found. Run notebooks/train.py first.")

    with open(clf_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def classify_email(text):
    model, vectorizer = load_classifier()
    X = vectorizer.transform([preprocess_pipeline(text)])
    return model.predict(X)[0]


def classify_with_confidence(text):
    model, vectorizer = load_classifier()
    X = vectorizer.transform([preprocess_pipeline(text)])

    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    return {
        "prediction": prediction,
        "confidence": float(np.max(probs)),
        "scores": {c: float(p) for c, p in zip(classes, probs)},
        "meta": CATEGORY_META.get(prediction, {"emoji": "📧", "color": "#888", "label": prediction}),
    }
