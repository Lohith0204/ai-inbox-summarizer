import os
import pickle

import numpy as np
import streamlit as st

from ml.preprocess import preprocess_pipeline

PRIORITY_META = {
    "high":   {"emoji": "🔴", "color": "#FF4444"},
    "medium": {"emoji": "🟡", "color": "#F1C40F"},
    "low":    {"emoji": "🟢", "color": "#2ECC71"},
}

_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.0}


@st.cache_resource(show_spinner=False)
def load_priority():
    from ml.classifier import load_classifier
    pri_path = os.path.join("models", "priority.pkl")
    if not os.path.exists(pri_path):
        raise FileNotFoundError("Priority model not found. Run notebooks/train.py first.")
    with open(pri_path, "rb") as f:
        model = pickle.load(f)
    _, vectorizer = load_classifier()
    return model, vectorizer


def predict_priority(text):
    model, vec = load_priority()
    X = vec.transform([preprocess_pipeline(text)])
    return model.predict(X)[0]


def predict_priority_full(text):
    model, vec = load_priority()
    X = vec.transform([preprocess_pipeline(text)])

    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    urgency = sum(_WEIGHT.get(c, 0.0) * p for c, p in zip(classes, proba))

    return {
        "label": label,
        "urgency_score": round(float(urgency), 3),
        "confidence": float(np.max(proba)),
        "scores": {c: float(p) for c, p in zip(classes, proba)},
        "meta": PRIORITY_META.get(label, {"emoji": "⬜", "color": "#888"}),
    }
