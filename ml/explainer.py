import os
import pickle

import numpy as np
import streamlit as st

from ml.preprocess import preprocess_pipeline


@st.cache_resource(show_spinner=False)
def _load_explainer():
    import shap
    from scipy.sparse import csr_matrix

    with open(os.path.join("models", "classifier.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join("models", "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    n_features = len(vectorizer.vocabulary_)
    background = csr_matrix((1, n_features), dtype=np.float32)

    explainer = shap.LinearExplainer(model, background, feature_dependence="independent")
    return explainer, model, vectorizer


def explain_prediction(text, top_n=12):
    try:
        import shap

        explainer, model, vectorizer = _load_explainer()

        processed = preprocess_pipeline(text)
        X = vectorizer.transform([processed])

        prediction = model.predict(X)[0]
        classes = list(model.classes_)
        pred_idx = classes.index(prediction)

        shap_vals = explainer.shap_values(X)

        if isinstance(shap_vals, list):
            sv = np.asarray(shap_vals[pred_idx]).flatten()
        else:
            sv = np.asarray(shap_vals).flatten()

        feature_names = vectorizer.get_feature_names_out()
        x_dense = np.asarray(X.todense()).flatten()
        nonzero_mask = x_dense > 0

        if nonzero_mask.sum() == 0:
            return {"status": "no_features", "prediction": prediction, "top_features": []}

        sv_nz = sv[nonzero_mask]
        fn_nz = feature_names[nonzero_mask]
        top_idx = np.argsort(np.abs(sv_nz))[::-1][:top_n]

        features = [
            {
                "word": str(fn_nz[i]),
                "shap": float(sv_nz[i]),
                "direction": "positive" if sv_nz[i] > 0 else "negative",
            }
            for i in top_idx
        ]

        return {"status": "success", "prediction": prediction, "top_features": features}

    except Exception as exc:
        return {"status": "error", "message": str(exc), "prediction": "", "top_features": []}
