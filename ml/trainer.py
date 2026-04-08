import os
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ml.preprocess import preprocess_pipeline

DATA_PATH = "data/emails.csv"
MODELS_DIR = "models"
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def run_training_pipeline(data_path=DATA_PATH, models_dir=MODELS_DIR, verbose=True):
    os.makedirs(models_dir, exist_ok=True)
    log = print if verbose else lambda *a, **k: None

    if not os.path.exists(data_path):
        log("Dataset not found — generating synthetic emails...")
        from data.generate_dataset import generate_dataset
        df = generate_dataset(save_path=data_path)
    else:
        df = pd.read_csv(data_path)

    log(f"Loaded {len(df):,} emails | categories: {df['category'].value_counts().to_dict()}")

    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(preprocess_pipeline)

    X = df["text"].values
    y_cat = df["category"].values
    y_pri = df["priority"].values

    X_tr, X_te, yc_tr, yc_te, yp_tr, yp_te = train_test_split(
        X, y_cat, y_pri, test_size=0.20, random_state=42, stratify=y_cat
    )

    log("Fitting TF-IDF vectoriser...")
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
    )
    X_tr_tfidf = vectorizer.fit_transform(X_tr)
    X_te_tfidf = vectorizer.transform(X_te)

    log("Training Logistic Regression (category classifier)...")
    clf = LogisticRegression(C=1.5, max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_tr_tfidf, yc_tr)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_tr_tfidf, yc_tr, cv=cv, scoring="f1_macro", n_jobs=-1)

    yc_pred = clf.predict(X_te_tfidf)
    cat_acc = accuracy_score(yc_te, yc_pred)
    cat_f1 = f1_score(yc_te, yc_pred, average="macro")
    cat_report = classification_report(yc_te, yc_pred, output_dict=True)
    cat_cm = confusion_matrix(yc_te, yc_pred, labels=clf.classes_).tolist()

    log(f"  Accuracy : {cat_acc:.4f}  |  Macro F1 : {cat_f1:.4f}")
    log(f"  5-Fold CV F1 : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    log("Training Random Forest (priority predictor)...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr_tfidf, yp_tr)

    yp_pred = rf.predict(X_te_tfidf)
    pri_acc = accuracy_score(yp_te, yp_pred)
    pri_f1 = f1_score(yp_te, yp_pred, average="macro")
    pri_report = classification_report(yp_te, yp_pred, output_dict=True)
    pri_cm = confusion_matrix(yp_te, yp_pred, labels=rf.classes_).tolist()

    log(f"  Accuracy : {pri_acc:.4f}  |  Macro F1 : {pri_f1:.4f}")

    for fname, obj in [("classifier.pkl", clf), ("priority.pkl", rf), ("vectorizer.pkl", vectorizer)]:
        with open(os.path.join(models_dir, fname), "wb") as f:
            pickle.dump(obj, f)

    def _per_class(report):
        skip = {"accuracy", "macro avg", "weighted avg"}
        return {
            k: {
                "f1": round(v["f1-score"], 4),
                "precision": round(v["precision"], 4),
                "recall": round(v["recall"], 4),
                "support": v["support"],
            }
            for k, v in report.items()
            if k not in skip
        }

    metrics = {
        "trained_at": datetime.now().isoformat(),
        "dataset_size": int(len(df)),
        "train_size": int(len(X_tr)),
        "test_size": int(len(X_te)),
        "classifier": {
            "name": "Logistic Regression",
            "accuracy": round(cat_acc, 4),
            "macro_f1": round(cat_f1, 4),
            "cv_f1_mean": round(float(cv_f1.mean()), 4),
            "cv_f1_std": round(float(cv_f1.std()), 4),
            "classes": clf.classes_.tolist(),
            "per_class": _per_class(cat_report),
            "confusion_matrix": cat_cm,
            "confusion_matrix_labels": clf.classes_.tolist(),
        },
        "priority_model": {
            "name": "Random Forest",
            "n_estimators": 150,
            "accuracy": round(pri_acc, 4),
            "macro_f1": round(pri_f1, 4),
            "classes": rf.classes_.tolist(),
            "per_class": _per_class(pri_report),
            "confusion_matrix": pri_cm,
            "confusion_matrix_labels": rf.classes_.tolist(),
        },
        "vectorizer": {
            "max_features": 12000,
            "ngram_range": [1, 2],
            "vocab_size": len(vectorizer.vocabulary_),
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    import csv
    LOGS_DIR = "logs"
    os.makedirs(LOGS_DIR, exist_ok=True)
    exp_log_path = os.path.join(LOGS_DIR, "experiments.csv")
    
    file_exists = os.path.isfile(exp_log_path)
    
    with open(exp_log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Dataset_Size", "Cat_Model", "Cat_Macro_F1", "Cat_CV_F1", "Pri_Model", "Pri_Macro_F1"])
            
        writer.writerow([
            metrics["trained_at"],
            metrics["dataset_size"],
            metrics["classifier"]["name"],
            metrics["classifier"]["macro_f1"],
            metrics["classifier"]["cv_f1_mean"],
            metrics["priority_model"]["name"],
            metrics["priority_model"]["macro_f1"],
        ])

    log(f"\n✅  Models saved  →  {models_dir}/")
    log(f"✅  Metrics saved →  {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    run_training_pipeline(verbose=True)
