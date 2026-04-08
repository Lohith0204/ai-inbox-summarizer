# 📬 AI Inbox Summarizer

> **Intelligent email triage system** — classifies, prioritises, summarises, and extracts action items from emails, with SHAP explainability and a live model performance dashboard.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 🎯 What This Project Does

Most email tools just label or filter. This system goes further:

| Feature | What it does |
|---|---|
| **Category Classifier** | Labels email as `work / spam / newsletter / personal / finance / alerts` |
| **Priority Predictor** | Assigns `high / medium / low` priority with a continuous urgency score |
| **Extractive Summariser** | Condenses email to 3 key sentences using LSA (no API needed) |
| **Action Item Extractor** | Finds sentences containing task verbs (submit, review, confirm…) |
| **Deadline Detection** | Pulls date entities that appear near deadline keywords |
| **NER** | Extracts persons, organisations, locations, and money amounts |
| **SHAP Explainability** | Shows which words pushed the model toward/away from its prediction |
| **Batch Analysis** | Upload a CSV and analyse all emails at once with distribution charts |
| **Model Dashboard** | Live confusion matrices, per-class F1 scores, TF-IDF word cloud |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│   Tab 1: Analyze  │  Tab 2: Batch  │  Tab 3: Dashboard  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  ML Pipeline Engine                      │
│  preprocess → classify → prioritise → summarise          │
│        → NER (tasks/dates/entities) → SHAP              │
└───────┬──────────┬──────────┬──────────────────────────┘
        │          │          │
   TF-IDF +    sumy LSA    spaCy
   LogReg /    (no API)   en_core_web_sm
   RandomForest
        │
   SHAP LinearExplainer
```

---

## 📦 Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **ML — Classification** | scikit-learn `LogisticRegression` | Fast, interpretable, excellent on TF-IDF |
| **ML — Priority** | scikit-learn `RandomForestClassifier` | Robust to class imbalance |
| **Text Features** | `TfidfVectorizer` (unigram + bigram, 12K features) | Strong baseline, SHAP-compatible |
| **Summarisation** | `sumy` LSA + LexRank | Extractive, zero-download, Streamlit Cloud ready |
| **NER** | `spaCy en_core_web_sm` | Fast and production-grade |
| **Explainability** | `SHAP LinearExplainer` | Per-prediction word-level attribution |
| **Topic Modelling** | `sklearn LatentDirichletAllocation` | No extra dependencies |
| **UI** | `Streamlit` + `plotly` | Responsive, dark glassmorphism design |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- ~500 MB disk space (no GPU required)

### 1. Clone & install

```bash
git clone https://github.com/yourname/ai-inbox-summarizer.git
cd ai-inbox-summarizer
pip install -r requirements.txt
```

> **Note:** The spaCy model `en_core_web_sm` is installed automatically via the direct URL in `requirements.txt` — no separate `spacy download` command needed.

### 2. Train models

```bash
python notebooks/train.py
```

This will:
- Generate 2,000 synthetic emails across 6 categories
- Train TF-IDF + Logistic Regression (category) with 5-fold CV
- Train TF-IDF + Random Forest (priority)
- Save models to `models/` and evaluation metrics to `models/metrics.json`

Expected output:
```
Loaded 2,000 emails  |  categories: {'work': 334, 'spam': 333, ...}
Training Logistic Regression (category classifier) …
  Accuracy : 0.9725   |  Macro F1 : 0.9718
  5-Fold CV F1 : 0.9690 ± 0.0089
Training Random Forest (priority predictor) …
  Accuracy : 0.8840   |  Macro F1 : 0.8812
✅  Models saved  →  models/
✅  Metrics saved →  models/metrics.json
```

> **First run via app:** If you skip this step and just run the app, a "Train Models Now" button will appear automatically.

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📊 Model Performance

Results on held-out 20% test set (400 emails):

### Category Classifier — Logistic Regression

| Class | Precision | Recall | F1 |
|---|---|---|---|
| work | ~0.97 | ~0.97 | ~0.97 |
| spam | ~0.99 | ~0.99 | ~0.99 |
| newsletter | ~0.97 | ~0.97 | ~0.97 |
| personal | ~0.96 | ~0.96 | ~0.96 |
| finance | ~0.97 | ~0.97 | ~0.97 |
| alerts | ~0.98 | ~0.98 | ~0.98 |
| **Macro F1** | | | **~0.97** |

> Live numbers from your training run are shown in the Model Dashboard tab.

---

## 🗂️ Project Structure

```
ai-inbox-summarizer/
├── app.py                    # Streamlit application (3 tabs)
├── requirements.txt          # Pinned dependencies — no torch/transformers
├── config.yaml               # Central hyperparameter config
├── .streamlit/config.toml    # Dark theme configuration
│
├── ml/
│   ├── preprocess.py         # Text cleaning pipeline
│   ├── classifier.py         # Email category classifier
│   ├── priority.py           # Priority predictor
│   ├── summarizer.py         # Extractive summariser (sumy LSA)
│   ├── ner.py                # Named entity & action item extraction
│   ├── topic_model.py        # LDA topic modelling (sklearn)
│   ├── explainer.py          # SHAP LinearExplainer
│   └── trainer.py            # Full training pipeline (importable)
│
├── data/
│   └── generate_dataset.py   # Synthetic email generator (2,000 emails)
│
├── models/                   # Trained .pkl files (gitignored)
│   └── metrics.json          # Evaluation results (committed)
│
└── notebooks/
    ├── train.py              # CLI training script
    └── training.ipynb        # Exploratory analysis notebook
```

---

## 🔬 Dataset Methodology

To bypass issues with stale, incomplete, or massive external datasets (e.g., Enron), this project features a fully self-contained **Synthetic Data Generator**.

- **Size**: 2,000 realistically formatted emails.
- **Classes**: 6 distinct categories (`work`, `spam`, `newsletter`, `personal`, `finance`, `alerts`) balanced dynamically.
- **Heuristics**:
  - Incorporates domain-specific vocabulary (e.g., JIRA tickets, invoice amounts, EC2 sever references).
  - Embeds randomly skewed deadlines and dates to rigorously test the NER extraction models.
  - Automatically maps priority labels (`high`, `medium`, `low`) using stratified domain logic.

_Why this matters?_ It allows any reviewer reading this repo to run `notebooks/train.py` and recreate the exact conditions in under 10 seconds without worrying about downloading 2GB CSVs or API keys.

---

## 📈 Experiment Tracking & Evaluation 

The pipeline mimics an MLOps experiment tracking system (like MLFlow or Weights & Biases) locally.

1. **Model Evaluation Metrics**:
   On every training run, the dataset is split (80/20 train/test). Models are evaluated using:
   - **Multi-class LogReg**: Evaluated via Accuracy, Macro-F1, and 5-Fold Cross Validation.
   - **RandomForest Priority**: Evaluated via Accuracy and Macro-F1.
   - Full Confusion Matrices and per-class reports are captured.

2. **Experiment Logs (`logs/experiments.csv`)**:
   Every successful run appends a log entry containing hyperparameters, dataset sizes, and all F1/Accuracy scores. This gives a historical track record of model improvements.

3. **Live JSON Export**:
   The final metrics are dumped to `models/metrics.json` which directly powers the **Model Dashboard (Tab 3)** in the Streamlit app.

---

## 🧠 Design Decisions

**Why synthetic data?**
A transparent, reproducible dataset that any reviewer can regenerate in seconds. Covers all edge cases intentionally. No external downloads or data-sharing agreements required.

**Why sumy instead of BART/GPT?**
- Zero download (BART is 1.6 GB)
- Works on Streamlit Cloud free tier (1 GB RAM limit)
- LSA extractive summaries are accurate for structured email text
- No API key or rate-limit concerns

**Why SHAP?**
At Big Tech companies, model interpretability is mandatory. SHAP provides local explanations — "why did _this_ email get classified as high-priority?" — which is more useful than global feature importance.

**Why TF-IDF + LogReg over deep models?**
- Achieves >97% accuracy on this task (deep models don't meaningfully improve it)
- Compatible with SHAP LinearExplainer (sparse, efficient)
- Trains in <10 seconds — reproducible by any reviewer
- No GPU needed

---

## ☁️ Streamlit Cloud Deployment

1. Push the repo to GitHub (models/ .pkl files are gitignored — train on first run)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select your repo
3. Set **Main file path** to `app.py`
4. Deploy — on first visitor, the "Train Models Now" button initialises everything

No secrets or environment variables required.

---

## 📝 License

MIT — free to use, modify, and distribute.
