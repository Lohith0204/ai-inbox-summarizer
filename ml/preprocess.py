import re
import nltk

for _res, _kind in [("stopwords", "corpora"), ("punkt", "tokenizers"), ("punkt_tab", "tokenizers")]:
    try:
        nltk.data.find(f"{_kind}/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)

from nltk.corpus import stopwords

_base_stops = set(stopwords.words("english"))
_keep = {"urgent", "important", "deadline", "asap", "immediately", "critical",
         "required", "action", "alert", "warning", "high", "priority", "overdue",
         "today", "tomorrow", "friday"}
STOP_WORDS = _base_stops - _keep


def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " url ", text)
    text = re.sub(r"\S+@\S+\.\S+", " emailaddr ", text)
    text = re.sub(r"\$[\d,]+\.?\d*", " moneyamt ", text)
    text = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", " datetoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    return " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 2)


def preprocess_pipeline(text):
    return remove_stopwords(clean_text(text))
