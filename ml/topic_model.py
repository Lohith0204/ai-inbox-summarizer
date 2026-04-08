import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from ml.preprocess import preprocess_pipeline

_TOPIC_LABEL_MAP = {
    frozenset(["meeting", "project", "team", "deadline", "review"]): "Project Management",
    frozenset(["invoice", "payment", "amount", "account", "due"]):    "Finance & Billing",
    frozenset(["click", "free", "prize", "verify", "account"]):       "Spam / Phishing",
    frozenset(["update", "digest", "newsletter", "weekly", "read"]):  "News & Updates",
    frozenset(["alert", "error", "failed", "server", "cpu"]):         "System Alerts",
    frozenset(["hey", "hope", "catch", "coffee", "weekend"]):         "Personal",
}


def _label_topic(top_words):
    word_set = set(top_words)
    for keywords, label in _TOPIC_LABEL_MAP.items():
        if len(keywords & word_set) >= 2:
            return label
    return "General"


def get_topics(texts, n_topics=4, n_words=8):
    if len(texts) < max(n_topics, 5):
        return [(0, "General", "Not enough emails for topic modelling")]

    processed = [preprocess_pipeline(t) for t in texts]

    try:
        vec = CountVectorizer(max_features=600, stop_words="english", min_df=2, max_df=0.90)
        X = vec.fit_transform(processed)
        feature_names = vec.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=n_topics, max_iter=20,
            learning_method="online", random_state=42, n_jobs=-1,
        )
        lda.fit(X)

        results = []
        for tid, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[: -n_words - 1: -1]
            top_words = [str(feature_names[i]) for i in top_idx]
            results.append((tid, _label_topic(top_words), " · ".join(top_words)))

        return results

    except Exception as e:
        return [(0, "Error", str(e))]


def assign_topics_to_emails(texts, n_topics=4):
    if len(texts) < n_topics:
        return [0] * len(texts)

    processed = [preprocess_pipeline(t) for t in texts]

    try:
        vec = CountVectorizer(max_features=600, stop_words="english", min_df=2)
        X = vec.fit_transform(processed)
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=15, random_state=42)
        dist = lda.fit_transform(X)
        return [int(np.argmax(d)) for d in dist]
    except Exception:
        return [0] * len(texts)
