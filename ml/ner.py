import streamlit as st

ACTION_VERBS = {
    "submit", "send", "review", "approve", "complete", "schedule",
    "update", "confirm", "prepare", "attend", "join", "respond",
    "check", "verify", "sign", "upload", "download", "report",
    "finalize", "present", "follow", "contact", "call", "meet",
    "investigate", "fix", "resolve", "escalate", "deploy", "merge",
}


@st.cache_resource(show_spinner=False)
def _load_nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def extract_entities(text):
    nlp = _load_nlp()
    doc = nlp(text[:5000])

    out = {"persons": [], "organizations": [], "dates": [], "locations": [], "money": []}
    label_map = {
        "PERSON": "persons", "ORG": "organizations",
        "DATE": "dates", "TIME": "dates",
        "GPE": "locations", "LOC": "locations",
        "MONEY": "money",
    }

    seen = set()
    for ent in doc.ents:
        key = label_map.get(ent.label_)
        if key and ent.text not in seen:
            out[key].append(ent.text)
            seen.add(ent.text)

    return out


def extract_action_items(text):
    nlp = _load_nlp()
    doc = nlp(text[:5000])
    items = []

    for sent in doc.sents:
        if any(v in sent.text.lower() for v in ACTION_VERBS):
            cleaned = sent.text.strip()
            if len(cleaned) > 15:
                items.append(cleaned)

    return items[:8]


def extract_deadlines(text):
    nlp = _load_nlp()
    doc = nlp(text[:5000])
    kws = {"deadline", "due", "by", "before", "until", "no later", "eod"}
    out = []

    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            ctx = text[max(0, ent.start_char - 60): ent.end_char].lower()
            if any(k in ctx for k in kws):
                out.append({
                    "date": ent.text,
                    "context": text[max(0, ent.start_char - 30): ent.end_char + 30].strip(),
                })

    return out


def full_ner(text):
    return {
        "entities": extract_entities(text),
        "action_items": extract_action_items(text),
        "deadlines": extract_deadlines(text),
    }
