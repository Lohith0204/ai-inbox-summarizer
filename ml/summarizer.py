import re
import nltk

for _tok in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{_tok}" if "punkt" in _tok else f"corpora/{_tok}")
    except LookupError:
        nltk.download(_tok, quiet=True)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANG = "english"


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def summarize(text, n_sentences=3):
    if not text or not isinstance(text, str):
        return "No content to summarise."

    text = text.strip()
    if len(text.split()) < 40:
        return text

    try:
        parser = PlaintextParser.from_string(text, Tokenizer(LANG))
        stemmer = Stemmer(LANG)
        summ = LsaSummarizer(stemmer)
        summ.stop_words = get_stop_words(LANG)
        result = summ(parser.document, n_sentences)
        out = " ".join(str(s) for s in result)
        return out if out.strip() else text[:300] + "…"
    except Exception:
        sents = _split_sentences(text)
        return " ".join(sents[:n_sentences])


def get_key_sentences(text, n=4):
    if not text or len(text.split()) < 20:
        return [text] if text else []

    try:
        parser = PlaintextParser.from_string(text, Tokenizer(LANG))
        stemmer = Stemmer(LANG)
        summ = LexRankSummarizer(stemmer)
        summ.stop_words = get_stop_words(LANG)
        return [str(s) for s in summ(parser.document, n)]
    except Exception:
        return _split_sentences(text)[:n]
