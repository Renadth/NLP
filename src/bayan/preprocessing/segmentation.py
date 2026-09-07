"""Lab 1 starter: sentence segmentation."""

import spacy

from bayan.preprocessing.core import preprocess


def build_pipeline():
    """Build a lightweight, language-agnostic sentence segmentation pipeline."""
    nlp = spacy.blank("xx")  # multilingual, rule-based tokenizer
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    """Preprocess raw text, then return non-empty sentence strings."""
    cleaned = preprocess(raw)
    doc = nlp(cleaned)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]