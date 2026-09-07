"""Lab 1 starter: audit four tokenizer candidates on Bayan AR/EN text."""
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}

DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """Total subword pieces / whitespace words, across all texts."""
    total_subwords = 0
    total_words = 0
    for text in texts:
        total_words += len(text.split())
        total_subwords += len(tokenizer.tokenize(text))
    if total_words == 0:
        return 0.0
    return total_subwords / total_words


def main():
    df = pd.read_csv(DATA, sep=None, engine="python")

    ar_texts = df.loc[df["lang"] == "ar", "text"].dropna().tolist()
    en_texts = df.loc[df["lang"] == "en", "text"].dropna().tolist()

    print(f"Loaded {len(ar_texts)} AR rows, {len(en_texts)} EN rows\n")

    results = []
    for checkpoint, short_name in CANDIDATES.items():
        print(f"Loading {short_name} ({checkpoint}) ...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        ar_fert = fertility(tokenizer, ar_texts)
        en_fert = fertility(tokenizer, en_texts)

        ar_lengths = [len(tokenizer.tokenize(t)) for t in ar_texts]
        en_lengths = [len(tokenizer.tokenize(t)) for t in en_texts]

        ar_p95 = float(np.percentile(ar_lengths, 95)) if ar_lengths else 0.0
        en_p95 = float(np.percentile(en_lengths, 95)) if en_lengths else 0.0

        results.append({
            "tokenizer": short_name,
            "checkpoint": checkpoint,
            "ar_fertility": round(ar_fert, 3),
            "en_fertility": round(en_fert, 3),
            "ar_p95_len": ar_p95,
            "en_p95_len": en_p95,
        })

    print("\n" + "=" * 90)
    header = f"{'Tokenizer':<12}{'AR fertility':<15}{'EN fertility':<15}{'AR p95 len':<13}{'EN p95 len':<13}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['tokenizer']:<12}{r['ar_fertility']:<15}{r['en_fertility']:<15}{r['ar_p95_len']:<13}{r['en_p95_len']:<13}")
    print("=" * 90)
    print("\nCopy the table above into BENCHMARKS.md.")


if __name__ == "__main__":
    main()