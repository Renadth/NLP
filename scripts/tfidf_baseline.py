"""Lab 3A: TF-IDF + LinearSVC baseline."""

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


# Allow imports from src/ when running:
# python scripts/tfidf_baseline.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bayan.models.data import build_topic_dataset


def main():
    print("=" * 60)
    print("LAB 3A — TF-IDF + LinearSVC BASELINE")
    print("=" * 60)

    # Load supplied deterministic splits.
    splits = build_topic_dataset()

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    print("\nDataset sizes:")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    print(f"\nNumber of topics: {train['topic'].nunique()}")

    # TF-IDF baseline.
    #
    # Character + word information can both be useful for Arabic text,
    # but this baseline starts with word n-grams.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )

    print("\nFitting TF-IDF vectorizer...")

    X_train = vectorizer.fit_transform(train["text"])
    X_validation = vectorizer.transform(validation["text"])

    y_train = train["topic"]
    y_validation = validation["topic"]

    print(f"TF-IDF feature matrix: {X_train.shape}")

    # Train linear classifier.
    classifier = LinearSVC()

    print("\nTraining LinearSVC...")
    classifier.fit(X_train, y_train)

    # Evaluate on validation split.
    predictions = classifier.predict(X_validation)

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Validation Macro-F1: {macro_f1:.4f}")

    print("\nNOTE:")
    print(
        "The final test split remains frozen and is not used "
        "for baseline tuning."
    )


if __name__ == "__main__":
    main()