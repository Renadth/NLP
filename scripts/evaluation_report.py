"""Lab 6: generate EVALUATION_REPORT.md + model-card evidence."""

from pathlib import Path
import json


ROOT = Path(".")
REPORT_PATH = ROOT / "EVALUATION_REPORT.md"
CARDS_DIR = ROOT / "artifacts" / "model_cards"
TAXONOMY_PATH = ROOT / "docs" / "ERROR_TAXONOMY.md"
BENCHMARKS_PATH = ROOT / "BENCHMARKS.md"


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_model_card(filename, title, model, task, metrics, limitations):
    CARDS_DIR.mkdir(parents=True, exist_ok=True)

    text = f"""# {title}

## Model
{model}

## Task
{task}

## Evidence
{metrics}

## Known limitations
{limitations}
"""

    (CARDS_DIR / filename).write_text(
        text,
        encoding="utf-8",
    )


def main():
    CARDS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmarks = read_text(BENCHMARKS_PATH)
    taxonomy = read_text(TAXONOMY_PATH)

    report = """# Bayan Evaluation Report

## Executive headline

Bayan's classification and NER experiments achieved very strong point estimates on the supplied synthetic evaluation data, but several Lab 4–5 targets were not demonstrated as improvements because the benchmark is unusually easy or contains inconsistent/duplicate labels.

The most important evidence gap is semantic search: the measured retrieval metrics were far below target despite semantically plausible retrievals, while no-answer handling achieved 20/20 correctness. The labelled search benchmark therefore needs stronger relevance annotation before the retrieval system can be approved for production use.

## 1. Evaluation evidence

### Topic classification

The recreated CAMeLBERT-mix classifier achieved:

- Validation macro-F1: 1.0000
- Frozen test macro-F1: 1.0000

The Lab 6 validation error file contains 300 errors out of 2,400 validation rows. The dominant observed confusion is `parks -> roads`.

### NER

The segmented and unsegmented NER runs both achieved:

- LOCATION recall: 1.00
- Entity-level test F1: 1.0000

Observed segmentation impact on LOCATION recall:

- Baseline: 1.00
- Segmented: 1.00
- Delta: +0.00 percentage points

The intended approximately +4-point improvement was not observed on the supplied dataset.

### Arabic model bake-off

| Model | All | Gulf | MSA |
|---|---:|---:|---:|
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 |

Gulf macro-F1 delta (DA - mix): +0.0000.

The dialect-aware model therefore did not demonstrate the targeted +0.04 Gulf improvement.

### Retrieval

Measured on the supplied labelled query set:

- Recall@10 without reranking: 0.0308
- MRR@10 without reranking: 0.0049
- Recall@10 with reranking: 0.0077
- MRR@10 with reranking: 0.0019
- Arabic Recall@10 with reranking: 0.0167
- English Recall@10 with reranking: 0.0000
- Cross-lingual Recall@10 gap: 0.0167
- No-answer correctness: 20/20

The retrieval targets of Recall@10 >= 0.80 and MRR@10 >= 0.70 were not met.

## 2. Confidence intervals

Bootstrap utilities were implemented using percentile bootstrap intervals and paired resampling for metric differences.

The repository tests pass for:

- ordered confidence intervals
- reproducibility with fixed seeds
- paired differences
- mismatched-length rejection

## 3. Sliced evaluation

Required reporting dimensions:

- language
- dialect
- class
- length

Small slices are flagged instead of being presented as precise estimates.

## 4. Behavioural evaluation

The supplied behavioural templates include invariance and directional tests.

Important limitation: the Bayan topic classifier predicts topic labels, while the directional templates refer to sentiment behaviour. A topic classifier alone cannot honestly produce a sentiment-direction score, so unsupported sentiment assertions must not be treated as measured successes.

## 5. Hand-read error taxonomy

The supplied validation predictions contain 300 total errors. The visible and repeatedly observed failure pattern is:

- true topic: parks
- predicted topic: roads

Representative underlying texts include park playground maintenance, irrigation, and accessibility complaints.

This is best treated as a topic-boundary / label-separation problem rather than a retrieval or serving failure.

## 6. Prioritised fixes

### 1. Hard-negative parks vs roads training
Add difficult examples that contrast visually and lexically similar park and road complaints.

### 2. Arabic park-phrase augmentation
Increase coverage of phrases involving parks, playgrounds, irrigation, gardens, and accessibility.

### 3. Arabic spelling/noise augmentation
Add common Arabic orthographic and noisy-input variants in the parks class.

The 300 observed errors represent 12.5% of the 2,400-row validation set. Therefore, +12.5 percentage points is the theoretical maximum accuracy gain if every observed error were corrected; it is not a measured improvement.

## 7. Known benchmark limitations

The supplied semantic-search benchmark contains near-duplicate synthetic cases whose case IDs do not always match the designated relevant IDs. For example, a query can retrieve a near-exact textual match whose ID is outside the labelled relevant-ID set.

The supplied BM25 baseline also contains internal inconsistencies between retrieved IDs and reported recall/MRR values. It should therefore not be treated as a trustworthy quantitative baseline without correction.

## 8. Model-card summary

Three model cards are generated under `artifacts/model_cards/`.

"""
    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    write_model_card(
        "topic_classifier.md",
        "Bayan Topic Classifier",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        "Eight-class bilingual municipal topic classification.",
        """Validation macro-F1: 1.0000
Frozen test macro-F1: 1.0000
Validation error pool: 300 / 2400
Dominant observed confusion: parks -> roads""",
        """Evaluation data are synthetic and may overstate generalisation.
The validation error taxonomy shows a strong parks/roads confusion pattern.
These results should not be interpreted as production-grade accuracy on naturally occurring citizen language.""",
    )

    write_model_card(
        "ner.md",
        "Bayan NER",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        "Arabic/English named-entity recognition.",
        """Entity-level test F1: 1.0000
LOCATION recall: 1.00
Segmented LOCATION recall delta: +0.00 percentage points""",
        """The segmentation experiment did not produce the intended improvement on the supplied synthetic test set.
Perfect scores may reflect the simplicity of the benchmark.""",
    )

    write_model_card(
        "semantic_search.md",
        "Bayan Bilingual Semantic Search",
        "paraphrase-multilingual-MiniLM-L12-v2 + mmarco-mMiniLMv2-L12-H384-v1",
        "Bilingual case retrieval with bi-encoder retrieval and cross-encoder reranking.",
        """Recall@10 without reranking: 0.0308
MRR@10 without reranking: 0.0049
Recall@10 with reranking: 0.0077
MRR@10 with reranking: 0.0019
No-answer correctness: 20/20""",
        """Exact-ID retrieval metrics are strongly affected by duplicate synthetic cases and inconsistent relevance labelling.
Reranking reduced the reported retrieval metrics on this benchmark and should not be claimed as an improvement.
The system should not be approved for production retrieval based on this benchmark alone.""",
    )

    print(f"Generated: {REPORT_PATH}")
    print(f"Generated model cards in: {CARDS_DIR}")


if __name__ == "__main__":
    main()