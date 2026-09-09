# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
## Tokenizer Audit Results

| Tokenizer  | AR fertility | EN fertility | AR p95 len | EN p95 len |
|------------|-------------|-------------|-----------|-----------|
| mBERT      | 2.153       | 1.510       | 25.0      | 23.0      |
| XLM-R      | 1.672       | 1.434       | 19.0      | 21.0      |
| CAMeLBERT  | 1.405       | 2.705       | 18.0      | 36.0      |
| DistilBERT | 4.527       | 1.298       | 45.0      | 19.0      |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 2 

| Component / Finding | bert-base-multilingual-cased | CAMeL-Lab/bert-base-arabic-camelbert-mix |
|---|---:|---:|
| Embeddings | 92,208,384 | 23,436,288 |
| Attention | 28,366,848 | 28,366,848 |
| FFN | 56,669,184 | 56,669,184 |
| Norms | 18,432 | 18,432 |
| Pooler | 590,592 | 590,592 |
| Other | 0 | 0 |
| **Total parameters** | **177,853,440** | **109,081,344** |

| Behaviour | Result |
|---|---|
| Numerical equivalence | True |
| Lower-triangular causal attention | True |
| Model family | Decoder-style causal attention |
| Adjacency-looking head | Detected |
| `[SEP]` sink behaviour | Detected |
| `[PAD]` leakage without mask | True |
| `[PAD]` leakage with correct mask | False |
| No `[PAD]` attention leakage after masking | True |

## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | 1.0000 | 1.0000 | N/A |
| Topic classifier (CAMeLBERT) | macro-F1 | 1.0000 | 1.0000 | 237.8243 s |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

### Lab 3A — Topic classifier
| NER | entity-F1 | 1.0000 | 1.0000 | 40.66 s |

- Checkpoint: `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- Train split: 9,592
- Validation split: 1,191
- Frozen test split: 1,217
- Frozen-test Macro-F1 improvement over TF-IDF baseline: **+0.0000**
- Target: baseline + 0.08
- Target status: **Not achievable on this dataset because the baseline Macro-F1 is already 1.0000**
- Dataset limitation: exact duplicate texts remain across the grouped train/validation data, contributing to the perfect baseline and Transformer scores.
- Classifier artefact: `/content/drive/MyDrive/SDA-AIE-211/artifacts/topic_classifier`

### Lab 3B — NER
- Checkpoint: `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- Data: `data/models/bayan_ner.conll`
- Split: 3,200 train / 400 validation / 400 test
- Entity-level F1: **1.0000** on validation and test
- Target: ≥ 0.80 — **met**
- Artefact: `artifacts/ner`
- Note: Results are on the supplied synthetic dataset and should not be interpreted as real-world NER performance.

### Lab 3B — Extractive QA
- Checkpoint: `distilbert-base-uncased-distilled-squad`
- Smoke set: 12 supplied questions
- Answerable: **12/12**
- Unanswerable: **0/0** (the supplied smoke set contains no unanswerable questions)
- Target in README: 9/9 answerable and 3/3 unanswerable
- Note: The supplied `qa_smoke_set.json` is inconsistent with the README target; all 12 questions are marked answerable.


### Lab 4 — NER segmentation impact

- Baseline (Day 2) LOCATION recall: 1.00
- Segmented LOCATION recall: 1.00
- LOCATION recall delta: **0.00 percentage points**
- Target: approximately +4 points
- Result: **Target not met on the supplied dataset**

### Lab 4 — Arabic model bake-off

| Model | All macro-F1 | Gulf macro-F1 | MSA macro-F1 |
|---|---:|---:|---:|
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 |

- Gulf macro-F1 delta (DA − mix): **+0.0000**
- Target: +0.04
- Result: **Target not met**


## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:

## TF-IDF + LinearSVC baseline
- Macro-F1 (frozen test split): 1.0000

- Macro-F1 (4 topics present in frozen test split): 1.0000
- Macro-F1 (forced across all 8 topics): 0.5000
- Note: test split contains only 4/8 topics (lighting, water, digital_services, parks) due to citizen-level grouping; roads/waste/billing/licensing citizens all landed in train.
