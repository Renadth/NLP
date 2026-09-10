# Bayan — Bilingual Citizen-Feedback NLP

Bayan is an end-to-end NLP system for analyzing **Arabic and English citizen feedback**. It covers preprocessing, classification, named-entity recognition, question answering, semantic search, evaluation, and production-oriented serving.

## Features

* Arabic normalization and segmentation
* Bilingual topic classification with CAMeLBERT
* Named-Entity Recognition (NER)
* Extractive Question Answering
* FAISS-based bilingual semantic search
* Bootstrap confidence intervals and sliced evaluation
* Behavioural testing and error analysis
* ONNX/INT8 inference optimization
* FastAPI serving

## Pipeline

```text
Citizen Feedback
      ↓
Preprocessing
      ↓
Classification ── NER ── QA
      ↓
Semantic Search
      ↓
Evaluation
      ↓
Optimized Serving
```

## Key Results

| Component                    |           Result |
| ---------------------------- | ---------------: |
| Topic classification         |  Macro-F1 1.0000 |
| NER                          | Entity F1 1.0000 |
| QA smoke test                |     12/12 passed |
| CPU dynamic-padding speed-up |            8.70× |
| Search Recall@10             |           0.0077 |

The supplied datasets are largely synthetic, so perfect scores should not be interpreted as production-level generalization.

## Technologies

Python · PyTorch · Transformers · CAMeL Tools · FAISS · Sentence Transformers · FastAPI · ONNX Runtime · pytest

## Project Status

Labs 3–7 cover modeling, search, evaluation, and inference optimization. Current work focuses on ONNX/INT8 optimization and FastAPI serving.

**Principle:** Measure first, optimize second, and report limitations honestly.

## SDAIA Academy
https://github.com/SDAIAAcademy
