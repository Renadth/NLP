# Bayan Bilingual Semantic Search

## Model
paraphrase-multilingual-MiniLM-L12-v2 + mmarco-mMiniLMv2-L12-H384-v1

## Task
Bilingual case retrieval with bi-encoder retrieval and cross-encoder reranking.

## Evidence
Recall@10 without reranking: 0.0308
MRR@10 without reranking: 0.0049
Recall@10 with reranking: 0.0077
MRR@10 with reranking: 0.0019
No-answer correctness: 20/20

## Known limitations
Exact-ID retrieval metrics are strongly affected by duplicate synthetic cases and inconsistent relevance labelling.
Reranking reduced the reported retrieval metrics on this benchmark and should not be claimed as an improvement.
The system should not be approved for production retrieval based on this benchmark alone.
