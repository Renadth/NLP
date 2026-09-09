"""Lab 5: labelled-query retrieval evaluation."""

import json
from pathlib import Path

import faiss
import numpy as np


INDEX_PREFIX = "artifacts/case_index_v1"
QUERIES_PATH = "data/search/bayan_queries.jsonl"
K = 10


def load_queries(path):
    queries = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    return queries


def recall_at_k(results, relevant_ids, k=10):
    retrieved = [x["case_id"] for x in results[:k]]
    return float(any(case_id in relevant_ids for case_id in retrieved))


def mrr_at_k(results, relevant_ids, k=10):
    for rank, item in enumerate(results[:k], start=1):
        if item["case_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0


def retrieve_without_reranking(searcher, query, k=10):
    query = searcher._normalize_query(query)

    if not query:
        return []

    vector = searcher.encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    faiss.normalize_L2(vector)

    scores, indices = searcher.index.search(
        vector,
        k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = dict(searcher.metadata[idx])
        item["score"] = float(score)

        results.append(item)

    return results


def evaluate_slice(queries, searcher):
    no_rerank_recalls = []
    no_rerank_mrrs = []

    rerank_recalls = []
    rerank_mrrs = []

    no_answer_correct = 0
    no_answer_total = 0

    for item in queries:
        query = item["query"]
        relevant_ids = set(item["relevant_case_ids"])

        # Stage 1: bi-encoder only.
        initial = retrieve_without_reranking(
            searcher,
            query,
            k=K,
        )

        no_rerank_recalls.append(
            recall_at_k(
                initial,
                relevant_ids,
                K,
            )
        )

        no_rerank_mrrs.append(
            mrr_at_k(
                initial,
                relevant_ids,
                K,
            )
        )

        # Stage 2: complete service with cross-encoder.
        reranked = searcher.search(
            query,
            k=K,
            candidates=50,
            min_score=-100.0,
        )

        rerank_recalls.append(
            recall_at_k(
                reranked,
                relevant_ids,
                K,
            )
        )

        rerank_mrrs.append(
            mrr_at_k(
                reranked,
                relevant_ids,
                K,
            )
        )

        if item["no_answer"]:
            no_answer_total += 1

            strict_results = searcher.search(
                query,
                k=K,
                candidates=50,
                min_score=0.25,
            )

            if not strict_results:
                no_answer_correct += 1

    return {
        "recall_no_rerank": float(
            np.mean(no_rerank_recalls)
        ),
        "mrr_no_rerank": float(
            np.mean(no_rerank_mrrs)
        ),
        "recall_rerank": float(
            np.mean(rerank_recalls)
        ),
        "mrr_rerank": float(
            np.mean(rerank_mrrs)
        ),
        "no_answer_correct": no_answer_correct,
        "no_answer_total": no_answer_total,
    }


def tune_no_answer_threshold(no_answer_queries, searcher):
    print("\nNo-answer threshold sweep")

    thresholds = [
        0.10,
        0.25,
        0.50,
        0.75,
        1.00,
        2.00,
        3.00,
        4.00,
        5.00,
    ]

    best_threshold = None
    best_correct = -1

    for threshold in thresholds:
        correct = 0

        for item in no_answer_queries:
            results = searcher.search(
                item["query"],
                k=5,
                candidates=50,
                min_score=threshold,
            )

            if not results:
                correct += 1

        print(
            f"min_score={threshold:.2f}: "
            f"{correct}/{len(no_answer_queries)}"
        )

        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold

    return best_threshold, best_correct


def main():
    from bayan.search.service import CaseSearch

    if not Path(QUERIES_PATH).exists():
        raise FileNotFoundError(
            f"Query file not found: {QUERIES_PATH}"
        )

    manifest_path = Path(
        f"{INDEX_PREFIX}_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Index manifest not found: {manifest_path}"
        )

    queries = load_queries(QUERIES_PATH)

    print("=" * 60)
    print("LAB 5 — RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Total queries: {len(queries)}")
    print(
        f"Arabic queries: "
        f"{sum(x['lang'] == 'ar' for x in queries)}"
    )
    print(
        f"English queries: "
        f"{sum(x['lang'] == 'en' for x in queries)}"
    )
    print(
        f"No-answer queries: "
        f"{sum(x['no_answer'] for x in queries)}"
    )

    searcher = CaseSearch(INDEX_PREFIX)

    answerable = [
        x for x in queries
        if not x["no_answer"]
    ]

    results = evaluate_slice(
        answerable,
        searcher,
    )

    print("\nOverall answerable results")
    print(
        f"Recall@10 without reranking: "
        f"{results['recall_no_rerank']:.4f}"
    )
    print(
        f"MRR@10 without reranking: "
        f"{results['mrr_no_rerank']:.4f}"
    )
    print(
        f"Recall@10 with reranking: "
        f"{results['recall_rerank']:.4f}"
    )
    print(
        f"MRR@10 with reranking: "
        f"{results['mrr_rerank']:.4f}"
    )

    # Language slices.
    ar_queries = [
        x for x in answerable
        if x["lang"] == "ar"
    ]

    en_queries = [
        x for x in answerable
        if x["lang"] == "en"
    ]

    ar_results = evaluate_slice(
        ar_queries,
        searcher,
    )

    en_results = evaluate_slice(
        en_queries,
        searcher,
    )

    print("\nLanguage slices")
    print(
        f"Arabic Recall@10 with reranking: "
        f"{ar_results['recall_rerank']:.4f}"
    )
    print(
        f"English Recall@10 with reranking: "
        f"{en_results['recall_rerank']:.4f}"
    )

    cross_lingual_gap = abs(
        ar_results["recall_rerank"]
        - en_results["recall_rerank"]
    )

    print(
        f"Cross-lingual Recall@10 gap: "
        f"{cross_lingual_gap:.4f}"
    )

    # No-answer behaviour.
    no_answer_queries = [
        x for x in queries
        if x["no_answer"]
    ]

    if no_answer_queries:
        best_threshold, best_correct = (
            tune_no_answer_threshold(
                no_answer_queries,
                searcher,
            )
        )

        print(
            f"\nBest no-answer threshold: "
            f"{best_threshold:.2f}"
        )
        print(
            f"No-answer correctness: "
            f"{best_correct}/{len(no_answer_queries)}"
        )
    else:
        print(
            "\nNo-answer correctness: "
            "0/0 — no no-answer queries supplied."
        )

    print("\nTargets")
    print(
        "Recall@10 >= 0.80: "
        + (
            "MET"
            if results["recall_rerank"] >= 0.80
            else "NOT MET"
        )
    )

    print(
        "MRR@10 >= 0.70: "
        + (
            "MET"
            if results["mrr_rerank"] >= 0.70
            else "NOT MET"
        )
    )

    if no_answer_queries:
        print(
            "No-answer >= 17/20: "
            + (
                "MET"
                if best_correct >= 17
                else "NOT MET"
            )
        )


if __name__ == "__main__":
    main()