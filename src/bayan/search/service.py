"""Lab 5: two-stage bilingual case search."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from bayan.preprocessing.arabic import ArabicProfile, normalize_arabic


BI_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
PREPROC_VERSION = "bayan_ar_v1"


class CaseSearch:
    def __init__(self, prefix: str):
        self.prefix = Path(prefix)

        self.index_path = Path(f"{self.prefix}.faiss")
        self.metadata_path = Path(f"{self.prefix}_metadata.jsonl")
        self.manifest_path = Path(f"{self.prefix}_manifest.json")

        for path in (
            self.index_path,
            self.metadata_path,
            self.manifest_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required search artefact not found: {path}"
                )

        self.manifest = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8"
            )
        )

        required = [
            "model",
            "preproc_version",
            "n_vectors",
            "dim",
        ]

        missing = [
            key
            for key in required
            if key not in self.manifest
        ]

        if missing:
            raise ValueError(
                f"Manifest missing required keys: {missing}"
            )

        if self.manifest["preproc_version"] != PREPROC_VERSION:
            raise ValueError(
                "Search preprocessing version does not match "
                f"expected {PREPROC_VERSION!r}."
            )

        if self.manifest["model"] != BI_ENCODER:
            raise ValueError(
                "Index model does not match the configured "
                f"bi-encoder: {BI_ENCODER!r}."
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        if self.index.ntotal != self.manifest["n_vectors"]:
            raise ValueError(
                "Manifest vector count does not match FAISS index."
            )

        if self.index.d != self.manifest["dim"]:
            raise ValueError(
                "Manifest dimension does not match FAISS index."
            )

        self.metadata = []

        with self.metadata_path.open(
            encoding="utf-8"
        ) as f:
            for line in f:
                if line.strip():
                    self.metadata.append(
                        json.loads(line)
                    )

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                "Metadata count does not match FAISS index."
            )

        self.encoder = SentenceTransformer(
            BI_ENCODER
        )

        self.reranker = CrossEncoder(
            CROSS_ENCODER
        )

        self.arabic_profile = ArabicProfile(
            name=PREPROC_VERSION,
            dediacritize=True,
        )

    def _normalize_query(self, query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            return ""

        # Preserve English while applying the course Arabic profile
        # to Arabic text.
        if any("\u0600" <= ch <= "\u06ff" for ch in query):
            query = normalize_arabic(
                query,
                self.arabic_profile,
            )

        return query

    def search(
        self,
        query: str,
        k: int = 5,
        candidates: int = 50,
        min_score: float = 0.25,
    ):
        query = self._normalize_query(query)

        if not query:
            return []

        if k <= 0:
            return []

        if candidates <= 0:
            return []

        candidates = max(
            candidates,
            k,
        )

        candidates = min(
            candidates,
            self.index.ntotal,
        )

        # Stage 1: bi-encoder retrieval.
        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(
            query_vector,
            candidates,
        )

        retrieved = []

        for score, idx in zip(
            scores[0],
            indices[0],
        ):
            if idx < 0:
                continue

            metadata = dict(
                self.metadata[idx]
            )

            metadata["bi_score"] = float(score)
            metadata["_index"] = int(idx)

            retrieved.append(metadata)

        if not retrieved:
            return []

        # Stage 2: cross-encoder reranking.
        pairs = [
            (
                query,
                item["case_text"],
            )
            for item in retrieved
        ]

        rerank_scores = self.reranker.predict(
            pairs
        )

        for item, score in zip(
            retrieved,
            rerank_scores,
        ):
            item["score"] = float(score)

        retrieved.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        # Honest no-result behaviour.
        filtered = [
            item
            for item in retrieved
            if item["score"] >= min_score
        ]

        results = []

        for item in filtered[:k]:
            clean_item = {
                key: value
                for key, value in item.items()
                if not key.startswith("_")
            }

            results.append(clean_item)

        return results