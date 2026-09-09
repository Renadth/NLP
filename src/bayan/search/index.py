"""Lab 5: versioned FAISS index build."""

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREPROC_VERSION = "bayan_ar_v1"


def build_index(*args, **kwargs):
    prefix = kwargs.get("prefix")
    limit = kwargs.get("limit")

    if prefix is None:
        raise ValueError("build_index() requires prefix=...")

    if limit is None:
        limit = 20000

    prefix = Path(prefix)

    data_path = Path("data/search/bayan_cases.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Case corpus not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    if limit > 0:
        df = df.head(limit).copy()

    if df.empty:
        raise ValueError("Case corpus is empty.")

    required_columns = {
        "case_id",
        "lang",
        "topic",
        "case_text",
        "resolution",
        "status",
        "closed_at",
        "synthetic",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    texts = (
        df["case_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    model = SentenceTransformer(MODEL_NAME)

    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    vectors = np.asarray(
        vectors,
        dtype="float32",
    )

    # Required by the Lab 5 contract:
    # L2-normalise vectors before indexing.
    faiss.normalize_L2(vectors)

    dim = int(vectors.shape[1])

    # Inner product on unit-normalised vectors == cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_path = Path(f"{prefix}.faiss")
    metadata_path = Path(f"{prefix}_metadata.jsonl")
    manifest_path = Path(f"{prefix}_manifest.json")

    faiss.write_index(
        index,
        str(index_path),
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in df.to_dict("records"):
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    manifest = {
        "model": MODEL_NAME,
        "preproc_version": PREPROC_VERSION,
        "n_vectors": int(index.ntotal),
        "dim": dim,
        "metric": "inner_product",
        "normalised": True,
        "corpus": str(data_path),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Indexed {index.ntotal} cases "
        f"with dimension {dim}."
    )
    print(f"FAISS index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_path}")

    return {
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }