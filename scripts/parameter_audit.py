"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "FFN": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, parameter in model.named_parameters():
        count = parameter.numel()

        if "embeddings" in name:
            buckets["embeddings"] += count

        elif "attention" in name:
            buckets["attention"] += count

        elif "intermediate" in name or (
            ".output.dense" in name
            and "attention" not in name
        ):
            buckets["FFN"] += count

        elif "LayerNorm" in name or "layernorm" in name.lower():
            buckets["norms"] += count

        elif "pooler" in name:
            buckets["pooler"] += count

        else:
            buckets["other"] += count

    buckets["total"] = sum(buckets.values())

    return buckets


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print("\n" + "=" * 60)
        print(ckpt)
        print("=" * 60)

        result = audit(ckpt)

        for bucket, count in result.items():
            print(f"{bucket:12}: {count:,}")