"""Lab 3 starter: extractive QA post-processing."""


def best_span(
    start_logits,
    end_logits,
    offsets,
    *,
    null_score,
    null_threshold,
    max_answer_len=30,
    top_k=20,
):
    candidates = []

    for start_idx, start_score in enumerate(start_logits):
        start_offset = offsets[start_idx]

        if start_offset is None:
            continue

        for end_idx in range(
            start_idx,
            min(len(end_logits), start_idx + max_answer_len),
        ):
            end_offset = offsets[end_idx]

            if end_offset is None:
                continue

            if end_offset[1] <= start_offset[0]:
                continue

            score = float(start_score + end_logits[end_idx])

            candidates.append(
                {
                    "start": start_idx,
                    "end": end_idx,
                    "score": score,
                    "offset": (
                        start_offset[0],
                        end_offset[1],
                    ),
                }
            )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    candidates = candidates[:top_k]

    if not candidates:
        return {
            "answer": None,
            "start": None,
            "end": None,
            "score": float(null_score),
            "offset": None,
        }

    best = candidates[0]

    # Return null only when the null score beats the best span
    # by more than the allowed threshold.
    if null_score - best["score"] > null_threshold:
        return {
            "answer": None,
            "start": None,
            "end": None,
            "score": float(null_score),
            "offset": None,
        }

    return {
        "answer": best["offset"],
        "start": best["start"],
        "end": best["end"],
        "score": best["score"],
        "offset": best["offset"],
    }