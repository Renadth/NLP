"""Lab 3B: run the 12-question QA smoke set."""

import json
from pathlib import Path

import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


CHECKPOINT = "distilbert-base-uncased-distilled-squad"


def load_smoke_set(path):
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    questions = []

    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                questions.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answer": (
                            None
                            if qa["is_impossible"]
                            else qa["answers"][0]["text"]
                        ),
                        "is_impossible": qa["is_impossible"],
                    }
                )

    return questions


def find_answer(
    question,
    context,
    tokenizer,
    model,
):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=384,
    )

    offset_mapping = inputs.pop("offset_mapping")[0].tolist()

    outputs = model(**inputs)

    start_logits = outputs.start_logits[0].detach().cpu().numpy()
    end_logits = outputs.end_logits[0].detach().cpu().numpy()

    offsets = [
        tuple(pair) if pair != [0, 0] else None
        for pair in offset_mapping
    ]

    null_score = float(start_logits[0] + end_logits[0])

    result = best_span(
        start_logits,
        end_logits,
        offsets,
        null_score=null_score,
        null_threshold=0.0,
    )

    if result["answer"] is None:
        return None

    start_char, end_char = result["answer"]

    return context[start_char:end_char]


def main():
    smoke_path = Path("data/eval/qa_smoke_set.json")

    if not smoke_path.exists():
        raise FileNotFoundError(
            f"QA smoke set not found: {smoke_path}"
        )

    questions = load_smoke_set(smoke_path)

    print(f"Loaded {len(questions)} smoke questions.")
    print(f"Checkpoint: {CHECKPOINT}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForQuestionAnswering.from_pretrained(CHECKPOINT)
    model.eval()

    answerable_total = 0
    answerable_correct = 0
    null_total = 0
    null_correct = 0

    for item in questions:
        predicted = find_answer(
            item["question"],
            item["context"],
            tokenizer,
            model,
        )

        expected = item["answer"]

        if item["is_impossible"]:
            null_total += 1

            if predicted is None:
                null_correct += 1
                status = "PASS"
            else:
                status = "FAIL"
        else:
            answerable_total += 1

            if predicted == expected:
                answerable_correct += 1
                status = "PASS"
            else:
                status = "FAIL"

        print(
            f"{item['id']}: {status} | "
            f"predicted={predicted!r} | expected={expected!r}"
        )

    print()
    print(
        f"Answerable: {answerable_correct}/{answerable_total}"
    )
    print(
        f"Unanswerable: {null_correct}/{null_total}"
    )


if __name__ == "__main__":
    main()