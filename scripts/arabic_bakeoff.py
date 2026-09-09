"""Lab 4: compare Arabic-centric checkpoints on all/Gulf/MSA slices."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


INCUMBENT_CHECKPOINT = "artifacts/topic_classifier"
DA_CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-da"

MAX_LENGTH = 128
OUTPUT_DIR = Path("artifacts/arabic_bakeoff_da")


class TopicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        )
    }


def predict_model(model, tokenizer, frame, label2id):
    dataset = TopicDataset(
        frame["text"].tolist(),
        [label2id[x] for x in frame["topic"]],
        tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="artifacts/arabic_bakeoff_eval",
            report_to="none",
            per_device_eval_batch_size=32,
        ),
    )

    prediction = trainer.predict(dataset)
    predicted = np.argmax(
        prediction.predictions,
        axis=-1,
    )

    return predicted


def slice_score(frame, predicted):
    true_labels = frame["topic"].tolist()
    topics = sorted(set(true_labels))

    true_ids = [topics.index(x) for x in true_labels]
    pred_ids = [topics.index(topics[p]) for p in predicted]

    return f1_score(
        true_ids,
        pred_ids,
        average="macro",
    )


def arabic_fertility(frame, tokenizer):
    total_subwords = 0
    total_words = 0

    for text in frame["text"].tolist():
        words = str(text).split()

        if not words:
            continue

        encoding = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
        )

        word_ids = encoding.word_ids()

        total_subwords += len(word_ids)
        total_words += len(words)

    if total_words == 0:
        return 0.0

    return total_subwords / total_words


def evaluate_model(model, tokenizer, frame, label2id):
    dataset = TopicDataset(
        frame["text"].tolist(),
        [label2id[x] for x in frame["topic"]],
        tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="artifacts/arabic_bakeoff_eval",
            report_to="none",
            per_device_eval_batch_size=32,
        ),
    )

    predictions = trainer.predict(dataset)
    predicted_ids = np.argmax(
        predictions.predictions,
        axis=-1,
    )

    true_ids = np.array(
        [label2id[x] for x in frame["topic"]]
    )

    results = {
        "all": f1_score(
            true_ids,
            predicted_ids,
            average="macro",
        )
    }

    for region in ["Gulf", "MSA"]:
        subset = frame[
            frame["dialect_region"] == region
        ].copy()

        if len(subset) == 0:
            results[region] = float("nan")
            continue

        subset_indices = subset.index.tolist()
        positions = [
            frame.index.get_loc(idx)
            for idx in subset_indices
        ]

        subset_true = true_ids[positions]
        subset_pred = predicted_ids[positions]

        results[region] = f1_score(
            subset_true,
            subset_pred,
            average="macro",
        )

    results["AR fertility"] = arabic_fertility(
        frame,
        tokenizer,
    )

    return results


def main():
    if not Path("data/raw/bayan_feedback.csv").exists():
        raise FileNotFoundError(
            "data/raw/bayan_feedback.csv was not found."
        )

    dataset = build_topic_dataset()

    train = dataset["train"].copy()
    test = dataset["test"].copy()

    # Restrict the bake-off to Arabic data.
    train_ar = train[
        train["lang"].str.lower() == "ar"
    ].copy()

    test_ar = test[
        test["lang"].str.lower() == "ar"
    ].copy()

    print("=" * 60)
    print("LAB 4 — ARABIC MODEL BAKE-OFF")
    print("=" * 60)
    print(f"Arabic train rows: {len(train_ar)}")
    print(f"Arabic test rows:  {len(test_ar)}")
    print()

    topics = sorted(
        set(train_ar["topic"]) | set(test_ar["topic"])
    )

    label2id = {
        topic: i
        for i, topic in enumerate(topics)
    }

    id2label = {
        i: topic
        for topic, i in label2id.items()
    }

    # ---------------------------------------------------------
    # 1. Day-2 incumbent: existing CAMeLBERT-mix artefact.
    # ---------------------------------------------------------
    incumbent_path = Path(INCUMBENT_CHECKPOINT)

    if not incumbent_path.exists():
        raise FileNotFoundError(
            f"Incumbent artefact not found: {incumbent_path}"
        )

    print("Loading incumbent CAMeLBERT-mix artefact...")

    incumbent_tokenizer = AutoTokenizer.from_pretrained(
        incumbent_path
    )

    incumbent_model = (
        AutoModelForSequenceClassification.from_pretrained(
            incumbent_path,
            num_labels=len(topics),
            label2id=label2id,
            id2label=id2label,
        )
    )

    incumbent_results = evaluate_model(
        incumbent_model,
        incumbent_tokenizer,
        test_ar,
        label2id,
    )

    # ---------------------------------------------------------
    # 2. CAMeLBERT-DA: fine-tune on Arabic train split.
    # ---------------------------------------------------------
    print()
    print("Loading CAMeLBERT-DA...")

    da_tokenizer = AutoTokenizer.from_pretrained(
        DA_CHECKPOINT
    )

    da_model = (
        AutoModelForSequenceClassification.from_pretrained(
            DA_CHECKPOINT,
            num_labels=len(topics),
            label2id=label2id,
            id2label=id2label,
        )
    )

    da_train_dataset = TopicDataset(
        train_ar["text"].tolist(),
        [label2id[x] for x in train_ar["topic"]],
        da_tokenizer,
    )

    da_eval_dataset = TopicDataset(
        test_ar["text"].tolist(),
        [label2id[x] for x in test_ar["topic"]],
        da_tokenizer,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    da_trainer = Trainer(
        model=da_model,
        args=training_args,
        train_dataset=da_train_dataset,
        eval_dataset=da_eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting CAMeLBERT-DA fine-tuning...")
    da_trainer.train()

    da_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=False,
    )

    da_tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    da_results = evaluate_model(
        da_model,
        da_tokenizer,
        test_ar,
        label2id,
    )

    # ---------------------------------------------------------
    # 3. Print results.
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("BAKE-OFF RESULTS")
    print("=" * 60)

    print(
        f"{'Model':<24}"
        f"{'All':>10}"
        f"{'Gulf':>10}"
        f"{'MSA':>10}"
        f"{'AR fertility':>15}"
    )

    print("-" * 69)

    for name, results in [
        ("CAMeLBERT-mix", incumbent_results),
        ("CAMeLBERT-DA", da_results),
    ]:
        print(
            f"{name:<24}"
            f"{results['all']:>10.4f}"
            f"{results['Gulf']:>10.4f}"
            f"{results['MSA']:>10.4f}"
            f"{results['AR fertility']:>15.3f}"
        )

    gulf_delta = (
        da_results["Gulf"]
        - incumbent_results["Gulf"]
    )

    print()
    print(
        f"CAMeLBERT-DA Gulf macro-F1 delta: "
        f"{gulf_delta:+.4f}"
    )

    if gulf_delta >= 0.04:
        print("Target: MET")
    else:
        print("Target: NOT MET")

    print()
    print(
        "Winner should be selected using Gulf-slice evidence, "
        "not aggregate F1 alone."
    )


if __name__ == "__main__":
    main()