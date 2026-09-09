"""Lab 3B: fine-tune token classification with correct NER alignment."""

import argparse
from pathlib import Path

import numpy as np
from seqeval.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from bayan.models.ner import align_labels

CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"

LABELS = [
    "O",
    "B-DATE",
    "B-LOCATION",
    "B-REFERENCE",
    "B-SERVICE",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="artifacts/ner",
        help="Where to save the trained NER artefact.",
    )
    return parser.parse_args()


def read_conll(path):
    sentences = []
    tokens = []
    labels = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if tokens:
                    sentences.append(
                        {
                            "tokens": tokens,
                            "labels": labels,
                        }
                    )
                    tokens = []
                    labels = []
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            tokens.append(" ".join(parts[:-1]))
            labels.append(parts[-1])

    if tokens:
        sentences.append(
            {
                "tokens": tokens,
                "labels": labels,
            }
        )

    return sentences


class NERDataset:

    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        encoding = self.tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=128,
        )

        word_ids = encoding.word_ids()

        word_labels = [
            LABEL2ID[label]
            for label in example["labels"]
        ]

        encoding["labels"] = align_labels(
            word_ids,
            word_labels,
        )

        return encoding

def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction

    predictions = np.argmax(predictions, axis=-1)

    true_predictions = []
    true_labels = []

    for prediction, label_ids in zip(predictions, labels):
        sentence_predictions = []
        sentence_labels = []

        for pred_id, label_id in zip(prediction, label_ids):
            if label_id == -100:
                continue

            sentence_predictions.append(ID2LABEL[int(pred_id)])
            sentence_labels.append(ID2LABEL[int(label_id)])

        true_predictions.append(sentence_predictions)
        true_labels.append(sentence_labels)

    return {
        "entity_f1": f1_score(
            true_labels,
            true_predictions,
        )
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/models/bayan_ner.conll")

    if not data_path.exists():
        raise FileNotFoundError(
            f"NER data not found: {data_path}"
        )

    print(f"Loading NER data from: {data_path}")

    examples = read_conll(data_path)

    print(f"Total sentences: {len(examples)}")

    train_examples, temp_examples = train_test_split(
        examples,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    validation_examples, test_examples = train_test_split(
        temp_examples,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    print(f"Train sentences: {len(train_examples)}")
    print(f"Validation sentences: {len(validation_examples)}")
    print(f"Test sentences: {len(test_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    train_dataset = NERDataset(
        train_examples,
        tokenizer,
    )

    validation_dataset = NERDataset(
        validation_examples,
        tokenizer,
    )

    test_dataset = NERDataset(
        test_examples,
        tokenizer,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        fp16=True,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting NER training...")
    trainer.train()

    print("\nValidation results:")
    validation_results = trainer.evaluate(
        eval_dataset=validation_dataset
    )
    print(validation_results)

    print("\nFrozen test results:")
    test_results = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )
    print(test_results)

    predictions = trainer.predict(test_dataset)

    predicted_ids = np.argmax(
        predictions.predictions,
        axis=-1,
    )

    true_predictions = []
    true_labels = []

    for prediction, label_ids in zip(
            predicted_ids,
            predictions.label_ids,
    ):
        sentence_predictions = []
        sentence_labels = []

        for pred_id, label_id in zip(
                prediction,
                label_ids,
        ):
            if label_id == -100:
                continue

            sentence_predictions.append(
                ID2LABEL[int(pred_id)]
            )
            sentence_labels.append(
                ID2LABEL[int(label_id)]
            )

        true_predictions.append(sentence_predictions)
        true_labels.append(sentence_labels)

    test_f1 = f1_score(
        true_labels,
        true_predictions,
    )

    print(f"\nFrozen test entity-level F1: {test_f1:.4f}")
    print("\nEntity-level classification report:")
    print(
        classification_report(
            true_labels,
            true_predictions,
        )
    )

    model.save_pretrained(
        output_dir,
        safe_serialization=False,
    )
    tokenizer.save_pretrained(output_dir)

    print(f"\nNER artefact saved to: {output_dir}")
    print(f"Frozen test entity-level F1: {test_f1:.4f}")

if __name__ == "__main__":
        main()
        



