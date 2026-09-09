"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
AutoModelForSequenceClassification,
AutoTokenizer,
Trainer,
TrainingArguments,
)

from bayan.models.data import build_topic_dataset

CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
MAX_LENGTH = 128

def parse_args():
parser = argparse.ArgumentParser()
parser.add_argument(
"--output-dir",
default="artifacts/topic_classifier",
help="Where to save the trained classifier artefact.",
)
return parser.parse_args()

class TopicDataset(torch.utils.data.Dataset):
def **init**(self, texts, labels, tokenizer):
self.encodings = tokenizer(
texts,
truncation=True,
padding="max_length",
max_length=MAX_LENGTH,
)
self.labels = labels

```
def __len__(self):
    return len(self.labels)

def __getitem__(self, idx):
    item = {
        key: torch.tensor(value[idx])
        for key, value in self.encodings.items()
    }
    item["labels"] = torch.tensor(self.labels[idx])
    return item
```

def compute_metrics(eval_pred):
logits, labels = eval_pred
predictions = np.argmax(logits, axis=-1)

```
return {
    "accuracy": accuracy_score(labels, predictions),
    "macro_f1": f1_score(labels, predictions, average="macro"),
}
```

def main():
args = parse_args()
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

```
print("=" * 60)
print("LAB 3A — CAMeLBERT TOPIC CLASSIFIER")
print("=" * 60)
print(f"Checkpoint: {CHECKPOINT}")
print(f"Output directory: {output_dir}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading grouped dataset...")
dataset = build_topic_dataset()

train_texts = list(dataset["train"]["text"])
valid_texts = list(dataset["validation"]["text"])
test_texts = list(dataset["test"]["text"])

topics = sorted(set(dataset["train"]["topic"]))
label2id = {topic: i for i, topic in enumerate(topics)}
id2label = {i: topic for topic, i in label2id.items()}

train_labels = [label2id[x] for x in dataset["train"]["topic"]]
valid_labels = [label2id[x] for x in dataset["validation"]["topic"]]
test_labels = [label2id[x] for x in dataset["test"]["topic"]]

print(f"Train: {len(train_texts)}")
print(f"Validation: {len(valid_texts)}")
print(f"Test: {len(test_texts)}")
print(f"Number of topics: {len(topics)}")

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

train_dataset = TopicDataset(train_texts, train_labels, tokenizer)
valid_dataset = TopicDataset(valid_texts, valid_labels, tokenizer)
test_dataset = TopicDataset(test_texts, test_labels, tokenizer)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=len(topics),
    label2id=label2id,
    id2label=id2label,
)

training_args = TrainingArguments(
    output_dir=str(output_dir),
    evaluation_strategy="epoch",
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

print("\n" + "=" * 60)
print("VALIDATION RESULT")
print("=" * 60)
validation_metrics = trainer.evaluate(valid_dataset)
print(validation_metrics)

print("\n" + "=" * 60)
print("FROZEN TEST RESULT")
print("=" * 60)
test_metrics = trainer.evaluate(test_dataset)
print(test_metrics)

print("\nSaving model and tokenizer...")
model.save_pretrained(output_dir, safe_serialization=False)
tokenizer.save_pretrained(output_dir)

print("\nSaved artefact:")
print(output_dir.resolve())

print("\nDone.")
```

if **name** == "**main**":
main()
