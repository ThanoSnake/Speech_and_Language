import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.load_datasets import load_MR

# ============================================================
# Config
# ============================================================
MODEL_NAME = "textattack/bert-base-uncased-SST-2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
WARMUP_RATIO = 0.1

# ============================================================
# Data
# ============================================================
X_train, y_train, X_test, y_test = load_MR()

le = LabelEncoder()
le.fit(list(set(y_train)))
y_train = le.transform(y_train)
y_test = le.transform(y_test)
n_classes = len(le.classes_)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )


def prepare_dataset(X, y):
    return Dataset.from_dict({"text": list(X), "label": list(y)})


train_set = prepare_dataset(X_train, y_train).map(tokenize, batched=True)
test_set = prepare_dataset(X_test, y_test).map(tokenize, batched=True)


# ============================================================
# Metrics
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
    }


# ============================================================
# Model & Trainer
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=n_classes
)

for param in model.parameters():
    param.requires_grad = True

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=0,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=test_set,
    compute_metrics=compute_metrics,
)


# ============================================================
# run_epochs
# ============================================================
def run_epochs(n):
    """Train for n additional epochs, resuming from the last checkpoint."""
    trainer.args.num_train_epochs += n

    last_ckpt = None
    if os.path.isdir(OUTPUT_DIR):
        last_ckpt = get_last_checkpoint(OUTPUT_DIR)

    trainer.train(resume_from_checkpoint=last_ckpt)
    results = trainer.evaluate()

    print(f"\n{'=' * 40}")
    print(f"Results after {trainer.args.num_train_epochs:.0f} total epochs:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"{'=' * 40}")

    return results


if __name__ == "__main__":
    run_epochs(6)
