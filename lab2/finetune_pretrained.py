import os
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from datasets import Dataset
from utils.load_datasets import load_MR, load_Semeval2017A

RESULTS_DIR = "./results/finetuned"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Datasets & models
# ============================================================
#
# MR (movie reviews, binary): No movie-review-specific base models
# exist, but these are standard English models and MR is standard
# English text.
#
# Semeval2017A (Twitter, 3-class): Twitter-pretrained bases
# (cardiffnlp, vinai) should outperform the general bert-base-cased,
# demonstrating that domain-matched pretraining matters.
#
# All 6 bases are distinct. The MR models are the bases of the
# finetuned models used in Q6 (transfer_pretrained.py):
#   bert-base-uncased       -> textattack/bert-base-uncased-SST-2
#   distilbert-base-uncased -> distilbert-base-uncased-finetuned-sst-2-english
#   roberta-base            -> siebert/sentiment-roberta-large-english (large->base)
#
# The Semeval models include two Twitter-pretrained bases and one
# general baseline as a control:
#   cardiffnlp/twitter-roberta-base -> cardiffnlp/twitter-roberta-base-sentiment
#   vinai/bertweet-base             -> finiteautomata/bertweet-base-sentiment-analysis
#   bert-base-cased                 -> general baseline (no domain pretraining)

DATASETS = {
    "MR": {
        "loader": load_MR,
        "models": [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
        ],
    },
    "Semeval2017A": {
        "loader": load_Semeval2017A,
        "models": [
            "cardiffnlp/twitter-roberta-base",
            "vinai/bertweet-base",
            "bert-base-cased",
        ],
    },
}

NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
MAX_LENGTH = 128


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    n_classes = logits.shape[1]
    average = "binary" if n_classes == 2 else "macro"
    return {
        "accuracy": accuracy_score(labels, predictions),
        "recall": recall_score(labels, predictions, average=average),
        "f1": f1_score(labels, predictions, average=average),
    }


# Global tokenizer — updated in the loop for each model
tokenizer = None


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )


def prepare_dataset(X, y):
    return Dataset.from_dict({"text": list(X), "label": list(y)})


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    for ds_name, ds_cfg in DATASETS.items():
        X_train, y_train, X_test, y_test = ds_cfg["loader"]()

        le = LabelEncoder()
        le.fit(list(set(y_train)))
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        n_classes = len(le.classes_)

        train_set = prepare_dataset(X_train, y_train)
        test_set = prepare_dataset(X_test, y_test)

        results_path = os.path.join(RESULTS_DIR, f"{ds_name}_results.txt")
        if not os.path.exists(results_path):
            with open(results_path, "w") as f:
                f.write(f"Dataset: {ds_name}\n{'=' * 60}\n\n")

        for model_name in ds_cfg["models"]:
            print(f"\n{'=' * 60}")
            print(f"  Dataset: {ds_name} | Finetuning: {model_name}")
            print(f"{'=' * 60}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=n_classes
            )

            tokenized_train = train_set.map(tokenize_function, batched=True)
            tokenized_test = test_set.map(tokenize_function, batched=True)

            args = TrainingArguments(
                output_dir=f"output/{ds_name}/{model_name.split('/')[-1]}",
                eval_strategy="epoch",
                num_train_epochs=NUM_EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                compute_metrics=compute_metrics,
            )

            start = time.time()
            trainer.train()
            elapsed = time.time() - start

            eval_results = trainer.evaluate()

            metrics_str = (
                f"  accuracy: {eval_results['eval_accuracy']}\n"
                f"  recall: {eval_results['eval_recall']}\n"
                f"  f1-score: {eval_results['eval_f1']}"
            )

            print(f"\nModel: {model_name}")
            print(f"  samples: {len(X_test)}")
            print(f"  training_time: {elapsed:.1f}s")
            print(metrics_str)

            with open(results_path, "a") as f:
                f.write(
                    f"Model: {model_name}\n"
                    f"  samples: {len(X_test)}\n"
                    f"  training_time: {elapsed:.1f}s\n"
                    f"  epochs: {NUM_EPOCHS}\n"
                    f"  batch_size: {BATCH_SIZE}\n"
                    f"  learning_rate: {LEARNING_RATE}\n"
                    f"{metrics_str}\n\n"
                )

            print(f"Saved to: {results_path}")
