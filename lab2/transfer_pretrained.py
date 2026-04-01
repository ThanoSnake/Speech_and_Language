import os
import time

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import pipeline

from training import get_metrics_report
from utils.load_datasets import load_MR, load_Semeval2017A

RESULTS_DIR = "./results/pretrained"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    "Semeval2017A": load_Semeval2017A,
    "MR": load_MR,
}

MODELS = {
    "Semeval2017A": [
        "cardiffnlp/twitter-roberta-base-sentiment",
        "finiteautomata/bertweet-base-sentiment-analysis",
        "j-hartmann/sentiment-roberta-large-english-3-classes",
    ],
    "MR": [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "siebert/sentiment-roberta-large-english",
        "textattack/bert-base-uncased-SST-2",
    ],
}

# Map each model's output labels to canonical: positive, negative, neutral
LABELS_MAPPING = {
    "cardiffnlp/twitter-roberta-base-sentiment": {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    },
    "distilbert-base-uncased-finetuned-sst-2-english": {
        "POSITIVE": "positive",
        "NEGATIVE": "negative",
    },
    "j-hartmann/sentiment-roberta-large-english-3-classes": {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    },
    "finiteautomata/bertweet-base-sentiment-analysis": {
        "POS": "positive",
        "NEG": "negative",
        "NEU": "neutral",
    },
    "siebert/sentiment-roberta-large-english": {
        "POSITIVE": "positive",
        "NEGATIVE": "negative",
    },
    "textattack/bert-base-uncased-SST-2": {
        "LABEL_0": "negative",
        "LABEL_1": "positive",
    },
}

if __name__ == "__main__":
    for ds_name, loader in DATASETS.items():
        X_train, y_train, X_test, y_test = loader()

        le = LabelEncoder()
        le.fit(list(set(y_train)))
        y_test_enc = le.transform(y_test)
        valid_labels = set(le.classes_)

        results_path = os.path.join(RESULTS_DIR, f"{ds_name}_results.txt")

        # Write header only if file doesn't exist yet
        if not os.path.exists(results_path):
            with open(results_path, "w") as f:
                f.write(f"Dataset: {ds_name}\n{'=' * 60}\n\n")

        for model_name in MODELS[ds_name]:
            print(f"\n{'=' * 60}")
            print(f"  Dataset: {ds_name} | Model: {model_name}")
            print(f"{'=' * 60}")

            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                truncation=True,
            )

            y_pred = []
            start = time.time()
            for x in tqdm(X_test):
                out = sentiment_pipeline(x)
                label = out[0]["label"]
                mapped = LABELS_MAPPING[model_name][label]
                y_pred.append(mapped)
            elapsed = time.time() - start

            n_samples = len(X_test)
            throughput = n_samples / elapsed

            y_pred_enc = le.transform(y_pred)
            metrics = get_metrics_report([y_test_enc], [y_pred_enc])

            print(f"\nModel: {model_name}")
            print(f"  samples: {n_samples}")
            print(f"  time: {elapsed:.1f}s ({throughput:.1f} samples/s)")
            print(metrics)

            # Append each model's results immediately
            with open(results_path, "a") as f:
                f.write(
                    f"Model: {model_name}\n"
                    f"  samples: {n_samples}\n"
                    f"  time: {elapsed:.1f}s ({throughput:.1f} samples/s)\n"
                    f"{metrics}\n\n"
                )

            print(f"Saved to: {results_path}")
