import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from attention import (
    MultiHeadAttentionModel,
    SimpleSelfAttentionModel,
    TransformerEncoderModel,
)
from config import EMB_PATH
from dataloading import MAX_SENTENCE_LEN, SentenceDataset
from early_stopper import EarlyStopper
from models import LSTM, BaselineDNN
from training import (
    eval_dataset,
    get_metrics_report,
    torch_train_val_split,
    train_dataset,
)
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################

DEBUG = True  # set False to silence all debug/diagnostic prints

EMBEDDINGS = os.path.join(
    EMB_PATH,
    "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt",
)
EMB_DIM = 50
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5
VAL_SIZE = 0.2
RESULTS_DIR = "./results_v2"
MODELS_DIR = "./models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

########################################################
# Load embeddings (shared across datasets)
########################################################

word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

########################################################
# Helpers
########################################################

DATASETS = {
    "MR": {
        "loader": load_MR,
        "criterion": nn.BCEWithLogitsLoss(),
        "output_dim": 1,
    },
    "Semeval2017A": {
        "loader": load_Semeval2017A,
        "criterion": nn.CrossEntropyLoss(),
        "output_dim": 3,
    },
}


def prepare_dataset(name):
    """Load, encode, and wrap a dataset. Returns train_set, test_loader, criterion, output_dim."""
    cfg = DATASETS[name]
    X_train, y_train, X_test, y_test = cfg["loader"]()

    # EX1 -- label encoding
    le = LabelEncoder()
    le.fit(y_train)
    y_train_raw = list(y_train[:10])  # save before overwriting
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    if DEBUG:
        print(f"\n[{name}] EX1 -- Label encoding")
        print(f"  Classes:             {le.classes_}")
        print(f"  First 10 raw labels: {y_train_raw}")
        print(f"  First 10 encoded:    {y_train[:10]}")

    train_set = SentenceDataset(X_train, y_train, word2idx)
    test_set = SentenceDataset(X_test, y_test, word2idx)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    return train_set, test_loader, cfg["criterion"], cfg["output_dim"]


def train_single(model, train_set, test_loader, criterion, epochs, save_path):
    """Train a model and return losses, final metrics string, and epochs run."""
    model.to(DEVICE)
    print(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)

    train_loader, val_loader = torch_train_val_split(
        train_set, BATCH_SIZE, BATCH_SIZE, val_size=VAL_SIZE
    )

    train_losses, val_losses, test_losses = [], [], []
    stopper = EarlyStopper(model, save_path, patience=PATIENCE)

    for epoch in range(1, epochs + 1):
        train_dataset(epoch, train_loader, model, criterion, optimizer)
        train_loss, _ = eval_dataset(train_loader, model, criterion)
        val_loss, _ = eval_dataset(val_loader, model, criterion)
        test_loss, _ = eval_dataset(test_loader, model, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        if stopper.early_stop(val_loss):
            break

    epochs_run = len(train_losses)
    model.load_state_dict(torch.load(save_path))
    _, (y_gold, y_pred) = eval_dataset(test_loader, model, criterion)
    metrics = get_metrics_report(y_gold, y_pred)
    print(f"\n{metrics}")

    return train_losses, val_losses, test_losses, metrics, epochs_run, y_gold, y_pred


def run_experiment(label, model_factory, results_name=None, reps=1):
    """Run a model on both datasets, save side-by-side plot and append to results file.

    label:        used for the plot filename and the run header (e.g. 'multihead_attention_h5')
    model_factory: callable(output_dim) -> model
    results_name: base name for the results txt file (defaults to label).
                  Multiple runs can share the same results_name to accumulate entries.
    reps:         number of independent runs to average over.
    """
    if results_name is None:
        results_name = label

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    results_sections = []
    model_arch = None

    for i, ds_name in enumerate(DATASETS):
        train_set, test_loader, criterion, output_dim = prepare_dataset(ds_name)
        save_path = os.path.join(MODELS_DIR, f"{label}_{ds_name}")

        accs, recs, f1s, epochs_list = [], [], [], []
        train_losses = val_losses = test_losses = (
            None  # will keep last rep for plotting
        )

        for rep in range(reps):
            print(f"\n{'=' * 60}")
            print(f"  {label} — {ds_name}  [rep {rep + 1}/{reps}]")
            print(f"{'=' * 60}")

            model = model_factory(output_dim)

            if model_arch is None:
                model_arch = str(model)
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )

            train_losses, val_losses, test_losses, _, epochs_run, y_gold, y_pred = (
                train_single(
                    model, train_set, test_loader, criterion, EPOCHS, save_path
                )
            )
            epochs_list.append(epochs_run)

            yg = np.concatenate(y_gold, axis=0)
            yp = np.concatenate(y_pred, axis=0)
            accs.append(accuracy_score(yg, yp))
            recs.append(recall_score(yg, yp, average="macro"))
            f1s.append(f1_score(yg, yp, average="macro"))

        # Aggregate across reps
        def fmt(vals):
            m = np.mean(vals)
            return f"{m:.10f}" if reps == 1 else f"{m:.10f} +/- {np.std(vals):.10f}"

        metrics_summary = (
            f"  reps: {reps}\n"
            f"  avg_epochs_run: {np.mean(epochs_list):.1f}/{EPOCHS}\n"
            f"  accuracy: {fmt(accs)}\n"
            f"  recall: {fmt(recs)}\n"
            f"  f1-score: {fmt(f1s)}"
        )

        ax = axes[i]
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{ds_name}")
        ax.legend()

        results_sections.append(f"Dataset: {ds_name}\n{metrics_summary}")

    fig.suptitle(f"Training Curves — {label}", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"loss_{label}.png")
    fig.savefig(plot_path)
    plt.close(fig)

    header = (
        f"{'=' * 60}\n"
        f"timestamp: {timestamp}\n"
        f"label: {label}\n"
        f"embeddings: {os.path.basename(EMBEDDINGS)} (dim={EMB_DIM})\n"
        f"emb_trainable: {EMB_TRAINABLE}\n"
        f"max_sentence_length: {MAX_SENTENCE_LEN}\n"
        f"batch_size: {BATCH_SIZE}\n"
        f"max_epochs: {EPOCHS}\n"
        f"learning_rate: {LEARNING_RATE}\n"
        f"optimizer: Adam\n"
        f"patience: {PATIENCE}\n"
        f"val_size: {VAL_SIZE}\n"
        f"device: {DEVICE}\n"
        f"total_params: {total_params:,}\n"
        + (
            f"trainable_params: {trainable_params:,}\n"
            if trainable_params != total_params
            else ""
        )
        + f"\narchitecture:\n{model_arch}\n"
    )

    results_path = os.path.join(RESULTS_DIR, f"{results_name}_results.txt")
    with open(results_path, "a") as f:
        f.write(header + "\n" + "\n\n".join(results_sections) + "\n")

    print(f"\nSaved plot: {plot_path}")
    print(f"Saved results: {results_path}")


#############################################################################
# Model factories
#############################################################################


def make_baseline(output_dim):
    return BaselineDNN(
        output_size=output_dim, embeddings=embeddings, trainable_emb=EMB_TRAINABLE
    )


def make_baseline_maxconcat(output_dim):
    return BaselineDNN(
        output_size=output_dim,
        embeddings=embeddings,
        trainable_emb=EMB_TRAINABLE,
        max_concat=True,
    )


def make_lstm(output_dim):
    return LSTM(
        output_size=output_dim,
        embeddings=embeddings,
        trainable_emb=EMB_TRAINABLE,
        bidirectional=False,
    )


def make_bilstm(output_dim):
    return LSTM(
        output_size=output_dim,
        embeddings=embeddings,
        trainable_emb=EMB_TRAINABLE,
        bidirectional=True,
    )


def make_self_attention_factory(head_scale=1):
    def factory(output_dim):
        return SimpleSelfAttentionModel(
            output_size=output_dim,
            embeddings=embeddings,
            head_scale=head_scale,
        )

    return factory


def make_multihead_attention_factory(n_head, head_scale=1):
    def factory(output_dim):
        return MultiHeadAttentionModel(
            output_size=output_dim,
            embeddings=embeddings,
            n_head=n_head,
            head_scale=head_scale,
        )

    return factory


def make_transformer_factory(n_head, n_layer):
    def factory(output_dim):
        return TransformerEncoderModel(
            output_size=output_dim,
            embeddings=embeddings,
            max_length=MAX_SENTENCE_LEN,
            n_head=n_head,
            n_layer=n_layer,
        )

    return factory


#############################################################################
# Run experiments — uncomment the ones you want to run
#############################################################################

REP = 15

# run_experiment("baseline", make_baseline, reps=REP)
# run_experiment("baseline_maxconcat", make_baseline_maxconcat, reps=REP)
# run_experiment("lstm", make_lstm, reps=REP)
run_experiment("bilstm", make_bilstm, reps=REP)

# run_experiment("self_attention", make_self_attention_factory(), reps=REP)

# for n_head in [2, 5, 10]:
#     run_experiment(
#         label=f"multihead_attention_h{n_head}",
#         model_factory=make_multihead_attention_factory(n_head),
#         results_name="multihead_attention",
#         reps=10,
#     )


# for n_head in [2, 5, 10]:
#     for n_layer in [2, 3, 5, 7, 10, 12]:
#         run_experiment(
#             label=f"transformer_h{n_head}_l{n_layer}",
#             model_factory=make_transformer_factory(n_head, n_layer),
#             results_name="transformer",
#             reps=5,
#         )

# Close-to-default configuration: h=10 (nearest divisor of 50 to original h=8),
# N=6 (exact match to original), d_ff=4*d=200 (same ratio as original 2048/512).
# run_experiment(
#     "transformer_default_h10_l6",
#     make_transformer_factory(10, 6),
#     results_name="transformer",
#     reps=5,
# )
