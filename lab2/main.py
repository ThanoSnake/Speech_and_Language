import os
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
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
RESULTS_DIR = "./results"
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

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

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

    return train_losses, val_losses, test_losses, metrics, epochs_run


def run_experiment(model_name, model_factory):
    """Run a model on both datasets, save side-by-side plot and combined results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    results_sections = []
    model_arch = None

    for i, ds_name in enumerate(DATASETS):
        print(f"\n{'=' * 60}")
        print(f"  {model_name} — {ds_name}")
        print(f"{'=' * 60}")

        train_set, test_loader, criterion, output_dim = prepare_dataset(ds_name)
        save_path = os.path.join(MODELS_DIR, f"{model_name}_{ds_name}")
        model = model_factory(output_dim)

        if model_arch is None:
            model_arch = str(model)

        train_losses, val_losses, test_losses, metrics, epochs_run = train_single(
            model, train_set, test_loader, criterion, EPOCHS, save_path
        )

        # Plot on subplot
        ax = axes[i]
        ax.plot(train_losses, label="Train Loss")
        ax.plot(val_losses, label="Val Loss")
        ax.plot(test_losses, label="Test Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{ds_name}")
        ax.legend()

        results_sections.append(
            f"Dataset: {ds_name}\n  epochs_run: {epochs_run}/{EPOCHS}\n{metrics}"
        )

    fig.suptitle(f"Training Curves — {model_name}", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, f"loss_{model_name}.png")
    fig.savefig(plot_path)
    plt.close(fig)

    header = (
        f"model: {model_name}\n"
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
        f"\n"
        f"architecture:\n{model_arch}\n"
    )

    results_path = os.path.join(RESULTS_DIR, f"{model_name}_results.txt")
    with open(results_path, "w") as f:
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


def make_self_attention(output_dim):
    return SimpleSelfAttentionModel(
        output_size=output_dim,
        embeddings=embeddings,
    )


def make_multihead_attention(output_dim):
    return MultiHeadAttentionModel(
        output_size=output_dim,
        embeddings=embeddings,
        n_head=5,
    )


def make_transformer(output_dim):
    return TransformerEncoderModel(
        output_size=output_dim,
        embeddings=embeddings,
        max_length=MAX_SENTENCE_LEN,
        n_head=5,
        n_layer=5,
    )


#############################################################################
# Run experiments — uncomment the ones you want to run
#############################################################################

run_experiment("baseline", make_baseline)
# run_experiment("baseline_maxconcat", make_baseline_maxconcat)
# run_experiment("lstm", make_lstm)
# run_experiment("bilstm", make_bilstm)
# run_experiment("self_attention", make_self_attention)
# run_experiment("multihead_attention", make_multihead_attention)
# run_experiment("transformer", make_transformer)
