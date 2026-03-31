import os
import warnings
from tkinter import Label

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from attention import MultiHeadAttentionModel, SimpleSelfAttentionModel
from config import EMB_PATH
from dataloading import SentenceDataset
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


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate(
    model,
    train_set,
    test_loader,
    criterion,
    epochs,
    save_path="./best_model",
    plot_title=None,
    plot_save=None,
):
    model.to(DEVICE)
    print(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    train_loader, val_loader = torch_train_val_split(
        train_set, BATCH_SIZE, BATCH_SIZE, val_size=0.2
    )

    train_losses = []
    val_losses = []
    test_losses = []
    stopper = EarlyStopper(model, save_path, patience=5)

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

    model.load_state_dict(torch.load(save_path))
    _, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, criterion)

    report = f"Dataset: {DATASET}\n{get_metrics_report(y_test_gold, y_test_pred)}"
    print(f"\n\n{report}")

    results_file = save_path + "_results.txt"
    with open(results_file, "w") as f:
        f.write(report + "\n")

    if plot_title:
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.plot(test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(plot_title)
        plt.legend()
        if plot_save:
            plt.savefig(plot_save)
        plt.show()

    return train_losses, val_losses, test_losses


########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
# print("\n=== EX1: Label Encoding ===")
# print(f"Dataset: {DATASET} | Classes: {sorted(set(y_train))}")
# print("First 10 raw labels:")
# print(y_train[0:10])

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
n_classes = le.classes_.size

# print(f"Label mapping: { {label: idx for idx, label in enumerate(le.classes_)} }")
# print("First 10 encoded labels:")
# print(y_train[0:10])
# print(f"Number of classes: {n_classes}\n")

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# EX3 - Print 5 examples in original form and as returned by SentenceDataset
# print("\n=== EX3: Example Encoding (first 5 training examples) ===")
# print(
#     f"max_length={train_set.max_length} | 0-padded if short, truncated if long | <unk> index={word2idx['<unk>']}"
# )
# print()
# for i in range(5):
#     print(f"[{i}] Original  : {X_train[i]}")
#     print(f"[{i}] Tokenized : {train_set.data[i]}")
#     example, label, length = train_set[i]
#     print(f"[{i}] Encoded   : {example}")
#     print(f"[{i}] Label     : {label} ({le.classes_[label]}) | Real length: {length}")
#     print()

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

if DATASET == "MR":
    criterion = nn.BCEWithLogitsLoss()
    output_dim = 1  # single logit for binary classification
elif DATASET == "Semeval2017A":
    criterion = nn.CrossEntropyLoss()
    output_dim = 3  # 3 classes: positive, negative, neutral
else:
    raise ValueError("Invalid dataset")

#############################################################################
# Baseline DNN (mean pooling)
#############################################################################
print("\n=== Baseline DNN (mean pooling) ===")
model = BaselineDNN(
    output_size=output_dim, embeddings=embeddings, trainable_emb=EMB_TRAINABLE
)
train_and_evaluate(
    model,
    train_set,
    test_loader,
    criterion,
    EPOCHS,
    "./best_model",
    plot_title=f"Training Curves - {DATASET} (mean)",
    plot_save=f"loss_curve_{DATASET}.png",
)

#############################################################################
# Q1 - DNN with mean + max pooling concatenation
#############################################################################
# print("\n=== Q1: DNN (mean + max pooling) ===")
# model_q1 = BaselineDNN(
#     output_size=output_dim, embeddings=embeddings,
#     trainable_emb=EMB_TRAINABLE, max_concat=True,
# )
# train_and_evaluate(model_q1, train_set, test_loader, criterion, EPOCHS, "./best_model_q1",
#     plot_title=f"Training Curves - {DATASET} (mean+max)", plot_save=f"loss_curve_{DATASET}_q1.png")

#############################################################################
# Q2 - LSTM
#############################################################################
# print("\n=== Q2.2: LSTM (unidirectional) ===")
# model_lstm = LSTM(
#     output_size=output_dim,
#     embeddings=embeddings,
#     trainable_emb=EMB_TRAINABLE,
#     bidirectional=False,
# )
# train_and_evaluate(
#     model_lstm,
#     train_set,
#     test_loader,
#     criterion,
#     EPOCHS,
#     "./best_model_q2",
#     plot_title=f"Training Curves - {DATASET} (LSTM)",
#     plot_save=f"loss_curve_{DATASET}_q2.png",
# )

# print("\n=== Q2.3: BiLSTM ===")
# model_bilstm = LSTM(
#     output_size=output_dim,
#     embeddings=embeddings,
#     trainable_emb=EMB_TRAINABLE,
#     bidirectional=True,
# )
# train_and_evaluate(
#     model_bilstm,
#     train_set,
#     test_loader,
#     criterion,
#     EPOCHS,
#     "./best_model_q2_bi",
#     plot_title=f"Training Curves - {DATASET} (BiLSTM)",
#     plot_save=f"loss_curve_{DATASET}_q2_bi.png",
# )

#############################################################################
# Q3 - Self-Attention
#############################################################################
# print("\n=== Q3.1: Self-Attention ===")
# model_sa = SimpleSelfAttentionModel(
#     output_size=output_dim,
#     embeddings=embeddings,
# )
# train_and_evaluate(
#     model_sa,
#     train_set,
#     test_loader,
#     criterion,
#     EPOCHS,
#     "./best_model_q3",
#     plot_title=f"Training Curves - {DATASET} (Self-Attention)",
#     plot_save=f"loss_curve_{DATASET}_q3.png",
# )

#############################################################################
# Q4 - MultiHead Attention
#############################################################################
# print("\n=== Q4: MultiHead Attention ===")
# model_mha = MultiHeadAttentionModel(
#     output_size=output_dim,
#     embeddings=embeddings,
#     n_head=5,
# )
# train_and_evaluate(
#     model_mha,
#     train_set,
#     test_loader,
#     criterion,
#     EPOCHS,
#     "./best_model_q4",
#     plot_title=f"Training Curves - {DATASET} (MultiHead Attention)",
#     plot_save=f"loss_curve_{DATASET}_q4.png",
# )
