import os
import warnings
from tkinter import Label

import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import eval_dataset, train_dataset
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

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
print("\n=== EX1: Label Encoding ===")
print(f"Dataset: {DATASET} | Classes: {sorted(set(y_train))}")
print("First 10 raw labels:")
print(y_train[0:10])

le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
n_classes = le.classes_.size

print(f"Label mapping: { {label: idx for idx, label in enumerate(le.classes_)} }")
print("First 10 encoded labels:")
print(y_train[0:10])
print(f"Number of classes: {n_classes}\n")

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# EX3 - Print 5 examples in original form and as returned by SentenceDataset
print("\n=== EX3: Example Encoding (first 5 training examples) ===")
print(f"max_length={train_set.max_length} | 0-padded if short, truncated if long | <unk> index={word2idx['<unk>']}")
print()
for i in range(5):
    print(f"[{i}] Original  : {X_train[i]}")
    print(f"[{i}] Tokenized : {train_set.data[i]}")
    example, label, length = train_set[i]
    print(f"[{i}] Encoded   : {example}")
    print(f"[{i}] Label     : {label} ({le.classes_[label]}) | Real length: {length}")
    print()

# # EX7 - Define our PyTorch-based DataLoader
# train_loader = ...  # EX7
# test_loader = ...  # EX7

# #############################################################################
# # Model Definition (Model, Loss Function, Optimizer)
# #############################################################################
# model = BaselineDNN(output_size=...,  # EX8
#                     embeddings=embeddings,
#                     trainable_emb=EMB_TRAINABLE)

# # move the mode weight to cpu or gpu
# model.to(DEVICE)
# print(model)

# # We optimize ONLY those parameters that are trainable (p.requires_grad==True)
# criterion = ...  # EX8
# parameters = ...  # EX8
# optimizer = ...  # EX8

# #############################################################################
# # Training Pipeline
# #############################################################################
# for epoch in range(1, EPOCHS + 1):
#     # train the model for one epoch
#     train_dataset(epoch, train_loader, model, criterion, optimizer)

#     # evaluate the performance of the model, on both data sets
#     train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
#                                                             model,
#                                                             criterion)

#     test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
#                                                          model,
#                                                          criterion)
