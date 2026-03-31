import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset, get_metrics_report
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt")

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
le = LabelEncoder()
y_train = le.fit_transform(y_train) # EX1
y_test = le.transform(y_test) # EX1
n_classes = le.classes_.size # EX1 - LabelEncoder.classes_.size

# Εκτύπωση για την επαλήθευση του ζητουμένου
print(f"First 10 train labels: {y_train[:10]}")
for i, label in enumerate(le.classes_):
    print(f"Class '{label}' is mapped to integer: {i}")
  

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# main.py - Μετά τον ορισμό του train_set
print("\n--- EX3: Verification ---")
for i in range(5):
    # Παίρνουμε το αντικείμενο από το dataset
    example_indices, label, length = train_set[i]
    
    print(f"Example {i+1}:")
    print(f"  Original (Tokens): {train_set.data[i]}")
    print(f"  Encoded (Indices): {example_indices}")
    print(f"  Label ID: {label}, Real Length: {length}")
    print("-" * 30)

# EX7 - Define our PyTorch-based DataLoader
#train_loader = ...  # EX7
#test_loader = ...  # EX7
# Για το training set θέλουμε ανακάτεμα (shuffle=True)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
# Για το test set δεν χρειάζεται ανακάτεμα, θέλουμε απλώς να το αξιολογήσουμε
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

actual_output_size = 1 if n_classes == 2 else n_classes

model = BaselineDNN(output_size=actual_output_size,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
#criterion = ...  # EX8
#parameters = ...  # EX8
#optimizer = ...  # EX8

# EX8 - Κριτήριο (Loss Function)
# Αν έχουμε 2 κλάσεις χρησιμοποιούμε BCEWithLogitsLoss, αλλιώς CrossEntropyLoss
if n_classes == 2:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()

# EX8 - Παράμετροι προς βελτιστοποίηση
# Θέλουμε ΜΟΝΟ όσες έχουν requires_grad = True (δηλαδή όχι το παγωμένο embedding layer)
parameters = [p for p in model.parameters() if p.requires_grad]

# EX8 - Optimizer (Προτείνεται ο Adam)
optimizer = torch.optim.Adam(parameters, lr=1e-3)

#############################################################################
# Training Pipeline
#############################################################################
train_losses = []
test_losses = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    # --- Μετά το τέλος του Training Pipeline loop ---

# 1. Εκτύπωση των μετρικών (Accuracy, Recall, F1) για το Test Set
print(f"\n" + "="*30)
print(f"FINAL METRICS FOR {DATASET}")
print("="*30)
# Χρησιμοποιούμε τη συνάρτηση get_metrics_report που υπάρχει στο training.py
report = get_metrics_report(y_test_gold, y_test_pred)
print(report)

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
