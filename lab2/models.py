import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # Μετατροπή των embeddings σε PyTorch Tensor (αν είναι numpy array)
        embeddings = torch.tensor(embeddings, dtype=torch.float)
        num_embeddings, emb_dim = embeddings.shape

        # EX4 - 1, 2 & 3: Ορισμός, Αρχικοποίηση και "Πάγωμα" του Embedding Layer
        # Η μέθοδος from_pretrained κάνει και τα 3 βήματα μαζί:
        # - Δημιουργεί το layer με τις σωστές διαστάσεις
        # - Φορτώνει τα βάρη από τον πίνακα embeddings
        # - Παγώνει (freeze=True) ή αφήνει τα βάρη να εκπαιδευτούν (freeze=False)
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=not trainable_emb)

        # 1 - define the embedding layer
        ...  # EX4

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        ...  # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        ...  # EX4

        # 4 - define a non-linear transformation of the representations
        ...  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        ...  # EX5

        # Σημείωση: Το emb_dim το παίρνουμε από το σχήμα των embeddings
        #emb_dim = embeddings.shape[1]
        hidden_dim = 100  # Επιλογή μας (π.χ. 64, 100, 128)

        # EX5 - 4: Μη γραμμικός μετασχηματισμός
        # Ένα Linear layer ακολουθούμενο από μια συνάρτηση ενεργοποίησης (ReLU)
        self.feature_transform = nn.Linear(emb_dim, hidden_dim)
        self.relu = nn.ReLU()

        # EX5 - 5: Τελικό Linear layer (Classification)
        # Προβολή από το hidden_dim στον αριθμό των κλάσεων (output_size)
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # 1 - embed the words, using the embedding layer
        #embeddings = ...  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        #representations = ...  # EX6

        # 3 - transform the representations to new ones.
        #representations = ...  # EX6

        # 4 - project the representations to classes using a linear layer
        #logits = ...  # EX6

        # 1 - Μετατροπή των indices σε embeddings
        # Είσοδος x: [batch_size, max_length]
        # Έξοδος embeddings: [batch_size, max_length, emb_dim]
        embeddings = self.embedding(x)

        # 2 - Δημιουργία αναπαράστασης πρότασης (Mean Pooling)
        # ΠΡΟΣΟΧΗ: Πρέπει να διαιρέσουμε με το πραγματικό μήκος (lengths), όχι το max_length.
        
        # Άθροισμα των embeddings κατά μήκος των λέξεων (dim=1)
        # sum_embeddings shape: [batch_size, emb_dim]
        sum_embeddings = torch.sum(embeddings, dim=1)
        
        # Μετατρέπουμε τα lengths σε float και τους δίνουμε σωστό σχήμα για τη διαίρεση
        # lengths.view(-1, 1) μετατρέπει το [batch_size] σε [batch_size, 1]
        representations = sum_embeddings / lengths.view(-1, 1).float()

        # 3 - Μη γραμμικός μετασχηματισμός (από το EX5)
        representations = self.feature_transform(representations)
        representations = self.relu(representations)

        # 4 - Τελική προβολή στις κλάσεις (Logits)
        logits = self.output_layer(representations)

        return logits


class LSTM(nn.Module):
    def __init__(self, output_size, embeddings, trainable_emb=False, bidirectional=False):

        super(LSTM, self).__init__()
        self.hidden_size = 100
        self.num_layers = 1
        self.bidirectional = bidirectional

        self.representation_size = 2 * \
            self.hidden_size if self.bidirectional else self.hidden_size

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.embeddings = nn.Embedding(num_embeddings, dim)
        self.output_size = output_size

        self.lstm = nn.LSTM(dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)

        if not trainable_emb:
            self.embeddings = self.embeddings.from_pretrained(
                torch.Tensor(embeddings), freeze=True)

        self.linear = nn.Linear(self.representation_size, output_size)

    def forward(self, x, lengths):
        batch_size, max_length = x.shape
        embeddings = self.embeddings(x)
        X = torch.nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False)

        ht, _ = self.lstm(X)

        # ht is batch_size x max(lengths) x hidden_dim
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # pick the output of the lstm corresponding to the last word
        # TODO: Main-Lab-Q2 (Hint: take actual lengths into consideration)
        representations = ...

        logits = self.linear(representations)

        return logits
