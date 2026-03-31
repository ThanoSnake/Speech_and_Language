from torch.utils.data import Dataset
from tqdm import tqdm


class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """

        # self.data = X
        # self.labels = y
        # self.word2idx = word2idx

        # EX2
        # EX2 - Initialization and Tokenization
        self.word2idx = word2idx
        self.labels = y
        
        # Λίστα για να αποθηκεύσουμε τα tokenized κείμενα
        self.data = []
        
        print("Tokenizing data...")
        for sentence in tqdm(X):
            # Απλός διαχωρισμός με βάση το κενό και μετατροπή σε πεζά
            # Μπορείς να χρησιμοποιήσεις .lower() για να ταυτίζονται οι λέξεις
            tokens = sentence.lower().split()
            self.data.append(tokens)

        # Εκτύπωση των πρώτων 10 παραδειγμάτων (Ζητούμενο 2)
        print("\nFirst 10 tokenized examples:")
        for i in range(10):
            print(f"{i+1}: {self.data[i]}")

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """

        # EX3
    def __getitem__(self, index):
        # EX3 - Get a single example and its label
        tokens = self.data[index]
        label = self.labels[index]
        
        # Ορίζουμε το μέγιστο μήκος (max_length)
        # Για το MR (ταινίες) το 50-60 είναι καλό, για Twitter το 50 είναι υπέρ αρκετό.
        max_length = 50 

        # 1. Mapping tokens to IDs
        indices = []
        unk_index = self.word2idx.get("<unk>") # Παίρνουμε το ID του <unk> από το dictionary
        
        for word in tokens:
            # Αν η λέξη υπάρχει, πάρε το ID της. Αν όχι, πάρε το unk_index.
            indices.append(self.word2idx.get(word, unk_index))

        # Κρατάμε το πραγματικό μήκος πριν το padding
        # Αν η πρόταση είναι μεγαλύτερη από το max_length, το μήκος περιορίζεται
        real_length = min(len(indices), max_length)

        # 2. Padding / Truncating (Zero-padding)
        if len(indices) < max_length:
            # Συμπλήρωση με μηδενικά (0) στο τέλος
            indices += [0] * (max_length - len(indices))
        else:
            # Περικοπή (Truncation) αν ξεπερνά το μέγιστο μήκος
            indices = indices[:max_length]

        import numpy as np
        return np.array(indices), label, real_length
        # return example, label, length
        #raise NotImplementedError

