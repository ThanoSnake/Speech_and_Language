# Embeddings

This directory contains pre-trained GloVe word embeddings (excluded from Git due to large file sizes).

## Embeddings Used

- **GloVe Twitter 27B** — Pre-trained on 2 billion tweets, 27 billion tokens
  - `glove.twitter.27B.25d.txt` (25 dimensions)
  - `glove.twitter.27B.50d.txt` (50 dimensions) — **default** used in `main.py`
  - `glove.twitter.27B.100d.txt` (100 dimensions)
  - `glove.twitter.27B.200d.txt` (200 dimensions)

## How to Download

1. Download from the [GloVe project page](https://nlp.stanford.edu/projects/glove/):
   ```
   wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
   ```
2. Extract into this directory:
   ```
   unzip glove.twitter.27B.zip -d embeddings/
   ```
