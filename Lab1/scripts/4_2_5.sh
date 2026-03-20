#!/bin/bash
# 4.2.5 - Ταξινόμηση αρχείων wav.scp, text, utt2spk (απαιτείται από Kaldi)
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

for split in train dev test; do
    sort -k1 data/$split/wav.scp  -o data/$split/wav.scp
    sort -k1 data/$split/text     -o data/$split/text
    sort -k1 data/$split/utt2spk  -o data/$split/utt2spk
    echo "Sorted data/$split/"
done
