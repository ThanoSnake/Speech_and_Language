#!/bin/bash
# 4.2.6 - Δημιουργία spk2utt από utt2spk
# Αντίστροφη αντιστοίχηση: speaker -> list of utterances
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

for split in train dev test; do
    utils/utt2spk_to_spk2utt.pl data/$split/utt2spk > data/$split/spk2utt
    echo "Created data/$split/spk2utt"
done
