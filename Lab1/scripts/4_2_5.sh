#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

for x in train dev test; do
    sort data/$x/wav.scp -o data/$x/wav.scp
    sort data/$x/text -o data/$x/text
    sort data/$x/utt2spk -o data/$x/utt2spk
done
