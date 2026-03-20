#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

for x in train dev test; do
    utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
done
