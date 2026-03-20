#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for x in train dev test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 \
        data/$x exp/make_mfcc/$x mfcc
done
