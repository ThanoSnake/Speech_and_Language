#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

for x in train dev test; do
    steps/compute_cmvn_stats.sh \
        data/$x exp/make_mfcc/$x mfcc
done
