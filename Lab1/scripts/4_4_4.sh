#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

for lm in ug bg; do
    for x in dev test; do
        cat exp/mono/decode_${x}_${lm}/scoring_kaldi/best_wer
    done
done
