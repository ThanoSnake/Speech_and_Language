#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

utils/prepare_lang.sh \
    --sil-prob 0.0 \
    --position-dependent-phones false \
    --num-sil-states 3 \
    data/local/dict \
    "sil" \
    data/local/lang_tmp \
    data/lang
