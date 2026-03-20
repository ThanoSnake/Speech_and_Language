#!/bin/bash
# 4.2.4 - Δημιουργία L.fst με prepare_lang.sh
# Παράμετροι για phone recognition: χωρίς position-dependent phones, sil-prob=0
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

utils/prepare_lang.sh \
    --sil-prob 0.0 \
    --position-dependent-phones false \
    --num-sil-states 3 \
    data/local/dict \
    "sil" \
    data/local/lang_tmp \
    data/lang

echo "L.fst created in data/lang/"
