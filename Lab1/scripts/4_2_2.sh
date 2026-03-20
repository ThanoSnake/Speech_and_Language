#!/bin/bash
# 4.2.2 - Δημιουργία ενδιάμεσων γλωσσικών μοντέλων με IRSTLM
# Δημιουργούνται unigram (n=1) και bigram (n=2) μοντέλα
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

export PATH=$PATH:$KALDI_ROOT/tools/irstlm/bin

# Unigram
build-lm.sh -i data/train/lm_train.text -n 1 \
    -o data/local/lm_tmp/lm_phone_ug.ilm.gz

# Bigram
build-lm.sh -i data/train/lm_train.text -n 2 \
    -o data/local/lm_tmp/lm_phone_bg.ilm.gz

echo "Language models built in data/local/lm_tmp/"
