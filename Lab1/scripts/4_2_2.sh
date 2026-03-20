#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

# Unigram
build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/lm_phone_ug.ilm.gz

# Bigram
build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/lm_phone_bg.ilm.gz
