#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz --eval=data/local/dict/lm_dev.text
compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz --eval=data/local/dict/lm_test.text

compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz --eval=data/local/dict/lm_dev.text
compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz --eval=data/local/dict/lm_test.text