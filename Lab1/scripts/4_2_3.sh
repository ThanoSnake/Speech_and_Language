#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

compile-lm data/local/lm_tmp/lm_phone_ug.ilm.gz -t=yes /dev/stdout \
    | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_ug.arpa.gz

compile-lm data/local/lm_tmp/lm_phone_bg.ilm.gz -t=yes /dev/stdout \
    | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_bg.arpa.gz
