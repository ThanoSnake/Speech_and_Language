#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

for lm_suffix in ug bg; do
    test_dir=data/lang_test_${lm_suffix}
    mkdir -p $test_dir
    cp -r data/lang/* $test_dir/

    gunzip -c data/local/nist_lm/lm_phone_${lm_suffix}.arpa.gz | \
        arpa2fst --disambig-symbol=#0 \
                 --read-symbol-table=$test_dir/words.txt \
                 - $test_dir/G.fst

    fstisstochastic $test_dir/G.fst
done