#!/bin/bash
# 4.2.7 - Δημιουργία G.fst (Grammar FST) από ARPA γλωσσικά μοντέλα
# Ακολουθεί την ίδια διαδικασία με τη διαδικασία timit του Kaldi
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for lm_suffix in ug bg; do
    test_dir=data/lang_test_${lm_suffix}
    mkdir -p $test_dir
    cp -r data/lang/* $test_dir/

    gunzip -c data/local/nist_lm/lm_phone_${lm_suffix}.arpa.gz | \
        arpa2fst --disambig-symbol=#0 \
                 --read-symbol-table=$test_dir/words.txt \
                 - $test_dir/G.fst

    fstisstochastic $test_dir/G.fst
    echo "G.fst created in $test_dir/"
done
