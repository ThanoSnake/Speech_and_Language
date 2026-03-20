#!/bin/bash
# 4.4.2 - Δημιουργία γράφου HCLG για monophone μοντέλο
# Δημιουργούνται γράφοι για unigram και bigram γλωσσικό μοντέλο
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for lm in ug bg; do
    utils/mkgraph.sh \
        data/lang_test_${lm} \
        exp/mono \
        exp/mono/graph_${lm}
    echo "HCLG graph created: exp/mono/graph_${lm}/"
done
