#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for lm in ug bg; do
    for x in dev test; do
        steps/decode.sh --cmd "$decode_cmd" --nj 4 exp/mono/graph_${lm} data/$x exp/mono/decode_${x}_${lm}
    done
done
