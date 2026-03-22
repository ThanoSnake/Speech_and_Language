#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh
. ./cmd.sh

for x in dev test; do
    for lm in ug bg; do
        mkdir -p exp/dnn/decode_${x}_${lm}
        graph=exp/tri1/graph_${lm}
        latgen-faster-mapped --acoustic-scale=0.1 --beam=15.0 --lattice-beam=8.0 --word-symbol-table=$graph/words.txt exp/tri1/final.mdl $graph/HCLG.fst "ark:exp/dnn/posteriors_${x}.ark" "ark:|gzip -c > exp/dnn/decode_${x}_${lm}/lat.1.gz"
        local/score.sh data/${x} $graph exp/dnn/decode_${x}_${lm}
    done
done

for x in dev test; do
    for lm in ug bg; do
        cat exp/dnn/decode_${x}_${lm}/scoring_kaldi/best_wer
    done
done
