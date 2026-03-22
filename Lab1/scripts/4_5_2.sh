#!/bin/bash
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

mkdir -p exp/dnn_feats

for x in train dev test; do
    apply-cmvn --utt2spk=ark:data/${x}/utt2spk scp:data/${x}/cmvn.scp scp:data/${x}/feats.scp ark:exp/dnn_feats/${x}_tmp.ark
    add-deltas ark:exp/dnn_feats/${x}_tmp.ark ark,scp:exp/dnn_feats/${x}.ark,exp/dnn_feats/${x}.scp
    rm exp/dnn_feats/${x}_tmp.ark
done
