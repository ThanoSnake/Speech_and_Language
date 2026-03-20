#!/bin/bash
# 4.1.3 - Δημιουργία φακέλου local με symlinks στο score_kaldi.sh
# Το score.sh απαιτείται από το steps/decode.sh
# Εκτέλεση από: ~/kaldi/egs/usc/

. ./path.sh

mkdir -p local
ln -sf $KALDI_ROOT/egs/wsj/s5/steps/score_kaldi.sh local/score_kaldi.sh
ln -sf $KALDI_ROOT/egs/wsj/s5/steps/score_kaldi.sh local/score.sh

echo "Created local/score_kaldi.sh and local/score.sh"
